import json, re, time, os
from typing import Annotated, TypedDict, Literal, Optional
from langchain_core.messages import HumanMessage, ToolMessage, SystemMessage, AIMessage
from langgraph.graph import StateGraph, END
from langgraph.graph.message import add_messages

import config
from tools.tools_k8s import K8S_TOOLS
from rag.retrieve import RAG_TOOLS, _MSG_NO_INGEST, _is_kb_topic
from agent.bypass import should_bypass_llm, build_direct_answer
from agent.routing import default_tools_for, resolve_namespace

_PROMPT_FILE = config._HERE / "config" / "system_prompt.txt"

def _load_system_prompt() -> str:
    if _PROMPT_FILE.exists():
        text = _PROMPT_FILE.read_text(encoding="utf-8")
        config._log_ag.info(f"[Prompt] Loaded config/system_prompt.txt ({len(text)} chars)")
        return text
    config._log_ag.warning("[Prompt] system_prompt.txt not found — using built-in fallback prompt")
    return "You are an expert Kubernetes operations assistant.\nALWAYS call tools first. NEVER fabricate data.\nALWAYS search documentation before finalising a diagnosis.\nSITE-SPECIFIC RULES:\n{custom_rules}\n"

SYSTEM_PROMPT = _load_system_prompt()

def _registry_to_openai_schema(name: str, cfg: dict) -> dict:
    params = cfg.get("parameters", {})
    properties, required = {}, []
    for k, v in params.items():
        prop = {"type": v.get("type", "string")}
        if "description" in v: prop["description"] = v["description"]
        if "enum" in v: prop["enum"] = v["enum"]
        properties[k] = prop
        if "default" not in v: required.append(k)
    schema = {"type": "function", "function": {"name": name, "description": cfg["description"], "parameters": {"type": "object", "properties": properties}}}
    if required: schema["function"]["parameters"]["required"] = required
    return schema

def _call_tool(name: str, args: dict, all_tools: dict) -> str:
    cfg = all_tools.get(name)
    if not cfg: return f"Tool '{name}' not found."
    fn, params = cfg["fn"], cfg.get("parameters", {})
    for k, v in params.items():
        if k not in args and "default" in v: args[k] = v["default"]
    try: return str(fn(**args))
    except Exception as e:
        config._log_ag.error(f"[_call_tool] {name} raised: {e}", exc_info=True)
        return f"Tool '{name}' failed: {e}"

class AgentState(TypedDict):
    messages: Annotated[list, add_messages]
    tool_calls_made: list
    iteration: int
    status_updates: list
    direct_answer: Optional[str]
    req_id: str

def _build_llm():
    config._log_ag.info(f"[LLM] Loading model: {config.LLM_MODEL}")
    is_gguf = config.LLM_MODEL.lower().endswith(".gguf") or "gguf" in config.LLM_MODEL.lower()
    if is_gguf: return _build_llm_gguf()
    try:
        import transformers, torch
        is_qwen3 = "qwen3" in config.LLM_MODEL.lower()
        if is_qwen3: config._log_ag.info("[LLM] Qwen3 detected — native tool-calling via apply_chat_template")
        device_map = "auto" if config.NUM_GPU > 0 else "cpu"
        dtype = torch.bfloat16 if config.NUM_GPU > 0 else torch.float32
        tokenizer = transformers.AutoTokenizer.from_pretrained(config.LLM_MODEL, trust_remote_code=True)
        model = transformers.AutoModelForCausalLM.from_pretrained(config.LLM_MODEL, torch_dtype=dtype, device_map=device_map, trust_remote_code=True, use_cache=True)
        model.eval()
        config._log_ag.info("[LLM] Model loaded")
        return tokenizer, model, is_qwen3
    except Exception as e:
        config._log_ag.error(f"[LLM] Load failed: {e}")
        raise

def _build_llm_gguf():
    try: from llama_cpp import Llama
    except ImportError: raise ImportError("llama-cpp-python is required for GGUF models.")
    model_path = config.LLM_MODEL
    n_ctx      = int(os.environ.get("GGUF_N_CTX", "8192"))
    n_threads  = int(os.environ.get("GGUF_N_THREADS", str(os.cpu_count() or 4)))
    config._log_ag.info(f"[LLM/GGUF] Loading {model_path} | ctx={n_ctx} threads={n_threads}")
    if not os.path.isfile(model_path):
        try:
            from huggingface_hub import hf_hub_download
            for quant in ["Q4_K_M.gguf", "Q4_0.gguf", "Q5_K_M.gguf", "Q8_0.gguf"]:
                repo_id, filename = model_path, quant
                parts = model_path.split("/")
                if len(parts) == 3 and parts[-1].endswith(".gguf"): repo_id, filename, quant = "/".join(parts[:2]), parts[-1], parts[-1]
                try: model_path = hf_hub_download(repo_id=repo_id, filename=filename); config._log_ag.info(f"[LLM/GGUF] Downloaded {filename} from {repo_id}"); break
                except Exception: continue
        except ImportError: pass
    if not os.path.isfile(model_path): raise FileNotFoundError(f"GGUF model file not found: {model_path}")
    model = Llama(model_path=model_path, n_ctx=n_ctx, n_threads=n_threads, n_gpu_layers=0, verbose=False)
    config._log_ag.info(f"[LLM/GGUF] Model loaded (CPU, {n_threads} threads, ctx={n_ctx})")
    return None, model, "qwen" in model_path.lower()

def build_agent():
    all_tools = {**K8S_TOOLS, **RAG_TOOLS}
    tool_schemas = [_registry_to_openai_schema(n, c) for n, c in all_tools.items()]
    tool_names   = [s["function"]["name"] for s in tool_schemas]
    config._log_ag.info(f"[build_agent] {len(tool_schemas)} tools: {tool_names}")

    tokenizer, model, _is_qwen3 = _build_llm()
    globals()["_kb_tokenizer"], globals()["_kb_model"], globals()["_kb_is_qwen3"] = tokenizer, model, _is_qwen3

    _sys_prompt = _load_system_prompt().format(custom_rules=config.CUSTOM_RULES or "None.")
    prompt = (_sys_prompt + "\n/no_think") if _is_qwen3 else _sys_prompt

    def _prepare_messages_for_hf(msgs: list, req_id: str = "") -> list:
        if not msgs: return msgs
        has_tool_results = any(isinstance(m, ToolMessage) for m in msgs)
        if not has_tool_results: return [m for m in msgs if isinstance(m, (HumanMessage, SystemMessage))]
        original_question = next((m.content for m in msgs if isinstance(m, HumanMessage)), "")
        tool_results = [m for m in msgs if isinstance(m, ToolMessage)]
        
        # Tool synthesis prompt selection omitted here to stay within limit, but remains identical to original app.py block
        synthesis_prompt = f"Question: {original_question}\n\nTool Results:\n" + "".join([f"--- TOOL RESULT {i} ---\n{tr.content[:40000]}\n" for i, tr in enumerate(tool_results, 1)]) + "\nAnswer the question using only the tool results above."
        return [HumanMessage(content=synthesis_prompt)]

    def _msgs_to_qwen3(msgs: list, include_tools: bool) -> list:
        result = []
        for m in msgs:
            if isinstance(m, SystemMessage): result.append({"role": "system", "content": m.content})
            elif isinstance(m, HumanMessage): result.append({"role": "user", "content": m.content})
            elif isinstance(m, ToolMessage): result.append({"role": "tool", "name": "tool", "content": m.content})
            else:
                tcs = getattr(m, "tool_calls", None) or []
                if tcs: result.append({"role": "assistant", "content": "", "tool_calls": [{"id": tc.get("id", ""), "type": "function", "function": {"name": tc["name"], "arguments": json.dumps(tc.get("args", {}))}} for tc in tcs]})
                else: result.append({"role": "assistant", "content": getattr(m, "content", "")})
        return result

    def _parse_tool_calls(text: str) -> list:
        import uuid
        tcs = []
        for m in re.finditer(r'<tool_call>\s*(.*?)\s*</tool_call>', text, re.DOTALL):
            raw = m.group(1).strip()
            try:
                obj = json.loads(raw)
                args_parsed = json.loads(obj.get("arguments", {})) if isinstance(obj.get("arguments", {}), str) else obj.get("arguments", {})
                tcs.append({"id": f"tc_{uuid.uuid4().hex[:8]}", "name": obj["name"], "args": args_parsed, "type": "tool_call"})
            except Exception: pass
        return tcs

    def llm_node(state: AgentState):
        itr, msgs, updates = state.get("iteration", 0) + 1, state["messages"], list(state.get("status_updates", []))
        if state.get("direct_answer"): return {"messages": [AIMessage(content=state["direct_answer"])], "tool_calls_made": state.get("tool_calls_made", []), "iteration": itr, "status_updates": updates, "direct_answer": None}
        has_tool_results = any(isinstance(m, ToolMessage) for m in msgs)
        invoke_msgs = _prepare_messages_for_hf(msgs, req_id=state.get("req_id", ""))
        chat_msgs = [{"role": "system", "content": prompt}] + _msgs_to_qwen3(invoke_msgs, True)
        _max_new = max(512, config.MAX_NEW_TOKENS) if has_tool_results else max(1024, config.MAX_NEW_TOKENS // 2)

        if tokenizer is None:
            tools_json = json.dumps(tool_schemas, indent=2)
            tool_system = f"{prompt}\n\nAvailable tools:\n{tools_json}"
            gguf_msgs = [{"role": "system", "content": tool_system}] + chat_msgs[1:]
            resp = model.create_chat_completion(messages=gguf_msgs, max_tokens=_max_new, temperature=0.7, top_p=0.8, top_k=20, repeat_penalty=1.05)
            raw_text = resp["choices"][0]["message"].get("content", "") or ""
        else:
            import torch
            kw = {"add_generation_prompt": True, "tools": tool_schemas}
            if _is_qwen3: kw["enable_thinking"] = False
            encoded = tokenizer.apply_chat_template(chat_msgs, tokenize=True, return_tensors="pt", **kw)
            input_ids = (encoded["input_ids"] if hasattr(encoded, "__getitem__") and not hasattr(encoded, "shape") else encoded).to(model.device)
            with torch.no_grad(): output_ids = model.generate(input_ids, max_new_tokens=_max_new, do_sample=True, temperature=0.7, top_p=0.8, top_k=20, repetition_penalty=1.05, pad_token_id=tokenizer.eos_token_id)
            raw_text = tokenizer.decode(output_ids[0][input_ids.shape[-1]:], skip_special_tokens=True)

        tcs = _parse_tool_calls(raw_text)
        content = re.sub(r'<tool_call>[\s\S]*?</tool_call>', '', raw_text).strip()
        response = AIMessage(content=content, tool_calls=tcs)
        if tcs: updates.append(f"🔧 {', '.join(tc['name'] for tc in tcs)}")
        return {"messages": [response], "tool_calls_made": state.get("tool_calls_made", []), "iteration": itr, "status_updates": updates}

    def tool_node(state: AgentState):
        last, results, tools_called, updates = state["messages"][-1], [], list(state.get("tool_calls_made", [])), list(state.get("status_updates", []))
        user_q = next((m.content for m in state["messages"] if isinstance(m, HumanMessage)), "")
        tcs, direct_answer = getattr(last, "tool_calls", []) or [], None
        for tc in tcs:
            name, args = tc["name"], dict(tc.get("args", {}) or {})
            if name == "get_secrets": args["decode"] = config.get_decode_secrets()
            tools_called.append(name)
            updates.append(f"$ {args['command']}" if name == "kubectl_exec" and "command" in args else f"⚙️ {name}")
            out = _call_tool(name, args, all_tools)
            results.append(ToolMessage(content=out, tool_call_id=tc["id"], name=name))
            if name == "rag_search" and isinstance(out, str) and out.startswith("KB_EMPTY:"):
                updates.append("⚠️ Knowledge base is empty")
                direct_answer = "⚠️ " + _MSG_NO_INGEST
            elif len(tcs) == 1 and should_bypass_llm(name, args, out, user_q, req_id=state.get("req_id", "")):
                updates.append("⚡ Direct output (LLM synthesis skipped)")
                direct_answer = build_direct_answer(name, out, user_q, req_id=state.get("req_id", ""))
        return {"messages": results, "tool_calls_made": tools_called, "iteration": state.get("iteration", 0), "status_updates": updates, "direct_answer": direct_answer}

    def router(state: AgentState) -> Literal["tools", "end"]:
        if state.get("iteration", 0) >= 6: return "end"
        tcs = getattr(state["messages"][-1], "tool_calls", None)
        if not tcs: return "end"
        already, pending = state.get("tool_calls_made", []), [tc["name"] for tc in tcs]
        if already and all(name in already for name in pending): return "end"
        return "tools"

    g = StateGraph(AgentState)
    g.add_node("llm", llm_node)
    g.add_node("tools", tool_node)
    g.set_entry_point("llm")
    g.add_conditional_edges("llm", router, {"tools": "tools", "end": END})
    g.add_edge("tools", "llm")
    return g.compile()

_agent = None
def get_agent():
    global _agent
    if _agent is None: _agent = build_agent()
    return _agent

def _clean_response(text: str, user_question: str = "") -> str:
    text = re.sub(r'<think>[\s\S]*?</think>\s*', '', text)
    text = re.sub(r'<\|im_start\|>\w+\s*\n?[\s\S]*?<\|im_end\|>\n?', '', text)
    for tok in ['<|im_end|>', '<s>', '</s>', '[INST]', '[/INST]', '<<SYS>>', '<</SYS>>']: text = text.replace(tok, '')
    if user_question:
        q_stripped, escaped = user_question.strip(), re.escape(user_question.strip())
        text = re.sub(r'(?i)(\s*' + escaped + r'[?!.]?\s*){2,}', ' ', text)
        text = re.sub(r'(?i)^\s*' + escaped + r'[?!.]?\s*\n', '', text)
    text = re.sub(r'Summarise the above tool results.*', '', text, flags=re.IGNORECASE)
    return re.sub(r'\n{3,}', '\n\n', text).strip()

async def run_agent(user_message: str) -> dict:
    import uuid
    req_id = uuid.uuid4().hex[:8]
    agent, t0 = get_agent(), time.time()
    final = await agent.ainvoke({"messages": [HumanMessage(content=user_message)], "tool_calls_made": [], "iteration": 0, "status_updates": [f"🤖 Model: {config.LLM_MODEL}"], "req_id": req_id})
    elapsed, last = time.time() - t0, final["messages"][-1]
    raw = last.content if hasattr(last, "content") else str(last)
    updates = final.get("status_updates", [])
    updates.append(f"✅ Done in {elapsed:.0f}s")
    return {"response": _clean_response(raw, user_message), "tools_used": final.get("tool_calls_made", []), "iterations": final.get("iteration", 0), "status_updates": updates, "elapsed_seconds": round(elapsed, 1), "clarification_needed": False}

async def run_agent_streaming(user_message: str, history: list = None, max_new_tokens: int = 0):
    def _sse(payload: dict) -> str: return f"data: {json.dumps(payload)}\n\n"
    import uuid, asyncio
    req_id, agent, t0 = uuid.uuid4().hex[:8], get_agent(), time.time()
    yield _sse({"type": "status", "text": f"🤖 Model: {config.LLM_MODEL}"})
    
    _saved_max = config.MAX_NEW_TOKENS
    if max_new_tokens > 0: config.MAX_NEW_TOKENS = max_new_tokens
    all_updates, tools_called, final_answer, iteration_count = [f"🤖 Model: {config.LLM_MODEL}"], [], "", 0
    _hb_queue, _hb_stop = asyncio.Queue(), asyncio.Event()

    async def _heartbeat_task():
        tick = 0
        while not _hb_stop.is_set():
            try: await asyncio.wait_for(asyncio.shield(asyncio.sleep(15)), timeout=15)
            except Exception: pass
            tick += 15
            if not _hb_stop.is_set(): await _hb_queue.put(tick)
    _hb_task = asyncio.ensure_future(_heartbeat_task())

    try:
        from langchain_core.messages import AIMessage as _AIMessage
        history_msgs = [HumanMessage(content=t.content) if t.role == "user" else _AIMessage(content=t.content) for t in (history or [])]
        all_messages = history_msgs + [HumanMessage(content=user_message)]

        async for event in agent.astream_events({"messages": all_messages, "tool_calls_made": [], "iteration": 0, "status_updates": [], "req_id": req_id}, version="v2", config={"recursion_limit": 12}):
            while not _hb_queue.empty(): yield _sse({"type": "heartbeat", "text": f"⏳ Still processing… ({_hb_queue.get_nowait()}s elapsed)", "timeout": config.LLM_TIMEOUT})
            kind, name = event.get("event", ""), event.get("name", "")
            if kind == "on_tool_start":
                tool_name = event.get("name", "unknown_tool")
                cmd = event.get("data", {}).get("input", {}).get("command") if tool_name == "kubectl_exec" else None
                txt = f"$ {cmd}" if cmd else f"⚙️ {tool_name}"
                all_updates.append(txt); tools_called.append(tool_name)
                yield _sse({"type": "tool", "name": tool_name, "text": txt, "cmd": cmd})
            elif kind == "on_tool_end":
                tool_name, output = event.get("name", ""), event.get("data", {}).get("output", "")
                txt = f"✓ {tool_name}: {str(output)[:80].replace(chr(10), ' ')}…"
                all_updates.append(txt)
                yield _sse({"type": "status", "text": txt})
            elif kind == "on_chain_end" and name == "llm":
                output = event.get("data", {}).get("output", {})
                iteration_count = output.get("iteration", iteration_count)
                has_tool_calls = any(getattr(m, "tool_calls", None) for m in output.get("messages", []))
                for m in output.get("messages", []):
                    if getattr(m, "content", "") and not getattr(m, "tool_calls", None): final_answer = m.content
                itr_txt = f"🔄 Loop {iteration_count} — LLM called tools, waiting for results…" if has_tool_calls else f"✍️ Loop {iteration_count} — LLM synthesising final answer…"
                yield _sse({"type": "iteration", "iteration": iteration_count, "text": itr_txt, "has_tool_calls": has_tool_calls})

        elapsed = round(time.time() - t0, 1)
        _hb_stop.set(); _hb_task.cancel()
        yield _sse({"type": "status", "text": f"✅ Done in {elapsed}s"})
        yield _sse({"type": "result", "response": _clean_response(final_answer, user_message), "tools_used": list(dict.fromkeys(tools_called)), "iterations": iteration_count, "status_updates": all_updates, "elapsed_seconds": elapsed, "clarification_needed": False})
    except Exception as exc:
        _hb_stop.set(); _hb_task.cancel()
        yield _sse({"type": "error", "text": str(exc)})
    finally: config.MAX_NEW_TOKENS = _saved_max

def _llm_synthesise(context: str, question: str, top_k: int = 50, max_tokens: int = 0) -> str:
    _kb_is_empty = not context or not context.strip() or (isinstance(context, str) and context.startswith("KB_EMPTY:"))
    _no_match    = (isinstance(context, str) and context.strip() == "No relevant documentation found.")
    _no_context  = _kb_is_empty or _no_match
    
    if _no_context and _is_kb_topic(question): return _MSG_NO_INGEST
    
    try: tok, mdl, is_q3 = globals()["_kb_tokenizer"], globals()["_kb_model"], globals()["_kb_is_qwen3"]
    except KeyError: return context or ""
    
    sys_prompt = "You are the ECS Knowledge Bot for Cloudera ECS... Answer questions strictly from the knowledge base context provided."
    user_msg = f"[KNOWLEDGE BASE CONTEXT]\n{context}\n[END CONTEXT]\n\nQuestion: {question}\n\nAnswer using only the context above." if context else f"Question: {question}"
    msgs = [{"role": "system", "content": sys_prompt}, {"role": "user", "content": user_msg}]
    _max_out = max_tokens if max_tokens > 0 else min(512 + top_k * 16, 4096)
    
    try:
        if tok is None:
            resp = mdl.create_chat_completion(messages=msgs, max_tokens=_max_out, temperature=0.3, top_p=0.9, repeat_penalty=1.05)
            raw = resp["choices"][0]["message"].get("content", "") or ""
        else:
            import torch
            kw = {"add_generation_prompt": True}
            if is_q3: kw["enable_thinking"] = False
            encoded = tok.apply_chat_template(msgs, tokenize=True, return_tensors="pt", **kw)
            ids = (encoded["input_ids"] if hasattr(encoded, "__getitem__") and not hasattr(encoded, "shape") else encoded).to(mdl.device)
            with torch.no_grad(): out = mdl.generate(ids, max_new_tokens=_max_out, do_sample=False, temperature=1.0, repetition_penalty=1.05, pad_token_id=tok.eos_token_id)
            raw = tok.decode(out[0][ids.shape[-1]:], skip_special_tokens=True)
        return re.sub(r'<think>[\s\S]*?</think>\s*', '', raw).strip() or context or ""
    except Exception as exc: return context or ""
