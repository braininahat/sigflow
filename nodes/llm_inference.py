"""LLM inference process node — streaming via llama-server.

Starts a llama-server subprocess, sends prompts via OpenAI-compatible API
with SSE streaming, and pushes partial text to a QML-visible text provider.
Logs tok/s, latency, and token counts for every invocation.
"""
from __future__ import annotations

import base64
import json
import logging
import re
import shutil
import subprocess
import time
from pathlib import Path
from urllib.request import Request, urlopen

from sigflow.node import process_node, Param
from sigflow.types import Event, Port, Sample

log = logging.getLogger(__name__)


@process_node(
    name="llm_inference",
    inputs=[Port("prompt", Event)],
    outputs=[Port("text", Event)],
    category="inference",
    params=[
        Param("hf_model", "str", "unsloth/Qwen3.5-0.8B-GGUF:Q3_K_XL", label="HF Model"),
        Param("port", "int", 8078, label="Server Port"),
        Param("gpu", "bool", False, label="GPU Offload"),
        Param("max_tokens", "int", 40, label="Max Tokens"),
        Param("context_size", "int", 2048, label="Context Size"),
    ],
)
def llm_inference(item, *, state, config):
    if "proc" not in state:
        _start_server(state, config)

    prompt_data = item.data  # {"system": str, "user": str, "max_tokens": int}
    system = prompt_data.get("system", "")
    user = prompt_data.get("user", "")
    max_tokens = prompt_data.get("max_tokens", config.get("max_tokens", 40))
    port = config.get("port", 8078)

    text_provider = state.get("text_provider")

    # Passthrough mode: skip LLM, forward text directly (used for template feedback)
    if prompt_data.get("_passthrough"):
        log.info("LLM passthrough: %s", user[:80])
        meta = {**item.metadata}
        if prompt_data.get("_feedback"):
            meta["is_feedback"] = True
        return {"text": item.replace(
            data={"text": user, "usage": {}, "latency_ms": 0, "tok_s": 0},
            metadata=meta,
            port_type=Event,
        )}

    # Build user content — multimodal if image_path provided
    image_path = prompt_data.get("image_path")
    if image_path:
        b64 = _prepare_image(image_path)
        user_content = [
            {"type": "image_url", "image_url": {"url": f"data:image/jpeg;base64,{b64}"}},
            {"type": "text", "text": user},
        ]
    else:
        user_content = user

    # Build messages — prepend conversation history if provided
    messages = [{"role": "system", "content": system}]
    history = prompt_data.get("_history", [])
    if history:
        messages.extend(history)
    messages.append({"role": "user", "content": user_content})

    payload = json.dumps({
        "messages": messages,
        "temperature": 0.7,
        "max_tokens": max_tokens,
        "stream": True,
        "stream_options": {"include_usage": True},
    }).encode()

    req = Request(
        f"http://127.0.0.1:{port}/v1/chat/completions",
        data=payload,
        headers={"Content-Type": "application/json"},
    )

    t0 = time.perf_counter()
    accumulated = ""
    usage = {}
    chunk_count = 0

    with urlopen(req, timeout=30) as resp:
        for line in resp:
            line = line.strip()
            if not line or not line.startswith(b"data: "):
                continue
            data_str = line[6:]
            if data_str == b"[DONE]":
                break
            chunk = json.loads(data_str)

            choices = chunk.get("choices", [])
            if choices:
                delta = choices[0].get("delta", {})
                token = delta.get("content", "")
                if token:
                    accumulated += token
                    chunk_count += 1
                    if text_provider is not None:
                        text_provider.update_text(accumulated)

            if "usage" in chunk:
                usage = chunk["usage"]

    latency_ms = (time.perf_counter() - t0) * 1000
    token_count = usage.get("completion_tokens", chunk_count)
    tok_s = token_count / (latency_ms / 1000) if latency_ms > 0 else 0

    # Strip thinking tags if present (Qwen3 sometimes wraps in <think>)
    text = re.sub(r"<think>.*?</think>", "", accumulated, flags=re.DOTALL).strip()

    log.info(
        "LLM: %s (%dtok, %.0fms, %.1ftok/s, prompt=%dtok)",
        repr(text[:80]), token_count, latency_ms, tok_s,
        usage.get("prompt_tokens", 0),
    )

    # Update perf info on provider
    if text_provider is not None:
        perf = f"{latency_ms:.0f}ms | {token_count}tok | {tok_s:.1f} tok/s"
        text_provider.update_perf(perf)

    # Notify orchestrator (e.g. for leak detection)
    on_text = state.get("on_text")
    if on_text is not None:
        on_text(text, prompt_data)

    # Safety net: strip target word from feedback output
    target_word = prompt_data.get("_target_word", "")
    if target_word and prompt_data.get("_feedback"):
        cleaned = re.sub(rf'\b{re.escape(target_word)}\b', "it", text, flags=re.IGNORECASE)
        if cleaned != text:
            log.info("LLM: stripped '%s' from feedback: %s → %s", target_word, repr(text[:60]), repr(cleaned[:60]))
            text = cleaned

    meta = {**item.metadata}
    if prompt_data.get("_feedback"):
        meta["is_feedback"] = True

    return {"text": item.replace(
        data={
            "text": text,
            "usage": usage,
            "latency_ms": latency_ms,
            "tok_s": tok_s,
        },
        metadata=meta,
        port_type=Event,
    )}


def _prepare_image(path, max_dim=512):
    """Resize image to max_dim and encode as JPEG base64."""
    import cv2
    img = cv2.imread(str(path))
    if img is None:
        # Fallback: read raw bytes if cv2 can't decode
        with open(path, "rb") as f:
            return base64.b64encode(f.read()).decode()
    h, w = img.shape[:2]
    if max(h, w) > max_dim:
        scale = max_dim / max(h, w)
        img = cv2.resize(img, (int(w * scale), int(h * scale)), interpolation=cv2.INTER_AREA)
    _, buf = cv2.imencode('.jpg', img, [cv2.IMWRITE_JPEG_QUALITY, 85])
    return base64.b64encode(buf).decode()


def _find_llama_server():
    """Find llama-server binary, preferring our CUDA build."""
    # Prefer project-local CUDA build
    project_root = Path(__file__).resolve().parents[3]
    local_bin = project_root / "third_party" / "llama.cpp" / "build" / "bin" / "llama-server"
    if local_bin.exists():
        return str(local_bin)
    system_bin = shutil.which("llama-server")
    if system_bin:
        return system_bin
    raise FileNotFoundError(
        "llama-server not found. Build with CUDA: cd third_party/llama.cpp && "
        "cmake -B build -DGGML_CUDA=ON -DCMAKE_BUILD_TYPE=Release && "
        "cmake --build build --target llama-server -j$(nproc)"
    )


def _start_server(state, config):
    """Start llama-server subprocess and wait for health check."""
    llama_bin = _find_llama_server()

    port = config.get("port", 8078)
    hf_model = config.get("hf_model", "unsloth/Qwen3.5-0.8B-GGUF:Q3_K_XL")
    ctx_size = config.get("context_size", 256)
    gpu = config.get("gpu", False)

    cmd = [
        llama_bin,
        "-hf", hf_model,
        "--port", str(port),
        "-c", str(ctx_size),
        "--jinja",
        "--chat-template-kwargs", '{"enable_thinking": false}',
        "--log-disable",
    ]
    if gpu:
        cmd += ["-ngl", "99"]

    log.info("starting llama-server: %s", " ".join(cmd))
    proc = subprocess.Popen(cmd, stdout=subprocess.DEVNULL, stderr=subprocess.PIPE)
    state["proc"] = proc

    for _ in range(60):
        try:
            urlopen(f"http://127.0.0.1:{port}/health", timeout=1)
            log.info("llama-server ready (pid=%d, gpu=%s)", proc.pid, gpu)
            return
        except Exception:
            time.sleep(0.5)

    proc.terminate()
    proc.wait()
    del state["proc"]
    raise TimeoutError("llama-server failed to start within 30s")


@llm_inference.cleanup
def llm_inference_cleanup(state, config):
    proc = state.get("proc")
    if proc is not None:
        log.info("stopping llama-server (pid=%d)", proc.pid)
        proc.terminate()
        try:
            proc.wait(timeout=5)
        except subprocess.TimeoutExpired:
            proc.kill()
            proc.wait()
