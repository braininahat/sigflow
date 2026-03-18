"""LLM inference process node — streaming via llama-server.

Starts a llama-server subprocess, sends prompts via OpenAI-compatible API
with SSE streaming, and pushes partial text to a QML-visible text provider.
Logs tok/s, latency, and token counts for every invocation.
"""
from __future__ import annotations

import atexit
import base64
import json
import logging
import os
import re
import shutil
import subprocess
import threading
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
        Param("hf_model", "str", "unsloth/Qwen3.5-4B-GGUF:UD-Q3_K_XL", label="HF Model"),
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

    # First request may be slow (KV cache allocation, model warmup)
    timeout = 120 if state.get("_first_request", True) else 60
    state["_first_request"] = False

    with urlopen(req, timeout=timeout) as resp:
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

    # Expose tok/s for metrics
    state["_tok_s"] = tok_s

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
    """Find llama-server binary, preferring user-local then bundled."""
    from sigflow.paths import DATA_DIR, resolve_path
    # User-local build (native-optimized, persists across app versions)
    user_bin = DATA_DIR / "bin" / "llama-server"
    if user_bin.exists():
        log.info("using user-local llama-server: %s", user_bin)
        return str(user_bin)
    # Bundled build (from AppImage/PyInstaller)
    local_bin = resolve_path("third_party/llama.cpp/build/bin/llama-server")
    if local_bin.exists():
        return str(local_bin)
    # System PATH
    system_bin = shutil.which("llama-server")
    if system_bin:
        return system_bin
    raise FileNotFoundError(
        "llama-server not found. Build with CUDA: cd third_party/llama.cpp && "
        "cmake -B build -DGGML_CUDA=ON -DCMAKE_BUILD_TYPE=Release && "
        "cmake --build build --target llama-server -j$(nproc)"
    )


def _resolve_cached_model(hf_model: str) -> str | None:
    """Resolve cached GGUF file path from HF model spec, or None if not cached."""
    cache_dir = Path(
        os.environ.get("LLAMA_CACHE")
        or os.environ.get("XDG_CACHE_HOME", Path.home() / ".cache")
    ) / "llama.cpp"
    repo_part, _, tag = hf_model.partition(":")
    owner, _, repo = repo_part.partition("/")
    manifest_path = cache_dir / f"manifest={owner}={repo}={tag}.json"
    if not manifest_path.exists():
        return None
    try:
        manifest = json.loads(manifest_path.read_text())
    except (json.JSONDecodeError, OSError):
        return None
    gguf_info = manifest.get("ggufFile", {})
    if not gguf_info:
        return None
    cached = cache_dir / f"{owner}_{repo}_{gguf_info['rfilename']}"
    return str(cached) if cached.exists() else None


# Module-level warmup process — started eagerly at app startup
_warmup_proc: subprocess.Popen | None = None


def shutdown_server():
    """Terminate llama-server subprocess to free GPU memory.

    Called via app.aboutToQuit and atexit as a fallback.
    """
    global _warmup_proc
    if _warmup_proc is not None and _warmup_proc.poll() is None:
        log.info("shutting down llama-server (pid=%d)", _warmup_proc.pid)
        _warmup_proc.terminate()
        try:
            _warmup_proc.wait(timeout=5)
        except subprocess.TimeoutExpired:
            _warmup_proc.kill()
            _warmup_proc.wait()


atexit.register(shutdown_server)


def warmup_server(
    hf_model: str = "unsloth/Qwen3.5-4B-GGUF:UD-Q3_K_XL",
    port: int = 8078,
    gpu: bool = False,
    ctx_size: int = 2048,
    callback=None,
):
    """Start llama-server eagerly in background (call before WiFi switches).

    If callback is provided, it's called (no args) once the server is healthy.
    """
    def _warmup():
        global _warmup_proc
        try:
            llama_bin = _find_llama_server()
        except FileNotFoundError:
            log.warning("llama-server not found, skipping warmup")
            if callback:
                callback()
            return

        cmd = [
            llama_bin,
            "-hf", hf_model,
            "--port", str(port),
            "-c", str(ctx_size),
            "--jinja",
            "--chat-template-kwargs", '{"enable_thinking": false}',
        ]
        if gpu:
            cmd += ["-ngl", "99"]

        env = os.environ.copy()
        llama_lib_dir = str(Path(llama_bin).parent)
        env["LD_LIBRARY_PATH"] = llama_lib_dir + ":" + env.get("LD_LIBRARY_PATH", "")

        log.info("warmup: starting llama-server: %s", " ".join(cmd))
        _warmup_proc = subprocess.Popen(
            cmd, stdout=subprocess.DEVNULL, stderr=subprocess.PIPE, env=env,
        )

        def _read_stderr():
            for line in _warmup_proc.stderr:
                log.info("llama-server: %s", line.decode(errors="replace").rstrip())
        threading.Thread(target=_read_stderr, daemon=True, name="llama-stderr").start()

        # Wait for health check
        for _ in range(120):
            if _warmup_proc.poll() is not None:
                log.error("warmup: llama-server exited (%d)", _warmup_proc.returncode)
                _warmup_proc = None
                break
            try:
                urlopen(f"http://127.0.0.1:{port}/health", timeout=1)
                log.info("warmup: llama-server ready (pid=%d)", _warmup_proc.pid)
                break
            except Exception:
                time.sleep(0.5)

        if callback:
            callback()

    threading.Thread(target=_warmup, daemon=True, name="llama-warmup").start()


def _start_server(state, config):
    """Start llama-server subprocess and wait for health check."""
    global _warmup_proc
    port = config.get("port", 8078)

    # Check if warmup server is already running
    if _warmup_proc is not None and _warmup_proc.poll() is None:
        try:
            urlopen(f"http://127.0.0.1:{port}/health", timeout=2)
            log.info("adopting warmup llama-server (pid=%d)", _warmup_proc.pid)
            state["proc"] = _warmup_proc
            state["_first_request"] = False  # already warm
            _warmup_proc = None
            return
        except Exception:
            log.warning("warmup server not healthy, starting fresh")
            _warmup_proc.terminate()
            _warmup_proc = None

    # Check if a server is already running on the port (e.g. from previous session)
    try:
        urlopen(f"http://127.0.0.1:{port}/health", timeout=2)
        log.info("llama-server already running on port %d", port)
        state["_first_request"] = False
        return
    except Exception:
        pass

    llama_bin = _find_llama_server()
    hf_model = config.get("hf_model", "unsloth/Qwen3.5-4B-GGUF:UD-Q3_K_XL")
    ctx_size = config.get("context_size", 256)
    gpu = config.get("gpu", False)

    # Prefer cached model (-m) for offline use, fall back to -hf
    cached_path = _resolve_cached_model(hf_model)
    if cached_path:
        log.info("using cached model: %s", cached_path)
        model_args = ["-m", cached_path]
    else:
        model_args = ["-hf", hf_model]

    cmd = [
        llama_bin,
        *model_args,
        "--port", str(port),
        "-c", str(ctx_size),
        "--jinja",
    ]
    if gpu:
        cmd += ["-ngl", "99"]

    # Set LD_LIBRARY_PATH: llama libs first, strip AppImage _internal paths
    env = os.environ.copy()
    llama_lib_dir = str(Path(llama_bin).parent)
    ld_path = env.get("LD_LIBRARY_PATH", "")
    clean_parts = [p for p in ld_path.split(":") if p and "_internal" not in p]
    env["LD_LIBRARY_PATH"] = ":".join([llama_lib_dir] + clean_parts)

    log.info("starting llama-server: %s", " ".join(cmd))
    proc = subprocess.Popen(cmd, stdout=subprocess.DEVNULL, stderr=subprocess.PIPE, env=env)
    state["proc"] = proc
    state["_first_request"] = True

    def _read_stderr():
        for line in proc.stderr:
            log.info("llama-server: %s", line.decode(errors="replace").rstrip())
    threading.Thread(target=_read_stderr, daemon=True, name="llama-stderr").start()

    for _ in range(120):
        if proc.poll() is not None:
            stderr = proc.stderr.read().decode(errors="replace")
            log.error("llama-server exited (%d): %s", proc.returncode, stderr[-500:])
            del state["proc"]
            raise RuntimeError(f"llama-server exited with code {proc.returncode}")
        try:
            urlopen(f"http://127.0.0.1:{port}/health", timeout=1)
            log.info("llama-server ready (pid=%d, gpu=%s)", proc.pid, gpu)
            return
        except Exception:
            time.sleep(0.5)

    proc.terminate()
    proc.wait()
    del state["proc"]
    raise TimeoutError("llama-server failed to start within 60s")


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
