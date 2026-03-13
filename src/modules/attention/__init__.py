import importlib.util
from typing import *

BACKEND = "flash_attn"
DEBUG = False


def _is_backend_available(backend: str) -> bool:
    if backend == "flash_attn":
        return importlib.util.find_spec("flash_attn") is not None
    if backend == "xformers":
        return importlib.util.find_spec("xformers.ops") is not None
    if backend in {"sdpa", "naive"}:
        return True
    return False


def _resolve_backend(preferred_backend: str) -> str:
    candidates = [
        preferred_backend,
        "flash_attn",
        "xformers",
        "sdpa",
        "naive",
    ]
    for backend in candidates:
        if _is_backend_available(backend):
            return backend
    raise RuntimeError("No supported attention backend is available")


def __from_env():
    import os

    global BACKEND
    global DEBUG

    env_attn_backend = os.environ.get("ATTN_BACKEND")
    env_sttn_debug = os.environ.get("ATTN_DEBUG")

    if env_attn_backend is not None and env_attn_backend in [
        "xformers",
        "flash_attn",
        "sdpa",
        "naive",
    ]:
        BACKEND = env_attn_backend
    if env_sttn_debug is not None:
        DEBUG = env_sttn_debug == "1"

    resolved_backend = _resolve_backend(BACKEND)
    if resolved_backend != BACKEND:
        print(
            f"[ATTENTION] Requested backend '{BACKEND}' is unavailable, "
            f"falling back to '{resolved_backend}'"
        )
        BACKEND = resolved_backend

    print(f"[ATTENTION] Using backend: {BACKEND}")


__from_env()


def set_backend(backend: Literal["xformers", "flash_attn", "sdpa", "naive"]):
    global BACKEND
    BACKEND = _resolve_backend(backend)


def set_debug(debug: bool):
    global DEBUG
    DEBUG = debug


from .full_attn import *
from .modules import *
