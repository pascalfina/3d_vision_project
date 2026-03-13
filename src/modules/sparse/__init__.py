import importlib
import importlib.util
from typing import *

BACKEND = "spconv"
DEBUG = False
ATTN = "xformers"


def _is_sparse_backend_available(backend: str) -> bool:
    if backend == "spconv":
        return importlib.util.find_spec("spconv.pytorch") is not None
    if backend == "torchsparse":
        return importlib.util.find_spec("torchsparse") is not None
    return False


def _is_attn_backend_available(backend: str) -> bool:
    if backend == "xformers":
        return importlib.util.find_spec("xformers.ops") is not None
    if backend == "flash_attn":
        return importlib.util.find_spec("flash_attn") is not None
    if backend in {"sdpa", "naive"}:
        return True
    return False


def _resolve_sparse_backend(preferred_backend: str) -> str:
    candidates = [preferred_backend, "spconv", "torchsparse"]
    for backend in candidates:
        if _is_sparse_backend_available(backend):
            return backend
    raise RuntimeError("No supported sparse backend is available")


def _resolve_attn_backend(preferred_backend: str) -> str:
    candidates = [preferred_backend, "xformers", "flash_attn", "sdpa", "naive"]
    for backend in candidates:
        if _is_attn_backend_available(backend):
            return backend
    raise RuntimeError("No supported sparse attention backend is available")


def __from_env():
    import os

    global BACKEND
    global DEBUG
    global ATTN

    env_sparse_backend = os.environ.get("SPARSE_BACKEND")
    env_sparse_debug = os.environ.get("SPARSE_DEBUG")
    env_sparse_attn = os.environ.get("SPARSE_ATTN_BACKEND")
    if env_sparse_attn is None:
        env_sparse_attn = os.environ.get("ATTN_BACKEND")

    if env_sparse_backend is not None and env_sparse_backend in [
        "spconv",
        "torchsparse",
    ]:
        BACKEND = env_sparse_backend
    if env_sparse_debug is not None:
        DEBUG = env_sparse_debug == "1"
    if env_sparse_attn is not None and env_sparse_attn in [
        "xformers",
        "flash_attn",
        "sdpa",
        "naive",
    ]:
        ATTN = env_sparse_attn

    resolved_backend = _resolve_sparse_backend(BACKEND)
    resolved_attn = _resolve_attn_backend(ATTN)
    if resolved_backend != BACKEND:
        print(
            f"[SPARSE] Requested backend '{BACKEND}' is unavailable, "
            f"falling back to '{resolved_backend}'"
        )
        BACKEND = resolved_backend
    if resolved_attn != ATTN:
        print(
            f"[SPARSE] Requested attention '{ATTN}' is unavailable, "
            f"falling back to '{resolved_attn}'"
        )
        ATTN = resolved_attn

    print(f"[SPARSE] Backend: {BACKEND}, Attention: {ATTN}")


__from_env()


def set_backend(backend: Literal["spconv", "torchsparse"]):
    global BACKEND
    BACKEND = _resolve_sparse_backend(backend)


def set_debug(debug: bool):
    global DEBUG
    DEBUG = debug


def set_attn(attn: Literal["xformers", "flash_attn", "sdpa", "naive"]):
    global ATTN
    ATTN = _resolve_attn_backend(attn)

__attributes = {
    "SparseTensor": "basic",
    "sparse_batch_broadcast": "basic",
    "sparse_batch_op": "basic",
    "sparse_cat": "basic",
    "sparse_unbind": "basic",
    "SparseGroupNorm": "norm",
    "SparseLayerNorm": "norm",
    "SparseGroupNorm32": "norm",
    "SparseLayerNorm32": "norm",
    "SparseReLU": "nonlinearity",
    "SparseSiLU": "nonlinearity",
    "SparseGELU": "nonlinearity",
    "SparseActivation": "nonlinearity",
    "SparseLinear": "linear",
    "sparse_scaled_dot_product_attention": "attention",
    "SerializeMode": "attention",
    "sparse_serialized_scaled_dot_product_self_attention": "attention",
    "sparse_windowed_scaled_dot_product_self_attention": "attention",
    "SparseMultiHeadAttention": "attention",
    "SparseConv3d": "conv",
    "SparseInverseConv3d": "conv",
    "SparseDownsample": "spatial",
    "SparseUpsample": "spatial",
    "SparseSubdivide": "spatial",
}

__submodules = ["transformer"]

__all__ = list(__attributes.keys()) + __submodules


def __getattr__(name):
    if name not in globals():
        if name in __attributes:
            module_name = __attributes[name]
            module = importlib.import_module(f".{module_name}", __name__)
            globals()[name] = getattr(module, name)
        elif name in __submodules:
            module = importlib.import_module(f".{name}", __name__)
            globals()[name] = module
        else:
            raise AttributeError(f"module {__name__} has no attribute {name}")
    return globals()[name]


# For Pylance
if __name__ == "__main__":
    import transformer

    from .attention import *
    from .basic import *
    from .conv import *
    from .linear import *
    from .nonlinearity import *
    from .norm import *
    from .spatial import *
