import warnings

try:
    import vllm
except ImportError:
    vllm = None

module = None
if vllm is not None:
    try:
        from . import v0_11_0 as module
    except ImportError as e:
        warnings.warn(f"Could not import vLLM plugin: {e}")

if module is not None:
    from vllm import ModelRegistry

    PenguinVLQwen3ForCausalLM = getattr(module, "PenguinVLQwen3ForCausalLM", None)
    if PenguinVLQwen3ForCausalLM is not None:
        ModelRegistry.register_model("PenguinVLQwen3ForCausalLM", PenguinVLQwen3ForCausalLM)


def launch_server():
    if vllm is None:
        raise ImportError("vllm is not installed.")
    if module is None:
        raise ImportError("vLLM plugin could not be loaded.")
    getattr(module, "run_vllm_cli")()
