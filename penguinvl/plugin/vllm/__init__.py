import re
import importlib
import warnings

try:
    import vllm
except ImportError:
    vllm = None


def load_module():
    if vllm is None:
        return None

    version = vllm.__version__
    splits = version.split(".")
    version = [int(x) for x in splits[:3]]

    if len(splits) > 3:
        dev = re.findall(r"dev(\d+)", splits[3])
        if dev and int(dev[0]) == 0:
            version[-1] -= 1

    module_name = f".v{'_'.join([str(x) for x in version])}"

    try:
        module = importlib.import_module(module_name, package="penguinvl.plugin.vllm")
    except Exception as e:
        warnings.warn(f"Could not import the implementation for your vLLM version ({vllm.__version__}): {e}")
        module = None

    return module


module = load_module()

if module is not None:
    from vllm import ModelRegistry

    PenguinVLQwen3ForCausalLM = getattr(module, "PenguinVLQwen3ForCausalLM", None)
    if PenguinVLQwen3ForCausalLM is not None:
        ModelRegistry.register_model("PenguinVLQwen3ForCausalLM", PenguinVLQwen3ForCausalLM)


def launch_server():
    if vllm is None:
        raise ImportError("vllm is not installed.")
    module = load_module()
    getattr(module, "run_vllm_cli")()
