"""Allow running: python -m penguinvl.plugin.vllm serve <model_path>"""
from . import launch_server

if __name__ == "__main__":
    launch_server()
