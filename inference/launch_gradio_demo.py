import sys
sys.path.append('.')

import argparse
import subprocess
from threading import Thread

from inference.interface import PenguinVLQwen3GradioInterface
from inference.server import PenguinVLQwen3PlainClient


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--model-path", "--model_path", type=str, required=True)
    parser.add_argument("--server-port", "--server_port", type=int, default=16667)
    parser.add_argument("--interface-port", "--interface_port", type=int, default=33666)
    parser.add_argument("--nproc", type=int, default=1)
    args = parser.parse_args()

    server_thread = Thread(
        target=lambda: subprocess.run(
            [
                "python", "-m",
                "inference.server.plain_server",
                "--model-path", args.model_path,
                "--nproc", str(args.nproc),
                "--port", str(args.server_port),
            ]
        )
    )
    server_thread.daemon = True
    server_thread.start()

    model_client = PenguinVLQwen3PlainClient(port=args.server_port)
    interface = PenguinVLQwen3GradioInterface(
        model_client,
        example_dir="./assets",
        server_name="0.0.0.0",
        server_port=args.interface_port,
    )
    interface.launch()
