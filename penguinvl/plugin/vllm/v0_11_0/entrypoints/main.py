"""CLI entrypoint for PenguinVL vLLM server."""

import argparse
import os
import signal
import sys
import uvloop

import vllm.version
from vllm.engine.arg_utils import EngineArgs
from vllm.entrypoints.cli.serve import ServeSubcommand as _ServeSubcommand
from vllm.logger import init_logger
from vllm.utils import FlexibleArgumentParser

from .api_server import run_server

logger = init_logger(__name__)


class ServeSubcommand(_ServeSubcommand):

    @staticmethod
    def cmd(args: argparse.Namespace) -> None:
        if args.model != EngineArgs.model:
            raise ValueError(
                "With `vllm serve`, you should provide the model as a "
                "positional argument instead of via the `--model` option.")
        args.model = args.model_tag
        uvloop.run(run_server(args))


def register_signal_handlers():
    def signal_handler(sig, frame):
        sys.exit(0)

    signal.signal(signal.SIGINT, signal_handler)
    signal.signal(signal.SIGTSTP, signal_handler)


def env_setup():
    if "VLLM_WORKER_MULTIPROC_METHOD" not in os.environ:
        logger.debug("Setting VLLM_WORKER_MULTIPROC_METHOD to 'spawn'")
        os.environ["VLLM_WORKER_MULTIPROC_METHOD"] = "spawn"


def run_vllm_cli():
    env_setup()

    parser = FlexibleArgumentParser(description="vLLM CLI")
    parser.add_argument('-v',
                        '--version',
                        action='version',
                        version=vllm.version.__version__)
    subparsers = parser.add_subparsers(required=False, dest="subparser")

    cmd = ServeSubcommand()
    cmd.subparser_init(subparsers).set_defaults(dispatch_function=cmd.cmd)
    args = parser.parse_args()
    cmd.validate(args)

    if hasattr(args, "dispatch_function"):
        args.dispatch_function(args)
    else:
        parser.print_help()
