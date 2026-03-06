"""PenguinVL API server for vLLM."""

import signal
from argparse import Namespace

from starlette.datastructures import State

from vllm.config import ModelConfig
from vllm.engine.protocol import EngineClient
from vllm.entrypoints.chat_utils import load_chat_template
from vllm.entrypoints.launcher import serve_http
from vllm.entrypoints.logger import RequestLogger
from vllm.entrypoints.openai.api_server import (
    create_server_socket, build_async_engine_client, build_app
)
from vllm.reasoning import ReasoningParserManager
from vllm.entrypoints.openai.serving_completion import OpenAIServingCompletion
from vllm.entrypoints.openai.serving_models import (BaseModelPath,
                                                    OpenAIServingModels)
from vllm.entrypoints.openai.tool_parsers import ToolParserManager
from vllm.logger import init_logger
from vllm.utils import is_valid_ipv6_address, set_ulimit
from vllm.version import __version__ as VLLM_VERSION

from .serving_chat import PenguinVLOpenAIServingChat


TIMEOUT_KEEP_ALIVE = 5

logger = init_logger('vllm.entrypoints.openai.api_server')


async def init_app_state(
    engine_client: EngineClient,
    model_config: ModelConfig,
    state: State,
    args: Namespace,
) -> None:
    if args.served_model_name is not None:
        served_model_names = args.served_model_name
    else:
        served_model_names = [args.model]

    if args.disable_log_requests:
        request_logger = None
    else:
        request_logger = RequestLogger(max_log_len=args.max_log_len)

    base_model_paths = [
        BaseModelPath(name=name, model_path=args.model)
        for name in served_model_names
    ]

    state.engine_client = engine_client
    state.log_stats = not args.disable_log_stats

    resolved_chat_template = load_chat_template(args.chat_template)
    if resolved_chat_template is not None:
        logger.info("Using supplied chat template:\n%s",
                    resolved_chat_template)

    assert model_config.runner_type == "generate"

    state.openai_serving_models = OpenAIServingModels(
        engine_client=engine_client,
        model_config=model_config,
        base_model_paths=base_model_paths,
        lora_modules=args.lora_modules,
        prompt_adapters=args.prompt_adapters,
    )
    await state.openai_serving_models.init_static_loras()
    state.openai_serving_chat = PenguinVLOpenAIServingChat(
        engine_client,
        model_config,
        state.openai_serving_models,
        args.response_role,
        request_logger=request_logger,
        chat_template=resolved_chat_template,
        chat_template_content_format=args.chat_template_content_format,
        return_tokens_as_token_ids=args.return_tokens_as_token_ids,
        enable_auto_tools=args.enable_auto_tool_choice,
        tool_parser=args.tool_call_parser,
        enable_reasoning=args.enable_reasoning,
        reasoning_parser=args.reasoning_parser,
        enable_prompt_tokens_details=args.enable_prompt_tokens_details,
    )
    state.openai_serving_completion = OpenAIServingCompletion(
        engine_client,
        model_config,
        state.openai_serving_models,
        request_logger=request_logger,
        return_tokens_as_token_ids=args.return_tokens_as_token_ids,
    )

    state.task = model_config.task
    state.enable_server_load_tracking = args.enable_server_load_tracking
    state.server_load_metrics = 0


async def run_server(args, **uvicorn_kwargs) -> None:
    logger.info("vLLM API server version %s", VLLM_VERSION)
    logger.info("args: %s", args)

    if args.tool_parser_plugin and len(args.tool_parser_plugin) > 3:
        ToolParserManager.import_tool_parser(args.tool_parser_plugin)

    valid_tool_parses = ToolParserManager.tool_parsers.keys()
    if args.enable_auto_tool_choice \
        and args.tool_call_parser not in valid_tool_parses:
        raise KeyError(f"invalid tool call parser: {args.tool_call_parser} "
                       f"(chose from {{ {','.join(valid_tool_parses)} }})")

    valid_reasoning_parses = ReasoningParserManager.reasoning_parsers.keys()
    if args.enable_reasoning \
        and args.reasoning_parser not in valid_reasoning_parses:
        raise KeyError(
            f"invalid reasoning parser: {args.reasoning_parser} "
            f"(chose from {{ {','.join(valid_reasoning_parses)} }})")

    sock_addr = (args.host or "", args.port)
    sock = create_server_socket(sock_addr)
    set_ulimit()

    def signal_handler(*_) -> None:
        raise KeyboardInterrupt("terminated")

    signal.signal(signal.SIGTERM, signal_handler)

    async with build_async_engine_client(args) as engine_client:
        app = build_app(args)

        model_config = await engine_client.get_model_config()
        await init_app_state(engine_client, model_config, app.state, args)

        def _listen_addr(a: str) -> str:
            if is_valid_ipv6_address(a):
                return '[' + a + ']'
            return a or "0.0.0.0"

        is_ssl = args.ssl_keyfile and args.ssl_certfile
        logger.info("Starting vLLM API server on http%s://%s:%d",
                    "s" if is_ssl else "", _listen_addr(sock_addr[0]),
                    sock_addr[1])

        shutdown_task = await serve_http(
            app,
            sock=sock,
            enable_ssl_refresh=args.enable_ssl_refresh,
            host=args.host,
            port=args.port,
            log_level=args.uvicorn_log_level,
            access_log=not args.disable_uvicorn_access_log,
            timeout_keep_alive=TIMEOUT_KEEP_ALIVE,
            ssl_keyfile=args.ssl_keyfile,
            ssl_certfile=args.ssl_certfile,
            ssl_ca_certs=args.ssl_ca_certs,
            ssl_cert_reqs=args.ssl_cert_reqs,
            **uvicorn_kwargs,
        )

    await shutdown_task
    sock.close()
