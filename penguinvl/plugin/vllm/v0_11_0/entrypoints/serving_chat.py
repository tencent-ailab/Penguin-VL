"""PenguinVL-specific OpenAI serving chat implementation for vLLM."""

from typing import List, Optional, Dict, Any, Callable, Sequence, Union, Tuple, Awaitable
from typing_extensions import Annotated

from pydantic import Field
from transformers import AutoProcessor
from vllm.inputs import TokensPrompt
from vllm.config import ModelConfig
from vllm.entrypoints.openai.serving_chat import OpenAIServingChat
from vllm.transformers_utils.tokenizer import AnyTokenizer, MistralTokenizer
from vllm.entrypoints.chat_utils import (ChatCompletionMessageParam,
                                         ChatTemplateContentFormatOption,
                                         ConversationMessage,
                                         apply_hf_chat_template,
                                         apply_mistral_chat_template,
                                         resolve_chat_template_content_format,
                                         AsyncMultiModalItemTracker,
                                         ModalityStr,
                                         _ChatTemplateContentFormat,
                                         _parse_chat_message_content,
                                         _postprocess_messages)
from vllm.entrypoints.openai.tool_parsers import ToolParser
from vllm.entrypoints.openai.serving_engine import (ChatLikeRequest,
                                                    TextTokensPrompt,
                                                    RequestPrompt)
from vllm.multimodal import MultiModalDataDict
from vllm.utils import is_list_of
from vllm.entrypoints.openai.protocol import (ChatCompletionRequest,
                                              EmbeddingChatRequest,
                                              TokenizeChatRequest)


class PenguinVLAsyncMultiModalItemTracker(AsyncMultiModalItemTracker):

    def _placeholder_str(self, modality: ModalityStr, current_count: int) -> Optional[str]:
        if modality == "image":
            return "<image>"
        raise TypeError(f"Unknown modality: {modality}")


def parse_chat_messages_futures(
    messages: List[ChatCompletionMessageParam],
    model_config: ModelConfig,
    tokenizer: AnyTokenizer,
    content_format: _ChatTemplateContentFormat,
) -> Tuple[List[ConversationMessage], Awaitable[Optional[MultiModalDataDict]]]:
    conversation: List[ConversationMessage] = []
    mm_tracker = PenguinVLAsyncMultiModalItemTracker(model_config, tokenizer)

    for msg in messages:
        sub_messages = _parse_chat_message_content(
            msg,
            mm_tracker,
            content_format,
        )
        conversation.extend(sub_messages)

    _postprocess_messages(conversation)

    return conversation, mm_tracker.all_mm_data()


class PenguinVLOpenAIServingChat(OpenAIServingChat):

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.hf_processor = AutoProcessor.from_pretrained(
            self.model_config.model,
            trust_remote_code=self.model_config.trust_remote_code,
        )

    async def _preprocess_chat(
        self,
        request: ChatLikeRequest,
        tokenizer: AnyTokenizer,
        messages: list[ChatCompletionMessageParam],
        chat_template: Optional[str],
        chat_template_content_format: ChatTemplateContentFormatOption,
        add_generation_prompt: bool = True,
        continue_final_message: bool = False,
        tool_dicts: Optional[list[dict[str, Any]]] = None,
        documents: Optional[list[dict[str, str]]] = None,
        chat_template_kwargs: Optional[dict[str, Any]] = None,
        tool_parser: Optional[Callable[[AnyTokenizer], ToolParser]] = None,
        truncate_prompt_tokens: Optional[Annotated[int, Field(ge=1)]] = None,
        add_special_tokens: bool = False,
    ) -> tuple[list[ConversationMessage], Sequence[RequestPrompt],
               list[TokensPrompt]]:
        model_config = self.model_config

        resolved_content_format = resolve_chat_template_content_format(
            chat_template,
            tool_dicts,
            chat_template_content_format,
            tokenizer,
            trust_remote_code=model_config.trust_remote_code,
        )
        conversation, mm_data_future = parse_chat_messages_futures(
            messages,
            model_config,
            tokenizer,
            content_format=resolved_content_format,
        )

        if chat_template is None:
            chat_template = self.hf_processor.chat_template

        _chat_template_kwargs: dict[str, Any] = dict(
            chat_template=chat_template,
            add_generation_prompt=add_generation_prompt,
            continue_final_message=continue_final_message,
            tools=tool_dicts,
            documents=documents,
        )
        _chat_template_kwargs.update(chat_template_kwargs or {})

        if "image_token" not in _chat_template_kwargs:
            _chat_template_kwargs["image_token"] = "<image>"

        request_prompt: Union[str, list[int]]
        if isinstance(tokenizer, MistralTokenizer):
            request_prompt = apply_mistral_chat_template(
                tokenizer,
                messages=messages,
                **_chat_template_kwargs,
            )
        else:
            request_prompt = apply_hf_chat_template(
                self.hf_processor.tokenizer,
                trust_remote_code=model_config.trust_remote_code,
                conversation=conversation,
                **_chat_template_kwargs,
            )

        mm_data = await mm_data_future

        should_parse_tools = tool_parser is not None and (hasattr(
            request, "tool_choice") and request.tool_choice != "none")

        if should_parse_tools:
            if not isinstance(request, ChatCompletionRequest):
                msg = "Tool usage is only supported for Chat Completions API"
                raise NotImplementedError(msg)
            request = tool_parser(tokenizer).adjust_request(request=request)

        if isinstance(request_prompt, str):
            prompt_inputs = await self._tokenize_prompt_input_async(
                request,
                tokenizer,
                request_prompt,
                truncate_prompt_tokens=truncate_prompt_tokens,
                add_special_tokens=add_special_tokens,
            )
        else:
            assert is_list_of(request_prompt, int), (
                "Prompt has to be either a string or a list of token ids")
            prompt_inputs = TextTokensPrompt(
                prompt=tokenizer.decode(request_prompt),
                prompt_token_ids=request_prompt)

        engine_prompt = TokensPrompt(
            prompt_token_ids=prompt_inputs["prompt_token_ids"])
        if mm_data is not None:
            engine_prompt["multi_modal_data"] = mm_data
        if request.mm_processor_kwargs is not None:
            engine_prompt["mm_processor_kwargs"] = request.mm_processor_kwargs

        return conversation, [request_prompt], [engine_prompt]
