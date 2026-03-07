import base64
import html
import mimetypes
from pathlib import Path

import markdown2
import torch
from IPython.display import HTML, Markdown, display

NOTEBOOK_STYLE = """
<style>
.penguin-panel {
    border: 1px solid #dbe3ef;
    border-radius: 18px;
    padding: 20px 24px;
    margin: 14px 0 20px;
    background: #ffffff;
    box-shadow: 0 1px 2px rgba(15, 23, 42, 0.05);
}
.penguin-media-panel {
    max-width: 920px;
    margin-left: auto;
    margin-right: auto;
}
.penguin-image-panel {
    max-width: 540px;
    margin-left: auto;
    margin-right: auto;
}
.penguin-image-shell {
    max-width: 450px;
    margin: 0 auto;
}
.penguin-grid {
    display: grid;
    grid-template-columns: repeat(2, minmax(0, 1fr));
    gap: 18px;
    margin: 14px 0 28px;
}
.penguin-stack {
    display: grid;
    grid-template-columns: 1fr;
    gap: 18px;
    margin: 14px 0 28px;
}
.penguin-question-panel {
    background: linear-gradient(180deg, #fff8f5 0%, #ffffff 100%);
    border-color: #f3d0c6;
}
.penguin-answer-panel {
    background: linear-gradient(180deg, #f5f7ff 0%, #ffffff 100%);
    border-color: #d4dcff;
}
.penguin-code-panel {
    background: linear-gradient(180deg, #f3f5f7 0%, #ffffff 100%);
    border-color: #d8dee6;
}
.penguin-badge {
    display: inline-flex;
    align-items: center;
    padding: 8px 14px;
    border-radius: 999px;
    font-size: 1rem;
    font-weight: 800;
    letter-spacing: 0.01em;
    margin-bottom: 14px;
}
.penguin-question-badge {
    color: #8d3b2e;
    background: #ffe1d8;
    border: 1px solid #f3c3b6;
}
.penguin-answer-badge {
    color: #2747b9;
    background: #e4ebff;
    border: 1px solid #c6d3ff;
}
.penguin-title {
    font-size: 1.35rem;
    font-weight: 700;
    color: #1f2a44;
    margin-bottom: 8px;
}
.penguin-body {
    color: #2a3652;
    line-height: 1.72;
    font-size: 1rem;
    white-space: pre-wrap;
    word-break: break-word;
}
.penguin-markdown {
    white-space: normal;
}
.penguin-caption {
    color: #5e6b84;
    font-size: 0.95rem;
    margin-top: 12px;
}
.penguin-table-wrap {
    overflow-x: auto;
    margin-top: 2px;
}
.penguin-markdown table {
    width: 100%;
    border-collapse: collapse;
    font-size: 0.95rem;
}
.penguin-markdown th,
.penguin-markdown td {
    border: 1px solid #dbe3ef;
    padding: 10px 12px;
    text-align: left;
    vertical-align: top;
}
.penguin-markdown thead th {
    background: #eef3ff;
}
.penguin-markdown tbody tr:nth-child(even) {
    background: #f8fbff;
}
.penguin-markdown p {
    margin: 0;
}
.penguin-markdown p + p {
    margin-top: 12px;
}
.penguin-markdown h1,
.penguin-markdown h2,
.penguin-markdown h3,
.penguin-markdown h4 {
    color: #1f2a44;
    line-height: 1.3;
    margin: 0 0 12px;
}
.penguin-markdown ul,
.penguin-markdown ol {
    margin: 8px 0 0;
    padding-left: 22px;
}
.penguin-markdown li + li {
    margin-top: 6px;
}
.penguin-markdown pre {
    margin: 12px 0 0;
    padding: 16px 18px;
    overflow-x: auto;
    border-radius: 14px;
    background: #edf0f3;
    border: 1px solid #d8dee6;
    line-height: 1.6;
}
.penguin-code-panel .penguin-markdown pre {
    background: #e6eaee;
    border-color: #cfd7df;
}
.penguin-markdown code {
    font-family: ui-monospace, SFMono-Regular, Menlo, Monaco, Consolas, Liberation Mono, Courier New, monospace;
    font-size: 0.92em;
}
.penguin-markdown :not(pre) > code {
    background: #eef3ff;
    padding: 0.15rem 0.35rem;
    border-radius: 6px;
}
.penguin-markdown pre code {
    background: transparent;
    padding: 0;
}
.penguin-image {
    width: auto;
    max-width: 100%;
    height: auto;
    display: block;
    border-radius: 12px;
    margin: 0 auto;
}
@media (max-width: 900px) {
    .penguin-grid {
        grid-template-columns: 1fr;
    }
}
</style>
"""


def install_notebook_helpers(namespace, *, model, processor, repo_root, assets_dir, suppress_output):
    repo_root = Path(repo_root)
    assets_dir = Path(assets_dir)

    display(HTML(NOTEBOOK_STYLE))

    def infer(conversation, max_new_tokens=2048, temperature=0.1, do_sample=True):
        inputs = processor(
            conversation=conversation,
            add_system_prompt=True,
            add_generation_prompt=True,
            return_tensors="pt",
        )
        inputs = {k: v.cuda() if isinstance(v, torch.Tensor) else v for k, v in inputs.items()}
        if "pixel_values" in inputs:
            inputs["pixel_values"] = inputs["pixel_values"].to(torch.bfloat16)

        with suppress_output():
            output_ids = model.generate(
                **inputs,
                max_new_tokens=max_new_tokens,
                temperature=temperature,
                do_sample=do_sample,
            )

        if output_ids.shape[1] > inputs["input_ids"].shape[1]:
            response_ids = output_ids[:, inputs["input_ids"].shape[1] :]
        else:
            response_ids = output_ids

        response = processor.batch_decode(response_ids, skip_special_tokens=True)[0].strip()
        return response

    def _html_breaks(text: str) -> str:
        return html.escape(text).replace("\n", "<br>")

    def _image_to_data_uri(image_path: Path) -> str:
        mime = mimetypes.guess_type(str(image_path))[0] or "image/png"
        with open(image_path, "rb") as f:
            encoded = base64.b64encode(f.read()).decode("ascii")
        return f"data:{mime};base64,{encoded}"

    def _public_path(path: Path) -> str:
        try:
            return str(path.relative_to(repo_root))
        except ValueError:
            return path.name

    def _looks_like_markdown_table(text: str) -> bool:
        lines = [line.strip() for line in text.strip().splitlines() if line.strip()]
        if len(lines) < 2:
            return False
        separator = lines[1].replace("|", "").replace(":", "").replace("-", "").strip()
        return lines[0].count("|") >= 2 and separator == ""

    def _looks_like_rich_markdown(text: str) -> bool:
        stripped = text.strip()
        if not stripped:
            return False
        if "```" in stripped:
            return True
        if _looks_like_markdown_table(stripped):
            return True
        markdown_prefixes = ("# ", "## ", "### ", "#### ", "- ", "* ", "> ")
        if any(line.lstrip().startswith(markdown_prefixes) for line in stripped.splitlines()):
            return True
        return any(
            line.lstrip().startswith(f"{i}. ")
            for line in stripped.splitlines()
            for i in range(1, 10)
        )

    def _prefers_stacked_layout(answer: str) -> bool:
        lines = [line for line in answer.splitlines() if line.strip()]
        if _looks_like_rich_markdown(answer):
            return True
        if len(lines) >= 8:
            return True
        return any(len(line) > 90 for line in lines)

    def _contains_code_block(answer: str) -> bool:
        return "```" in answer

    def _render_answer_html(answer: str) -> str:
        if _looks_like_rich_markdown(answer):
            rendered_html = markdown2.markdown(
                answer,
                extras=[
                    "fenced-code-blocks",
                    "tables",
                    "break-on-newline",
                    "code-friendly",
                    "cuddled-lists",
                ],
            )
            if _looks_like_markdown_table(answer):
                return (
                    "<div class='penguin-body penguin-markdown'>"
                    f"<div class='penguin-table-wrap'>{rendered_html}</div>"
                    "</div>"
                )
            return f"<div class='penguin-body penguin-markdown'>{rendered_html}</div>"
        return f"<div class='penguin-body'>{_html_breaks(answer)}</div>"

    def show_image_panel(image_path: Path, title="Input Image", caption=None):
        caption_html = ""
        if caption:
            caption_html = f"<div class='penguin-caption'>{_html_breaks(caption)}</div>"
        display(
            HTML(
                f"""
                <div class='penguin-panel penguin-media-panel penguin-image-panel'>
                    <div class='penguin-title'>{html.escape(title)}</div>
                    <div class='penguin-image-shell'>
                        <img class='penguin-image' src='{_image_to_data_uri(image_path)}' />
                    </div>
                    {caption_html}
                </div>
                """
            )
        )

    def show_video_panel(video_path: Path, fps=1, max_frames=180):
        public_path = _public_path(video_path)
        display(
            HTML(
                f"""
                <div class='penguin-panel penguin-media-panel'>
                    <div class='penguin-label'>Input Video</div>
                    <div class='penguin-title'>{html.escape(video_path.name)}</div>
                    <div class='penguin-body'><strong>Source:</strong> {html.escape(public_path)}<br><strong>FPS:</strong> {fps}<br><strong>Max Frames:</strong> {max_frames}</div>
                    <div class='penguin-caption'>The notebook keeps video display lightweight for GitHub. Open the local file if you want to inspect the raw video while reviewing the saved outputs.</div>
                </div>
                """
            )
        )

    def show_turn(question: str, answer: str, round_id=None):
        response_label = "Model Response" if round_id is None else f"Round {round_id} Response"
        answer_html = _render_answer_html(answer)
        layout_class = "penguin-stack" if _prefers_stacked_layout(answer) else "penguin-grid"
        answer_panel_class = "penguin-panel penguin-answer-panel"
        if _contains_code_block(answer):
            answer_panel_class += " penguin-code-panel"
        display(
            HTML(
                f"""
                <div class='{layout_class}'>
                    <div class='penguin-panel penguin-question-panel'>
                        <div class='penguin-badge penguin-question-badge'><strong>Question</strong></div>
                        <div class='penguin-body'>{_html_breaks(question)}</div>
                    </div>
                    <div class='{answer_panel_class}'>
                        <div class='penguin-badge penguin-answer-badge'><strong>{html.escape(response_label)}</strong></div>
                        {answer_html}
                    </div>
                </div>
                """
            )
        )

    def run_single_image_case(title: str, image_name: str, question: str):
        image_path = assets_dir / image_name
        display(Markdown(f"### {title}"))
        show_image_panel(image_path, caption=image_name)

        conversation = [
            {
                "role": "user",
                "content": [
                    {"type": "image", "image": {"image_path": str(image_path)}},
                    {"type": "text", "text": question},
                ],
            }
        ]
        response = infer(conversation)
        show_turn(question, response)
        return response

    def run_multiround_image_case(title: str, image_name: str, questions):
        image_path = assets_dir / image_name
        display(Markdown(f"### {title}"))
        show_image_panel(image_path, caption=image_name)

        conversation = []
        answers = []
        for round_id, question in enumerate(questions, start=1):
            if round_id == 1:
                conversation.append(
                    {
                        "role": "user",
                        "content": [
                            {"type": "image", "image": {"image_path": str(image_path)}},
                            {"type": "text", "text": question},
                        ],
                    }
                )
            else:
                conversation.append({"role": "user", "content": question})

            response = infer(conversation)
            conversation.append({"role": "assistant", "content": response})
            show_turn(question, response, round_id=round_id)
            answers.append(response)

        return answers

    def run_multiround_video_case(title: str, video_name: str, questions, fps=1, max_frames=180):
        video_path = assets_dir / video_name
        display(Markdown(f"### {title}"))
        show_video_panel(video_path, fps=fps, max_frames=max_frames)

        conversation = [{"role": "system", "content": "You are a helpful assistant."}]
        answers = []
        for round_id, question in enumerate(questions, start=1):
            if round_id == 1:
                conversation.append(
                    {
                        "role": "user",
                        "content": [
                            {
                                "type": "video",
                                "video": {
                                    "video_path": str(video_path),
                                    "fps": fps,
                                    "max_frames": max_frames,
                                },
                            },
                            {"type": "text", "text": question},
                        ],
                    }
                )
            else:
                conversation.append({"role": "user", "content": question})

            response = infer(conversation)
            conversation.append({"role": "assistant", "content": response})
            show_turn(question, response, round_id=round_id)
            answers.append(response)

        return answers

    def run_mixed_case(title: str, video_name: str, image_name: str, question: str, fps=1, max_frames=180):
        video_path = assets_dir / video_name
        image_path = assets_dir / image_name
        display(Markdown(f"### {title}"))
        show_video_panel(video_path, fps=fps, max_frames=max_frames)
        show_image_panel(image_path, title="Reference Image", caption=image_name)

        conversation = [
            {
                "role": "user",
                "content": [
                    {"type": "text", "text": question},
                    {
                        "type": "video",
                        "video": {
                            "video_path": str(video_path),
                            "fps": fps,
                            "max_frames": max_frames,
                        },
                    },
                    {"type": "text", "text": "\n\nImage\n"},
                    {"type": "image", "image": {"image_path": str(image_path)}},
                ],
            }
        ]
        response = infer(conversation)
        show_turn(question, response)
        return response

    def run_text_case(title: str, question: str):
        display(Markdown(f"### {title}"))
        conversation = [{"role": "user", "content": question}]
        response = infer(conversation)
        show_turn(question, response)
        return response

    namespace.update(
        {
            "NOTEBOOK_STYLE": NOTEBOOK_STYLE,
            "infer": infer,
            "show_image_panel": show_image_panel,
            "show_video_panel": show_video_panel,
            "show_turn": show_turn,
            "run_single_image_case": run_single_image_case,
            "run_multiround_image_case": run_multiround_image_case,
            "run_multiround_video_case": run_multiround_video_case,
            "run_mixed_case": run_mixed_case,
            "run_text_case": run_text_case,
        }
    )


__all__ = ["install_notebook_helpers"]
