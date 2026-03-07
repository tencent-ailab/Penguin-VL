import base64
import html
import mimetypes
from io import BytesIO
from pathlib import Path

import torch
from IPython.display import HTML, Markdown, display

try:
    from PIL import Image as PILImage
except ImportError:  # pragma: no cover - pillow is expected but optional
    PILImage = None


NOTEBOOK_STYLE = ""


def install_notebook_helpers(namespace, *, model, processor, repo_root, assets_dir, suppress_output):
    repo_root = Path(repo_root)
    assets_dir = Path(assets_dir)

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

    def _public_path(path: Path) -> str:
        try:
            return str(path.relative_to(repo_root))
        except ValueError:
            return path.name

    def _image_to_data_uri(image_path: Path, max_width=450) -> str:
        mime = mimetypes.guess_type(str(image_path))[0] or "image/png"

        if PILImage is None:
            with open(image_path, "rb") as f:
                payload = f.read()
            encoded = base64.b64encode(payload).decode("ascii")
            return f"data:{mime};base64,{encoded}"

        with PILImage.open(image_path) as image:
            resized = image.copy()
            if resized.width > max_width:
                resampling = getattr(PILImage, "Resampling", PILImage)
                resized.thumbnail((max_width, 10_000), resampling.LANCZOS)

            output = BytesIO()
            format_name = image.format if image.format in {"PNG", "JPEG", "WEBP"} else "PNG"
            if format_name == "JPEG" and resized.mode not in ("RGB", "L"):
                resized = resized.convert("RGB")
            resized.save(output, format=format_name)

        mime = {
            "PNG": "image/png",
            "JPEG": "image/jpeg",
            "WEBP": "image/webp",
        }[format_name]
        encoded = base64.b64encode(output.getvalue()).decode("ascii")
        return f"data:{mime};base64,{encoded}"

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

    def _blockquote(text: str) -> str:
        lines = html.escape(text).splitlines() or [""]
        return "\n".join([f"> {line}" if line else ">" for line in lines])

    def _render_answer_markdown(answer: str) -> str:
        if _looks_like_rich_markdown(answer):
            return answer.strip()
        return _blockquote(answer)

    def show_image_panel(image_path: Path, title="Input Image", caption=None):
        public_path = _public_path(image_path)
        alt_text = html.escape(image_path.name)
        data_uri = _image_to_data_uri(image_path)

        display(Markdown(f"**{title}**"))
        display(HTML(f"<img src='{data_uri}' alt='{alt_text}' width='450' />"))

        if caption:
            display(Markdown(f"`{public_path}`"))

    def show_video_panel(video_path: Path, fps=1, max_frames=180):
        public_path = _public_path(video_path)
        display(
            Markdown(
                "\n".join(
                    [
                        "**Input Video**",
                        "",
                        f"`{video_path.name}`",
                        "",
                        f"- Source: `{public_path}`",
                        f"- FPS: `{fps}`",
                        f"- Max Frames: `{max_frames}`",
                    ]
                )
            )
        )

    def show_turn(question: str, answer: str, round_id=None):
        response_label = "Model Response" if round_id is None else f"Round {round_id} Response"
        answer_markdown = _render_answer_markdown(answer)
        turn_markdown = "\n".join(
            [
                "**Question**",
                "",
                _blockquote(question),
                "",
                f"**{response_label}**",
                "",
                answer_markdown,
                "",
                "---",
            ]
        )
        display(Markdown(turn_markdown))

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
