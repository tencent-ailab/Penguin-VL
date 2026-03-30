, <p align="center">
    <img src="assets/logo.png" width="150" style="margin-bottom: 0.2;"/>
</p>

<h3 align="center">Penguin-VL: Exploring the Efficiency Limits of VLM with LLM-based Vision Encoders</h3>

<h5 align="center">
  <a href="https://huggingface.co/tencent/Penguin-VL-2B"><img src="https://img.shields.io/badge/🤗-2B_Model-F6C343.svg" alt="Hugging Face"></a>
  <a href="https://huggingface.co/tencent/Penguin-VL-8B"><img src="https://img.shields.io/badge/🤗-8B_Model-F6C343.svg" alt="Hugging Face"></a>
  <a href="https://huggingface.co/tencent/Penguin-Encoder"><img src="https://img.shields.io/badge/🤗-Encoder-F6C343.svg" alt="Hugging Face"></a>
  <a href="https://huggingface.co/datasets/tencent/Penguin-Recap-I"><img src="https://img.shields.io/badge/🤗-Image_Dataset-F6C343.svg" alt="Hugging Face"></a>
  <a href="https://huggingface.co/datasets/tencent/Penguin-Recap-V"><img src="https://img.shields.io/badge/🤗-Video_Dataset-F6C343.svg" alt="Hugging Face"></a> <br>
  <a href="https://huggingface.co/spaces/tencent/Penguin-VL"><img src="https://img.shields.io/badge/🤗-Demo_Space_L40-F6C343.svg" alt="Hugging Face"></a>
  <a href="https://huggingface.co/spaces/lkeab/Penguin-VL-8B"><img src="https://img.shields.io/badge/🤗-Demo_Space_Zero-F6C343.svg" alt="Hugging Face"></a> <br>
  <a href="https://penguin-vl.github.io"><img src="https://img.shields.io/badge/Project-Page-blue?style=plastic&logo=Google%20chrome&logoColor=blue" alt="Project Page"></a>
  <a href="https://huggingface.co/papers/2603.06569"><img src="https://img.shields.io/badge/🤗-Paper%20In%20HF-8B5CF6.svg" alt="hf_paper"></a>
  <a href="https://arxiv.org/abs/2603.06569"><img src="https://img.shields.io/badge/Arxiv-2603.06569-B91C1C.svg?logo=arXiv" alt="arXiv"></a>
</h5>

---

## 📰 News
* **[2026.03.30]** 🔥🔥 We release **[Penguin-Recap-V](https://huggingface.co/datasets/tencent/Penguin-Recap-V)**! This dataset features multi-granularity video annotations with descriptions across three temporal scales: Dense time-level, Paragraph-level, and Video-level.
* **[2026.03.26]** 🔥🔥 The evaluation of Penguin-VL on benchmarks is now supported in **[lmms-eval](https://github.com/EvolvingLMMs-Lab/lmms-eval/)**.
* **[2026.03.20]** 🔥🔥 We release **[Penguin-Recap-I](https://huggingface.co/datasets/tencent/Penguin-Recap-I)**, our reconstructed high-quality image training data for Penguin-VL, on Hugging Face.
* **[2026.03.17]** We release **training code** for Penguin-VL, details see [§ Training](#training).
* **[2026.03.10]** Penguin-VL got **`#1 Paper of the day`** in [huggingface daily paper](https://huggingface.co/papers/date/2026-03-09).
* **[2026.03.09]** Release inference code, vLLM plugin, and Gradio demo for Penguin-VL.
* **[2026.03.09]** Release [Penguin-VL-2B](https://huggingface.co/tencent/Penguin-VL-2B), [Penguin-VL-8B](https://huggingface.co/tencent/Penguin-VL-8B), and [Penguin Vision Encoder](https://huggingface.co/tencent/Penguin-Encoder) on Hugging Face.

---

## 📌 TODO

- [x] Release our re-captioned training data - Penguin-Recap-I (Image)
- [x] Release training code
- [x] Release model checkpoint
- [x] Release inference code

---

## ✨ Overview

**Penguin-VL** is a compact vision-language model family built to study how far multimodal efficiency can be pushed by redesigning the **vision encoder**, rather than only scaling data or model size.

Most modern VLMs rely on vision encoders pretrained with large-scale **contrastive objectives** such as CLIP/SigLIP-style pretraining. Penguin-VL argues that this setup can be suboptimal for multimodal reasoning because contrastive learning favors coarse category-level invariances over the fine-grained signals needed for **OCR, document understanding, dense captioning, and complex reasoning**. Instead, Penguin-VL introduces **Penguin-Encoder**, a vision encoder **initialized from a text-only LLM**, so the visual backbone starts closer to the language model representation space and learns more data-efficiently.

<p align="center">
  <img src="assets/framework.png" alt="Penguin-VL framework overview" width="920"/>
</p>
<p align="center">
  <em>Framework overview of Penguin-VL: an LLM-initialized vision encoder, mixed-supervision pretraining, and efficient video token compression.</em>
</p>

### Highlights

- **LLM → Vision Encoder initialization (Penguin-Encoder)**  
  Initialize the vision encoder from a text-only LLM (e.g., Qwen3-0.6B), convert causal attention to **bidirectional attention**, and add **2D-RoPE** for variable-resolution vision tokens.

- **Mixed-supervision encoder pretraining**  
  Warm up the LLM-initialized encoder with a reconstruction/distillation objective under a teacher vision encoder (amplitude / direction / relation losses) to inject visual knowledge stably, then switch to high-resolution alignment.

- **Video efficiency via Temporal Redundancy-Aware (TRA) token compression**  
  Dynamically allocate token budgets across **key frames vs. intermediate frames** under a global token budget to scale to long videos more efficiently.

- **Unified training recipe**  
  A low-to-high resolution curriculum + instruction tuning strategy that balances image and video capabilities at compact scale.

---

## 📈 Results

Penguin-VL-2B delivers a strong accuracy-efficiency tradeoff across image and video benchmarks, with especially solid gains on OCR-heavy and reasoning-heavy tasks where fine-grained visual understanding matters most.

<p align="center">
  <img src="assets/2b_results.png" alt="Penguin-VL-2B benchmark results" width="980"/>
</p>
<p align="center">
  <em>Benchmark snapshot for Penguin-VL-2B across image and video evaluation suites.</em>
</p>

The released checkpoints and encoder weights are listed below.

---

## 📦 Model Zoo

| Model | Hugging Face |
| :---- | :----------- |
| **Penguin-VL-2B** | [tencent/Penguin-VL-2B](https://huggingface.co/tencent/Penguin-VL-2B) |
| **Penguin-VL-8B** | [tencent/Penguin-VL-8B](https://huggingface.co/tencent/Penguin-VL-8B) |
| **Penguin Vision Encoder** | [tencent/Penguin-Encoder](https://huggingface.co/tencent/Penguin-Encoder) |

---

## 🛠️ Environment Setup

### Requirements

- **Python** = 3.11.13 (recommended)  
- **PyTorch** ≥ 2.5 (CUDA 12.4 recommended)  
- **CUDA** ≥ 11.8  

### Installation

```bash
# Clone the repository
git clone <repo_url>
cd <repo_name>

# Recommended: create and activate a clean conda environment
conda create -n PenguinVL python=3.11.13 -y
conda activate PenguinVL

# INSTALL ffmpeg if you don't have it on your system
conda install ffmpeg -y # optional

# Install dependencies (inference + Gradio demo)
pip install -r requirements.txt

# NOTE: If you plan to use vLLM, it's recommended to install vLLM before flash-attn (see § vLLM Inference).
# Install Flash Attention (recommended for faster inference)
pip install flash-attn==2.8.3 --no-build-isolation
```

### Version Notes

| Use Case | Recommended |
| :------- | :---------- |
| **Transformers inference** | `transformers==4.51.3` |
| **vLLM inference** | Install vLLM separately (see [§ vLLM Inference](#-vllm-inference)) |

---

## 🤖 Inference (Transformers)

Use HuggingFace `AutoModelForCausalLM` + `AutoProcessor` for image, video, and text.

```bash
python inference/example_penguinvl.py
```

You can provide a customized `--model-path` argument to the script (default: `tencent/Penguin-VL-8B`). You can also set it to `tencent/Penguin-VL-2B`. Supported formats:

- **Video:** `type: "video"` with `video_path`, `fps`, `max_frames`
- **Image:** `type: "image"` with `image_path`
- **Mixed:** image + video + text in one conversation
- **Text-only:** plain text dialogue

---

## 📓 Cookbook

Checkout the inference notebook for a GitHub-friendly walkthrough of Penguin-VL across diverse tasks.  
Unlike a multi-notebook cookbook, Penguin-VL currently provides **one consolidated notebook** that covers multiple representative examples in a single place.

| Notebook | Description |
| :------- | :---------- |
| [Inference Recipes](inference/notebooks/01_penguinvl_inference_recipes.public.ipynb) | Demonstrations of Penguin-VL for **visual code generation**, **OCR/document parsing**, **creative image understanding**, **table extraction**, **multi-round chart analysis**, **multi-round video understanding**, **mixed video+image prompting**, and a **text-only baseline**. |

If you want to re-execute the notebook locally and regenerate the GitHub-previewable output:

```bash
export PENGUIN_VL_MODEL_PATH=tencent/Penguin-VL-8B

jupyter nbconvert \
  --to notebook \
  --execute \
  --output 01_penguinvl_inference_recipes.public.ipynb \
  --ExecutePreprocessor.timeout=-1 \
  --ExecutePreprocessor.kernel_name=penguinvl \
  inference/notebooks/01_penguinvl_inference_recipes.source.ipynb
```

The clean source notebook lives at [inference/notebooks/01_penguinvl_inference_recipes.source.ipynb](inference/notebooks/01_penguinvl_inference_recipes.source.ipynb).

---

## 🤗 Gradio Demo (Local UI)

Launch a local web UI with image/video upload and chat.

### Quick Start

```bash
python inference/launch_gradio_demo.py --model-path tencent/Penguin-VL-8B
```

Then open **http://localhost:33666** (or your machine’s IP + port) in a browser.

### Options

| Option | Description | Default |
| :----- | :----------- | :------ |
| `--model-path` | Model path or HuggingFace ID | *required* |
| `--server-port` | Backend inference server port | 16667 |
| `--interface-port` | Gradio web UI port | 33666 |
| `--nproc` | Number of backend worker processes | 1 |

### Examples

```bash
# 2B model, default ports
python inference/launch_gradio_demo.py --model-path tencent/Penguin-VL-2B

# 8B model, custom UI port
python inference/launch_gradio_demo.py --model-path tencent/Penguin-VL-8B --interface-port 8080

# Multi-worker backend
python inference/launch_gradio_demo.py --model-path tencent/Penguin-VL-8B --nproc 4
```

---

## ⚡ vLLM Inference

> Installing **vLLM 0.11.0** requires **PyTorch 2.8** and the corresponding compatible version of **Flash Attention**. This setup may different from the default Transformers inference environment (which recommends PyTorch ≥ 2.5). To avoid version conflicts, you may need to create a separate environment or upgrade dependencies accordingly.  
> **Install order note:** if you plan to use vLLM, it's recommended to install **vLLM first**, and then install **Flash Attention**.

### Environment

- The vLLM plugin targets **vLLM 0.11.0** (`penguinvl/plugin/vllm/v0_11_0/`).
- vLLM is not in `requirements.txt` by default; install it separately:

```bash
pip install vllm==0.11.0
```

### Troubleshooting

- **Flash Attention / `flash-attn` import errors** (e.g., `ImportError: ... undefined symbol: ...`): try reinstalling `flash-attn`:

```bash
pip uninstall flash-attn
pip install flash-attn --no-cache --no-build-isolation
```

- **`cannot find -lcuda` during flashinfer build**:

```bash
export LIBRARY_PATH=/usr/lib/x86_64-linux-gnu:$LIBRARY_PATH
# or /usr/local/cuda/lib64 depending on your CUDA install
```

### Start vLLM Server

```bash
# Single GPU
python -m penguinvl.plugin.vllm serve tencent/Penguin-VL-8B

# Multi-GPU (e.g. 8B on 2 GPUs)
python -m penguinvl.plugin.vllm serve tencent/Penguin-VL-8B --port 8000 --tensor-parallel-size 2
```

Additional options: `--host`, `--max-model-len`, etc. (see vLLM 0.11 `serve` docs).

### vLLM Demo Script

Run text, image, video, and batch demos:

```bash
# All demos (single GPU)
CUDA_VISIBLE_DEVICES=0 python inference/test_vllm_infer.py --model-path tencent/Penguin-VL-8B

# Text-only
CUDA_VISIBLE_DEVICES=0 python inference/test_vllm_infer.py --model-path tencent/Penguin-VL-8B --demo text

# Image (requires --image-path)
CUDA_VISIBLE_DEVICES=0 python inference/test_vllm_infer.py --model-path tencent/Penguin-VL-8B --demo image --image-path assets/inputs/horse_poet.png

# Video
CUDA_VISIBLE_DEVICES=0 python inference/test_vllm_infer.py --model-path tencent/Penguin-VL-8B --demo video --video-path assets/inputs/polar_bear.mp4

# 8B with tensor parallelism (2 GPUs)
CUDA_VISIBLE_DEVICES=0,1 python inference/test_vllm_infer.py --model-path tencent/Penguin-VL-8B --tensor-parallel-size 2
```

| Argument | Description |
| :------- | :---------- |
| `--model-path` | HuggingFace model name or local path |
| `--demo` | `text` \| `image` \| `batch` \| `video` \| `all` |
| `--tensor-parallel-size` | Number of GPUs for tensor parallelism |
| `--max-new-tokens` | Max tokens to generate |
| `--max-model-len` | Max context length |
| `--gpu-memory-utilization` | GPU memory fraction (0–1) |

---
<a id="training"></a>
## 🗝️ Training

### Training Data

We release **Penguin-Recap-I** as the public image training data accompanying Penguin-VL:
[https://huggingface.co/datasets/tencent/Penguin-Recap-I](https://huggingface.co/datasets/tencent/Penguin-Recap-I)

The release currently covers the image-side recap data and contains three subsets:
- `datacomp_coyo_penguin`
- `sa1b_penguin`
- `openimages_penguin`

For `datacomp_coyo_penguin`, we provide the original image URLs in each JSON entry for downloading.

For `sa1b_penguin` and `openimages_penguin`, we provide the training annotations together with image file names / relative paths, so users can map each sample back to the original image resources and download the raw images from OpenDataLab or the official sources:

- OpenDataLab OpenImagesV6: https://opendatalab.com/OpenDataLab/OpenImagesV6/tree/main/raw
- OpenDataLab SA-1B: https://opendatalab.com/OpenDataLab/SA-1B/tree/main/raw
- Official Segment Anything / SA-1B: https://ai.meta.com/datasets/segment-anything/
- Official OpenImages: https://storage.googleapis.com/openimages/web/index.html

### Training Pipeline Overview


Penguin-VL adopts a **4-stage curriculum**:

| Stage | Script | Description | Trainable Modules |
| :---- | :----- | :---------- | :---------------- |
| **Stage 1** | `vision_encoder_pretrain.sh` | Vision encoder warm-up with reconstruction / distillation losses. The LLM-initialized encoder learns to extract visual features under supervision from a VideoLLaMA3 vision encoder teacher. | Vision encoder + projector |
| **Stage 2** | `vision_encoder_pretrain_hres.sh` | High-resolution alignment. Continues from Stage 1 with higher sequence budgets to handle dense text and document images. | All parameters |
| **Stage 3** | `pretrain.sh` | Full multi-modal pre-training on large-scale image and video corpora. | All parameters |
| **Stage 4** | `sft.sh` | Supervised fine-tuning (instruction tuning) on high-quality chat/task data. | All parameters |

---

### Step 1: Prepare Training Data

Organize all images and videos under a single `data_root` directory:

```bash
data_root/
├── images/
│   ├── image_0001.jpg
│   └── ...
├── videos/
│   ├── video_0001.mp4
│   └── ...
├── annotations_image.jsonl
├── annotations_video.jsonl
└── ...
```

Each annotation file is a JSONL file where every line is a JSON object in the following format:

**Image example:**
```json
{
    "id": "sample_0001",
    "image": ["images/image_0001.jpg"],
    "conversations": [
        {"from": "human", "value": "<image>\nWhat is shown in the image?"},
        {"from": "gpt",   "value": "The image shows a golden retriever playing on a beach."}
    ]
}
```

**Video example:**
```json
{
    "id": "sample_0002",
    "video": ["videos/video_0001.mp4"],
    "conversations": [
        {"from": "human", "value": "<video>\nBriefly describe what happens in the video."},
        {"from": "gpt",   "value": "A person assembles a bicycle in a garage, checking each component carefully."}
    ]
}
```

**Text-only example:**
```json
{
    "id": "sample_0003",
    "conversations": [
        {"from": "human", "value": "What is the capital of France?"},
        {"from": "gpt",   "value": "The capital of France is Paris."}
    ]
}
```

> **Notes**
> - Multiple annotation files can be passed simultaneously to `--data_path`.
> - If `<image>` / `<video>` tokens are absent from the first user turn, they will be prepended automatically.
> - Both `.json` (list) and `.jsonl` (one object per line) formats are supported. `.jsonl` with HuggingFace `datasets` is recommended for large corpora.

---

### Step 1.5: (Optional) Pre-compute Sequence Lengths

`penguinvl/tools/calculate_seqlen.py` is a preprocessing utility that runs **before training** to pre-compute the approximate sequence length of every sample in a JSONL annotation file. The resulting length index can be passed to `--data_lengths_path` so the dataloader can sort samples by length, reducing padding waste and speeding up training.

The script runs in two phases internally:

1. **Metadata extraction** — resolves each sample's image resolution (via PIL) or video dimensions / frame count (via `ffprobe`), then writes an enriched `<input>_meta.jsonl` with `width`, `height`, and `frames` fields added to each record.
2. **Length estimation** — tokenizes all conversation text with the specified tokenizer and adds an estimated visual token count based on the resolution, then saves a length-sorted index tensor to `lengths.pt`.

Both phases run in parallel across all available CPU cores.

#### Usage

```bash
python penguinvl/tools/calculate_seqlen.py \
    --input  /path/to/annotations.jsonl \
    --root   /path/to/data_root \
    --tk-path Qwen/Qwen3-0.6B \
    --fps 1 \
    --max-frames 180
```

| Argument | Description | Default |
| :------- | :---------- | :------ |
| `--input` / `-i` | Input JSONL annotation file | *required* |
| `--root` / `-r` | Root directory for resolving image/video paths | `""` |
| `--tk-path` | Tokenizer path or HuggingFace model ID used for text length estimation | `Qwen/Qwen3-0.6B` |
| `--fps` | Frame rate used to estimate the number of video frames | `1` |
| `--max-frames` | Maximum frame count cap for video length estimation | `180` |
| `--chunksize` | Lines per worker chunk for `imap_unordered` | `100` |

#### Outputs

| File | Description |
| :--- | :---------- |
| `<input>_meta.jsonl` | Copy of the input JSONL with `width`, `height`, and `frames` fields added to each record. |
| `<root>/lengths.pt` | A `torch.LongTensor` of **length-sorted sample indices**. Pass this to `--data_lengths_path` in the training script. |

#### Connecting to the Training Script

After generating `lengths.pt`, add `--data_lengths_path` and `--group_by_modality_length True` to your training script:

```bash
--group_by_modality_length True
--data_lengths_path /path/to/data_root/lengths.pt
```

This enables length-sorted batching, which significantly reduces padding overhead when training on datasets with high length variance (e.g. mixed image + video data).

---

### Step 2: Configure Training Scripts

Training scripts live in `scripts/train/`. Edit the following variables at the top of each script before launching:

| Variable | Description | Example |
| :------- | :---------- | :------ |
| `DATA_DIR` | Root directory of your dataset | `/data/penguinvl_data` |
| `OUTP_DIR` | Root directory for checkpoints | `work_dirs` |
| `WANDB_PROJECT` | W&B project name | `penguinvl_qwen3_exp` |
| `ARG_WORLD_SIZE` | Number of nodes | `1` |
| `ARG_NPROC_PER_NODE` | Number of GPUs per node | `8` |
| `GLOBAL_BATCH_SIZE` | Effective global batch size | `128` |
| `LOCAL_BATCH_SIZE` | Per-GPU batch size | `2` |

Gradient accumulation is derived automatically:
```
GRADIENT_ACCUMULATION_STEPS = GLOBAL_BATCH_SIZE / (WORLD_SIZE × NPROC_PER_NODE × LOCAL_BATCH_SIZE)
```

---

### Step 3: Run Training

#### Stage 1 — Vision Encoder Pretraining

```bash
bash scripts/train/vision_encoder_pretrain.sh [NUM_NODES] [NUM_GPUS_PER_NODE]
```

Key arguments specific to Stage 1:
```bash
--model_path        Qwen/Qwen3-1.7B                            # LLM part
--vision_encoder    Cyril666/SFL-Encoder-Pretrained-Qwen3      # LLM-initialized vision encoder (converted from Qwen/Qwen-0.6B and modified the layer parameter names.)
--use_reconstruct   True                                       # Enable Stage 1 reconstruction / distillation loss
--vision_encoder_teacher DAMO-NLP-SG/VL3-SigLIP-NaViT          # VideoLLaMA3 vision encoder teacher checkpoint
--model_max_length  4096
--mm_max_length     2048
```

#### Stage 2 — High-Resolution Encoder Pretraining

```bash
bash scripts/train/vision_encoder_pretrain_hres.sh [NUM_NODES] [NUM_GPUS_PER_NODE]
```

Loads from `stage_1` checkpoint. Increases context budgets for high-resolution inputs:
```bash
--model_max_length  16384
--mm_max_length     10240
```

#### Stage 3 — Full Pre-training

```bash
bash scripts/train/pretrain.sh [NUM_NODES] [NUM_GPUS_PER_NODE]
```

Loads from `stage_2` checkpoint. All three modules (vision encoder, projector, LLM) are jointly trained.

#### Stage 4 — Supervised Fine-tuning

```bash
bash scripts/train/sft.sh [NUM_NODES] [NUM_GPUS_PER_NODE]
```

Loads from `stage_3` checkpoint. Uses high-quality instruction-following data for final alignment.

---

### Key Training Arguments Reference

| Argument | Description | Default |
| :------- | :---------- | :------ |
| `--model_type` | Model architecture type | `penguinvl_qwen3` |
| `--model_path` | Path to LLM backbone or previous stage checkpoint | — |
| `--vision_encoder` | Path or HF ID of the vision encoder | — |
| `--vision_projector_type` | Projector architecture | `mlp2x_gelu` |
| `--use_reconstruct` | Enable the Stage 1 visual reconstruction / distillation loss | `False` |
| `--vision_encoder_teacher` | VideoLLaMA3 vision encoder teacher checkpoint | `None` |
| `--data_path` | Space-separated list of annotation files | — |
| `--data_folder` | Root folder for all media files | — |
| `--fps` | Video sampling frame rate | `1` |
| `--max_frames` | Maximum number of frames per video | `180` |
| `--image_merge_size` | Token merge factor for images | `1` |
| `--video_merge_size` | Token merge factor for video frames | `2` |
| `--model_max_length` | Maximum total sequence length (truncation) | `512` |
| `--mm_max_length` | Maximum visual token budget per sample | `10240` |
| `--llm_lr` | Learning rate for the LLM backbone | `None` |
| `--vision_encoder_lr` | Learning rate for the vision encoder | `None` |
| `--vision_projector_lr` | Learning rate for the MLP projector | `None` |
| `--embedding_lr` | Learning rate for embedding layers | `None` |
| `--deepspeed` | DeepSpeed config path | `scripts/zero1.json` |
| `--gradient_checkpointing` | Enable gradient checkpointing | `True` |
| `--use_batch_flattening` | Flatten variable-length sequences in a batch | `True` |

---

### Distributed Training (Multi-node)

The scripts support multi-node training via `torchrun`. Pass `WORLD_SIZE`, `NPROC_PER_NODE`, `MASTER_ADDR`, `MASTER_PORT`, and `RANK` as environment variables or positional arguments:

```bash
# Node 0 (master)
WORLD_SIZE=2 NPROC_PER_NODE=8 MASTER_ADDR=<node0_ip> MASTER_PORT=16667 RANK=0 \
    bash scripts/train/sft.sh

# Node 1
WORLD_SIZE=2 NPROC_PER_NODE=8 MASTER_ADDR=<node0_ip> MASTER_PORT=16667 RANK=1 \
    bash scripts/train/sft.sh
```

---

## 📁 Project Structure

```text
.
├── penguinvl/                    # Core model and processor code
│   ├── plugin/vllm/              # vLLM plugin (v0_11_0)
│   ├── tools/                    # Tool scripts
│   └── train/                    # Training code
├── scripts/                      # Training scripts
├── inference/
│   ├── example_penguinvl.py      # Transformers inference example
│   ├── test_vllm_infer.py        # vLLM inference demo
│   ├── launch_gradio_demo.py     # Gradio local demo
│   ├── notebooks/                # Executed and source Jupyter notebooks
│   ├── server/                   # Backend for Gradio
│   ├── interface/                # Gradio UI
│   └── transformers_api/         # Transformers model/processor wrappers
├── assets/
│   ├── framework.png             # README framework figure
│   ├── 2b_results.png            # README benchmark figure
│   └── inputs/                   # Demo images and videos
└── requirements.txt
```

---

## 📄 License

This project is released under the [Apache 2.0 License](LICENSE).

## 📚 Citation

If you use Penguin-VL in your research, please cite:

```bibtex
@article{Penguin-VL,
  title={Penguin-VL: Exploring the Efficiency Limits of VLM with LLM-based Vision Encoders},
  author={Boqiang Zhang and Lei Ke and Ruihan Yang and Qi Gao and Tianyuan Qu and Rossell Chen and Dong Yu and Leoweiliang},
  journal={arXiv preprint arXiv:2603.06569},
  year={2026}
}
```

---

If you find this project useful, please consider giving it a ⭐ on GitHub. Issues and PRs are welcome.
