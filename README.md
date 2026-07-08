<div align="center">

<h1>LingBot-Vision: Vision Pretraining for Dense Spatial Perception</h1>

</div>

<div align="center">

[![PDF](https://img.shields.io/static/v1?label=Paper&message=PDF&color=red&logo=adobeacrobatreader)](https://github.com/Robbyant/lingbot-vision/blob/main/paper.pdf)
[![arXiv](https://img.shields.io/static/v1?label=Paper&message=arXiv&color=lightgrey&logo=arxiv)](https://arxiv.org/abs/2607.05247)
[![Project](https://img.shields.io/badge/Project-Website-blue)](https://technology.robbyant.com/lingbot-vision)
[![HuggingFace](https://img.shields.io/static/v1?label=%F0%9F%A4%97%20Model&message=HuggingFace&color=orange)](https://huggingface.co/collections/robbyant/lingbot-vision)
[![ModelScope](https://img.shields.io/static/v1?label=%F0%9F%A4%96%20Model&message=ModelScope&color=purple)](https://www.modelscope.cn/collections/Robbyant/LingBot-Vision)
[![License](https://img.shields.io/badge/License-Apache--2.0-green)](LICENSE)

</div>

-----

### 🔭 Meet LingBot-Vision! A family of self-supervised ViT backbones for dense spatial perception! 🖼️📐

**LingBot-Vision** is a family of self-supervised Vision Transformer backbones for dense spatial perception, from ViT-S/16 up to a 1.1B-parameter ViT-g/16. The flagship model is pretrained with **masked boundary modeling** — a boundary-centric objective that encourages spatially structured patch features while retaining strong semantic representations.

![Boundary-centric masked modeling](./assets/teaser_crop.png)

<sub>**Boundary-centric masked modeling.** Each row shows the input image, the PCA projection of frozen patch tokens, teacher-discovered boundary tokens, and cosine-similarity maps from selected boundary-token queries. The features capture semantic grouping and geometric structure at the same time.</sub>

LingBot-Vision learns boundaries, shapes, and semantic regions all together, making it a drop-in visual encoder for dense downstream tasks:

- 🎨 **Dense feature visualization** — PCA maps of frozen patch tokens reveal coherent object regions and crisp boundaries
- 📏 **Depth estimation** — frozen patch tokens expose spatial structure to lightweight dense readouts
- 🧩 **Semantic segmentation** — boundary-faithful features align region transitions with object contours
- 🎬 **Video object segmentation** — training-free token matching and label propagation with frozen features
- 🤖 **Depth completion** — LingBot-Vision is the visual encoder initialization for LingBot-Depth 2.0 (see below)

## 🌊 Meet LingBot-Depth 2.0

By simply replacing the encoder with LingBot-Vision at the ViT-L/16 and ViT-g/16 scales, and scaling the curated RGB-D training corpus from 3M to 150M samples, LingBot-Depth 2.0 achieves substantial performance gains over the previous and other system, as detailed in the technical report.

![LingBot-Depth 2.0 on mirror and glass scenes](./assets/lingbot_depth2_mirror_glass.jpg)

<sub>**LingBot-Depth 2.0 on mirror and glass scenes.** Each group shows input RGB, raw sensor depth, refined depth, and refined point clouds. Raw depth is missing on difficult surfaces such as window panes, glass balustrades, and reflective floors. LingBot-Depth 2.0 completes these regions as stable, contiguous surfaces across frames.</sub>

---

## 📦 Model Zoo

We train a ViT-g/16 teacher with roughly 1.1B parameters and distill ViT-L, ViT-B, and ViT-S backbones from it for inference and downstream use. Full training and evaluation details are covered in the technical report.

All released weights are **backbone-only** `.pt` checkpoints, stored as `model.pt` in each model repository (see the full [Hugging Face collection](https://huggingface.co/collections/robbyant/lingbot-vision)):

| Model | Backbone | Embed dim | Hugging Face Weights | ModelScope Weights |
|-------|----------|----------:|----------------------|--------------------|
| **LingBot-Vision-Giant**<br><sub>highest-quality dense features</sub> | ViT-g/16 · SwiGLU · fp32 RoPE · 4 register tokens | 1536 | [vit-giant](https://huggingface.co/robbyant/lingbot-vision-vit-giant) | [vit-giant](https://www.modelscope.cn/models/Robbyant/lingbot-vision-vit-giant) |
| **LingBot-Vision-Large** ⭐<br><sub>recommended: strong features, practical inference</sub> | ViT-L/16, distilled from Giant | 1024 | [vit-large](https://huggingface.co/robbyant/lingbot-vision-vit-large) | [vit-large](https://www.modelscope.cn/models/Robbyant/lingbot-vision-vit-large) |
| **LingBot-Vision-Base**<br><sub>balanced inference cost</sub> | ViT-B/16, distilled from Giant | 768 | [vit-base](https://huggingface.co/robbyant/lingbot-vision-vit-base) | [vit-base](https://www.modelscope.cn/models/Robbyant/lingbot-vision-vit-base) |
| **LingBot-Vision-Small**<br><sub>lightweight demos and downstream use</sub> | ViT-S/16, distilled from Giant | 384 | [vit-small](https://huggingface.co/robbyant/lingbot-vision-vit-small) | [vit-small](https://www.modelscope.cn/models/Robbyant/lingbot-vision-vit-small) |

Config files are packaged under `lingbot_vision/configs/` and selected automatically by `load_pretrained_backbone`.

## 🔧 Installation

**Requirements**: Python ≥ 3.10 · PyTorch ≥ 2.0 · CUDA-capable GPU (recommended for large-model inference)

**1. Clone the repository**

```bash
git clone https://github.com/robbyant/lingbot-vision.git
cd lingbot-vision
```

**2. Create a conda environment**

```bash
conda create -n lingbot-vision python=3.10 -y
conda activate lingbot-vision
```

**3. Install lingbot-vision**

```bash
python -m pip install -r requirements.txt
python -m pip install -e .
```

## 🚀 Quick Start

### Load LingBot-Vision Pretrains with PyTorch

The model is automatically downloaded from Hugging Face on first use. This example uses the small model for a lightweight smoke run; `large` is the default variant, and `giant` is available as the largest backbone.

```python
import torch

from lingbot_vision import load_pretrained_backbone, extract_patch_tokens, load_image

device = "cuda" if torch.cuda.is_available() else "cpu"
dtype = torch.bfloat16 if device == "cuda" else torch.float32

# Downloads model.pt from robbyant/lingbot-vision-vit-small.
backbone, embed_dim = load_pretrained_backbone(
    variant="small",
    device=device,
    dtype=dtype,
)

img_norm, _, _ = load_image(
    "examples/example.png",
    size=512,
    patch_size=backbone.patch_size,
    mode="square",
)
patch_tokens, patch_grid = extract_patch_tokens(backbone, img_norm, device, dtype)

print(patch_tokens.shape, patch_grid, embed_dim)
# torch.Size([1, 1024, 384]) (32, 32) 384
```

`patch_tokens` has shape `[B, H * W, C]`, where `H` and `W` are the patch-grid dimensions. `variant` can be `giant`, `large`, `base`, or `small`; if omitted, it defaults to `large`. You can also pass a local directory or an explicit Hugging Face model repo to `load_pretrained_backbone`.

### Run the PCA demo

Download a backbone checkpoint from Hugging Face (or ModelScope, once available), then run:

```bash
./scripts/run_pca_demo.sh \
  --config-file lingbot_vision/configs/lbot_vision_vitl.yaml \
  --ckpt /path/to/model.pt \
  --input examples/example.png \
  --out outputs/pca_demo \
  --size 512 \
  --mode square \
  --dtype bf16
```

Images are loaded as RGB, resized according to `--size` and `--mode`, aligned to the model patch size, and normalized with ImageNet statistics. The demo maps the top three PCA components of the patch tokens to RGB and writes both PCA-only and input/PCA panel visualizations to the output directory. Use `--dtype fp32 --device cpu` for CPU-only inference.

<details>
<summary><b>All demo options</b></summary>

| Parameter | Description |
| --- | --- |
| `--config-file` | Model config file under `lingbot_vision/configs/`. |
| `--ckpt` | Local path to a pure backbone `.pt` checkpoint. |
| `--input` | Image file or directory of images. |
| `--out` | Output directory for PCA visualizations. |
| `--size` | Target input size. For ViT-g/16, 512 gives a 32 x 32 patch grid. |
| `--mode` | `square` resizes to `size` x `size` (does not preserve aspect ratio); `shortest` resizes the shortest side to `size`, then center-crops a `size` x `size` square. |
| `--dtype` | `bf16`, `fp16`, or `fp32`. |
| `--device` | PyTorch device, for example `cuda` or `cpu`. |

</details>

### Checkpoint format

Released checkpoints are `.pt` files containing backbone weights only — no optimizer states, projection heads, or training-time boundary heads. The loader accepts a raw state dict or a dictionary with a `backbone` entry:

```python
state_dict
{"backbone": state_dict}
```

If checkpoint keys are prefixed with `backbone.`, the loader strips the prefix automatically.

## 📖 Citation

```bibtex
@article{lingbot-vision2026,
  title={Vision Pretraining for Dense Spatial Perception},
  author={Fu, Zelin and Tan, Bin and Sun, Changjiang and Liu, Shaohui and Zheng, Kecheng and Xu, Yinghao and Zhu, Xing and Shen, Yujun and Xue, Nan},
  journal={arXiv preprint arXiv:2607.05247},
  year={2026}
}
```

## 📜 License

This project is released under the Apache License 2.0. See [LICENSE](LICENSE) for details.

## 🙏 Acknowledgments

LingBot-Vision is part of the LingBot spatial perception effort. We thank [DINOv2](https://github.com/facebookresearch/dinov2) and [DINOv3](https://github.com/facebookresearch/dinov3) for their contributions to self-supervised learning.

## 📮 Contact

For questions, discussions, or collaborations: 

- **Issues**: Open an [issue](https://github.com/robbyant/lingbot-vision/issues) on GitHub
- **Email**: Contact [Zelin Fu](https://github.com/TakuLingFu) (fuzelin.fzl@antgroup.com) or [Nan Xue](https://xuenan.net) (xuenan@ieee.org)
