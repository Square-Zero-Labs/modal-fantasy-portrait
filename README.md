[中文阅读](./README_zh.md)
# FantasyPortrait: Enhancing Multi-Character Portrait Animation with Expression-Augmented Diffusion Transformers

[![Home Page](https://img.shields.io/badge/Project-FantasyPortrait-blue.svg)](https://fantasy-amap.github.io/fantasy-portrait/)
[![arXiv](https://img.shields.io/badge/Arxiv-2507.12956-b31b1b.svg?logo=arXiv)](https://arxiv.org/abs/2507.12956)
[![hf_dataset](https://img.shields.io/badge/🤗%20Dataset-FantasyPortrait-yellow.svg)](https://huggingface.co/datasets/acvlab/FantasyPortrait-Multi-Expr)
[![hf_model](https://img.shields.io/badge/🤗%20Model-FantasyPortrait-green.svg)](https://huggingface.co/acvlab/FantasyPortrait)
[![hf_paper](https://img.shields.io/badge/🤗-FantasyPortrait-red.svg)](https://huggingface.co/papers/2507.12956)
[![ms_model](https://img.shields.io/badge/ModelScope-Model-9cf.svg)](https://modelscope.cn/models/amap_cvlab/FantasyPortrait)
[![ms_dataset](https://img.shields.io/badge/ModelScope-Dataset-ff69b4.svg)](https://www.modelscope.cn/datasets/amap_cvlab/FantasyPortrait-Multi-Expr)

## 🔥 Latest News!!
* August 14, 2025: Our work is merged to [ComfyUI-Wan](https://github.com/kijai/ComfyUI-WanVideoWrapper/tree/main/fantasyportrait) ! Thank [kijai](https://github.com/kijai) for the update 👏!
* August 12, 2025: We released the inference code, model weights and datasets.

## Demo
For more interesting results, please visit our [website](https://fantasy-amap.github.io/fantasy-portrait/).

| ![单人示例](./assert/demo/danren_1.gif) | ![对比](./assert/demo/duibi.gif) |
| :---: | :---: |
| ![动物](./assert/demo/dongwu.gif) | ![双人1](./assert/demo/shuangren_1.gif) |
| ![双人2](./assert/demo/shuangren_2.gif) | ![三人](./assert/demo/sanren.gif) |


## Quickstart
### 🛠️Installation

Clone the repo:

```
git clone https://github.com/Fantasy-AMAP/fantasy-portrait.git
cd fantasy-portrait
```

Install dependencies:
```
apt-get install ffmpeg
# Ensure torch >= 2.0.0
pip install -r requirements.txt
# Note: flash attention must be installed
pip install flash_attn
```

### 📦Multi-Expr Dataset
We make public the first multi-portrait facial expression video dataset **Multi-Expr Dataset**, Please download it via the [ModelScope](https://www.modelscope.cn/datasets/amap_cvlab/FantasyPortrait-Multi-Expr) or [Huggingface](https://huggingface.co/datasets/acvlab/FantasyPortrait-Multi-Expr).

### 🧱Model Download
| Models        |                       Download Link                                           |    Notes                      |
| --------------|-------------------------------------------------------------------------------|-------------------------------|
| Wan2.1-I2V-14B-720P  |      🤗 [Huggingface](https://huggingface.co/Wan-AI/Wan2.1-I2V-14B-720P)    🤖 [ModelScope](https://www.modelscope.cn/models/Wan-AI/Wan2.1-I2V-14B-720P)     | Base model
| FantasyPortrait      |      🤗 [Huggingface](https://huggingface.co/acvlab/FantasyPortrait/)     🤖 [ModelScope](https://www.modelscope.cn/models/amap_cvlab/FantasyPortrait/)         | Our emo condition weights

Download models using huggingface-cli:
``` sh
pip install "huggingface_hub[cli]"
huggingface-cli download Wan-AI/Wan2.1-I2V-14B-720P --local-dir ./models/Wan2.1-I2V-14B-720P
huggingface-cli download acvlab/FantasyPortrait --local-dir ./models
```

Download models using modelscope-cli:
``` sh
pip install modelscope
modelscope download Wan-AI/Wan2.1-I2V-14B-720P --local_dir ./models/Wan2.1-I2V-14B-720P
modelscope download amap_cvlab/FantasyPortrait  --local_dir ./models
```

### 🔑 Single-Portrait Inference
``` sh
bash infer_single.sh
```

### 🔑 Multi-Portrait Inference
If you use input image and drive videos with multiple people, you can run as follows:
``` sh
bash infer_multi.sh
```
If you use input image with multiple people and different multiple single-human driven videos, you can run as follows:
```sh
bash infer_multi_diff.sh
```

### 📦Speed and VRAM Usage
We present a detailed table here. The model is tested on a single A100.

|`torch_dtype`|`num_persistent_param_in_dit`|Speed|Required VRAM|
|-|-|-|-|
|torch.bfloat16|None (unlimited)|15.5s/it|40G|
|torch.bfloat16|7*10**9 (7B)|32.8s/it|20G|
|torch.bfloat16|0|42.6s/it|5G|



## 🧩 Community Works
We ❤️ contributions from the open-source community! If your work has improved FantasyPortrait, please inform us.
Or you can directly e-mail [frank.jf@alibaba-inc.com](mailto://frank.jf@alibaba-inc.com). We are happy to reference your project for everyone's convenience.

## 🔗Citation
If you find this repository useful, please consider giving a star ⭐ and citation
```
@article{wang2025fantasyportrait,
  title={FantasyPortrait: Enhancing Multi-Character Portrait Animation with Expression-Augmented Diffusion Transformers},
  author={Wang, Qiang and Wang, Mengchao and Jiang, Fan and Fan, Yaqi and Qi, Yonggang and Xu, Mu},
  journal={arXiv preprint arXiv:2507.12956},
  year={2025}
}
```

## Acknowledgments
Thanks to [Wan2.1](https://github.com/Wan-Video/Wan2.1), [PD-FGC](https://github.com/Dorniwang/PD-FGC-inference) and [DiffSynth-Studio](https://github.com/modelscope/DiffSynth-Studio) for open-sourcing their models and code, which provided valuable references and support for this project. Their contributions to the open-source community are truly appreciated.

## Star History

[![Star History Chart](https://api.star-history.com/svg?repos=Fantasy-AMAP/fantasy-portrait&type=Date)](https://www.star-history.com/#Fantasy-AMAP/fantasy-portrait&Date)

