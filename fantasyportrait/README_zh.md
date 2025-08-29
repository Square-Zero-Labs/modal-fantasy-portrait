[English](./README.md)
# FantasyPortrait：基于表情增强扩散变换器的多角色肖像动画生成

[![Home Page](https://img.shields.io/badge/Project-FantasyPortrait-blue.svg)](https://fantasy-amap.github.io/fantasy-portrait/)
[![arXiv](https://img.shields.io/badge/Arxiv-2507.12956-b31b1b.svg?logo=arXiv)](https://arxiv.org/abs/2507.12956)
[![hf_dataset](https://img.shields.io/badge/🤗%20Dataset-FantasyPortrait-yellow.svg)](https://huggingface.co/datasets/acvlab/FantasyPortrait-Multi-Expr)
[![hf_model](https://img.shields.io/badge/🤗%20Model-FantasyPortrait-green.svg)](https://huggingface.co/acvlab/FantasyPortrait)
[![hf_paper](https://img.shields.io/badge/🤗-FantasyPortrait-red.svg)](https://huggingface.co/papers/2507.12956)
[![ms_model](https://img.shields.io/badge/ModelScope-Model-9cf.svg)](https://modelscope.cn/models/amap_cvlab/FantasyPortrait)
[![ms_dataset](https://img.shields.io/badge/ModelScope-Dataset-ff69b4.svg)](https://www.modelscope.cn/datasets/amap_cvlab/FantasyPortrait-Multi-Expr)

## 🔥 最新动态！！
* 2025年8月14日：我们的工作被加入到[ComfyUI-Wan](https://github.com/kijai/ComfyUI-WanVideoWrapper/tree/main/fantasyportrait) ! 感谢 [kijai](https://github.com/kijai) 更新 👏!
* 2025年8月12日：我们已发布推理代码、模型权重和数据集。

## 演示
更多有趣的结果，请访问我们的[网站](https://fantasy-amap.github.io/fantasy-portrait/)。

| ![单人示例](./assert/demo/danren_1.gif) | ![对比](./assert/demo/duibi.gif) |
| :---: | :---: |
| ![动物](./assert/demo/dongwu.gif) | ![双人1](./assert/demo/shuangren_1.gif) |
| ![双人2](./assert/demo/shuangren_2.gif) | ![三人](./assert/demo/sanren.gif) |

## 快速开始
### 🛠️ 安装

克隆仓库：

```
git clone https://github.com/Fantasy-AMAP/fantasy-portrait.git
cd fantasy-portrait
```

安装依赖：
```
apt-get install ffmpeg
# 确保 torch >= 2.0.0
pip install -r requirements.txt
# 注意：必须安装 flash attention
pip install flash_attn
```

### 📦 Multi-Expr 数据集
我们公开了首个多人肖像面部表情视频数据集 **Multi-Expr Dataset**，请通过这个[ModelScope](https://www.modelscope.cn/datasets/amap_cvlab/FantasyPortrait-Multi-Expr)或者[Huggingface](https://huggingface.co/datasets/acvlab/FantasyPortrait-Multi-Expr)下载。


### 🧱 模型下载
| 模型        |                       下载链接                                           |    说明                      |
| --------------|-------------------------------------------------------------------------------|-------------------------------|
| Wan2.1-I2V-14B-720P  |      🤗 [Huggingface](https://huggingface.co/Wan-AI/Wan2.1-I2V-14B-720P)    🤖 [ModelScope](https://www.modelscope.cn/models/Wan-AI/Wan2.1-I2V-14B-720P)     | 基础模型
| FantasyPortrait      |      🤗 [Huggingface](https://huggingface.co/acvlab/FantasyPortrait/)     🤖 [ModelScope](https://www.modelscope.cn/models/amap_cvlab/FantasyPortrait/)         | 我们的表情条件权重

使用 huggingface-cli 下载模型：
``` sh
pip install "huggingface_hub[cli]"
huggingface-cli download Wan-AI/Wan2.1-I2V-14B-720P --local-dir ./models/Wan2.1-I2V-14B-720P
huggingface-cli download acvlab/FantasyPortrait --local-dir ./models
```

使用 modelscope-cli 下载模型：
``` sh
pip install modelscope
modelscope download Wan-AI/Wan2.1-I2V-14B-720P --local_dir ./models/Wan2.1-I2V-14B-720P
modelscope download amap_cvlab/FantasyPortrait  --local_dir ./models
```

### 🔑 单人肖像推理
``` sh
bash infer_single.sh
```

### 🔑 多人肖像推理
如果你使用多人的输入图像和多人的驱动视频，您可以运行如下脚本：
``` sh
bash infer_multi.sh
```

如果您使用多人输入图像和不同的多个单人驱动的视频，您可以运行如下脚本：
```sh
bash infer_multi_diff.sh
```

### 📦 速度与显存占用
我们在此提供详细表格。模型在单张A100上进行测试。

|`torch_dtype`|`num_persistent_param_in_dit`|速度|所需显存|
|-|-|-|-|
|torch.bfloat16|None (无限制)|15.5秒/迭代|40G|
|torch.bfloat16|7*10**9 (7B)|32.8秒/迭代|20G|
|torch.bfloat16|0|42.6秒/迭代|5G|



## 🧩 社区贡献
我们 ❤️ 来自开源社区的贡献！如果您的工作改进了 FantasyPortrait，请告知我们。
您也可以直接发送邮件至 [frank.jf@alibaba-inc.com](mailto://frank.jf@alibaba-inc.com)。我们很乐意引用您的项目，方便大家使用。

## 🔗 引用
如果本仓库对您有帮助，请考虑给我们一个 star ⭐ 并引用以下论文：
```
@article{wang2025fantasyportrait,
  title={FantasyPortrait: Enhancing Multi-Character Portrait Animation with Expression-Augmented Diffusion Transformers},
  author={Wang, Qiang and Wang, Mengchao and Jiang, Fan and Fan, Yaqi and Qi, Yonggang and Xu, Mu},
  journal={arXiv preprint arXiv:2507.12956},
  year={2025}
}
```

## 致谢
感谢 [Wan2.1](https://github.com/Wan-Video/Wan2.1)、[PD-FGC](https://github.com/Dorniwang/PD-FGC-inference) 和 [DiffSynth-Studio](https://github.com/modelscope/DiffSynth-Studio) 开源他们的模型和代码，为本项目提供了宝贵的参考和支持。我们非常感谢他们对开源社区的贡献。

## Star 历史

[![Star History Chart](https://api.star-history.com/svg?repos=Fantasy-AMAP/fantasy-portrait&type=Date)](https://www.star-history.com/#Fantasy-AMAP/fantasy-portrait&Date)
