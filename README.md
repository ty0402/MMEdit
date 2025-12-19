# MMEDIT

[![arXiv](https://img.shields.io/badge/arXiv-25xx.xxxxx-brightgreen.svg?style=flat-square)](https://arxiv.org/abs/25xx.xxxxx)
[![Project Page](https://img.shields.io/badge/Project%20Page-Demo-purple?style=flat-square)](https://ty0402.github.io/MMEditing/)
[![Hugging Face Models](https://img.shields.io/badge/%F0%9F%A4%97%20Hugging%20Face-Models-blue?style=flat-square)](https://huggingface.co/CocoBro/MMEdit)
[![License](https://img.shields.io/badge/License-Apache%202.0-yellow?style=flat-square)](./LICENSE)


## Introduction
🟣 **MMEDIT** is a state-of-the-art audio generation model built upon the powerful [Qwen2-Audio 7B](https://huggingface.co/Qwen/Qwen2-Audio-7B). It leverages the robust audio understanding and instruction-following capabilities of the large language model to achieve precise and high-fidelity audio editing.

---
## Model Download
| Models   | 🤗 Hugging Face |
|-------|-------|
| MMEdit| [MMEdit](https://huggingface.co/CocoBro/MMEdit) |

download our pretrained model into ./ckpt/mmedit/

---

## Model Usage
### 🔧 Dependencies and Installation
- Python >= 3.10
- [PyTorch >= 2.5.0](https://pytorch.org/)
- [CUDA Toolkit](https://developer.nvidia.com/cuda-downloads)
- Dependent models:
  - [Qwen/Qwen2-Audio-7B-Instruct](https://huggingface.co/Qwen/Qwen2-Audio-7B-Instruct), download into `./ckpt/qwen2-audio-7B-Instruct/`

```bash
# 1. Clone the repository
git clone https://github.com/xycs6k8r2Anonymous/MMEdit.git
cd MMEDIT

# 2. Create environment
conda create -n mmedit python=3.10 -y
conda activate mmedit

# 3. Install PyTorch and dependencies
pip install torch==2.5.1 torchvision==0.20.1 torchaudio==2.5.1 --index-url https://download.pytorch.org/whl/cu121
pip install -r requirements.txt

# Download Qwen2-Audio-7B-Instruct
huggingface-cli download Qwen/Qwen2-Audio-7B-Instruct --local-dir ./ckpt/qwen2-audio-7B-instruct

# Download MMEdit (Our Model)
huggingface-cli download CocoBro/MMEdit --local-dir ./ckpt/mmedit
```

## 📂 Data Preparation

For detailed instructions on the data pipeline, and dataset structure used for training, please refer to our separate documentation:

👉 **[Data Pipeline & Preparation Guide](./datapipeline/datapipeline.md)**


## ⚡ Quick Start




### 1. Inference
You can quickly generate example audio with the following code:

```
bash bash_scripts/infer_single.sh
```

The output will be save at inference/example


---

## 🚀 Usage

### 1. Configuration
Before running inference or training, please check `configs/config.yaml`. The project uses `hydra` for configuration management, allowing easy overrides via command line.

### 2. Inference
To run batch inference using the provided scripts:

```bash
cd src
bash bash_scripts/inference.sh
```

### 3. Training
Ensure you have downloaded the **Qwen2-Audio-7B-Instruct** checkpoint to `./ckpt/qwen2-audio-7B-instruct` and prepared your data according to the [Data Pipeline Guide](./docs/DATA_PIPELINE.md).

```bash
cd src
# Launch distributed training
bash bash_scripts/train_dist.sh
```

---

## 📝 Todo
- [ ] Release inference code and checkpoints.
- [ ] Release training scripts.
- [ ] Add HuggingFace Gradio Demo.
- [ ] Release evaluation metrics and post-processing tools.

## 🤝 Acknowledgement
We thank the following open-source projects for their inspiration and code:
* [Qwen2-Audio](https://github.com/QwenLM/Qwen2-Audio)
* [Uniflowaudio](https://github.com/wsntxxn/UniFlow-Audio)
* [AudioTime](https://github.com/wsntxxn/UniFlow-Audio)


## 🖊️ Citation
If you find this project useful, please cite our paper:

```bibtex
@article{mmedit2024,
  title={MMEDIT: Audio Generation based on Qwen2-Audio 7B},
  author={Your Name and Collaborators},
  journal={arXiv preprint arXiv:25xx.xxxxx},
  year={2024}
}
```
