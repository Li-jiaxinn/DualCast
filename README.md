<h3 align="center">
    DualCast: A Two-Stage Multi-modal Learning Framework with Truncated Diffusion for Precipitation Nowcasting
</h3>
![DualCast Framework Architecture](https://raw.githubusercontent.com/Li-jiaxinn/DualCast/main/dualcast.png)

# 📕 Introduction:
In this work, we propose **DualCast**, a two-stage multi-modal learning framework for precipitation nowcasting. DualCast addresses the challenges of cross-modal dependency, spatial displacement, and low efficiency through two core designs: 
1. Progressive Fusion Model (PFM): A deterministic stage that utilizes HA-Mamba and DC-Mamba to capture consistency and complementarity from Radar and Satellite data.
2. Truncated Diffusion Model (TDM): A probabilistic stage based on Brownian Bridge mechanism. It leverages the fusion outputs as structural prior, significantly improving inference speed and detail fidelity compared to standard diffusion models.

# 📖 Usage:
## 1. Clone Repository
```shell
git clone https://github.com/Li-jiaxinn/DualCast
```
## 2. Requirements
Our code is based on Python 3.10 and CUDA 12.1. The major libraries are listed as follows:

```shell
cd DualCast
conda create -n dualcast python=3.10
conda activate dualcast

# Install PyTorch for CUDA 12.1
pip install torch==2.1.0 torchvision==0.16.0 torchaudio==2.1.0 --index-url https://download.pytorch.org/whl/cu121

# Install Mamba and Causal Conv (Pre-compiled wheels for CUDA 12.1 + Torch 2.1)
# Download matching .whl files from https://github.com/state-spaces/mamba and https://github.com/Dao-AILab/causal-conv1d
pip install path/to/mamba_ssm-2.2.4+cu12torch2.1cxx11abiFALSE-cp310-cp310-linux_x86_64.whl
pip install path/to/causal_conv1d-1.5.0.post8+cu12torch2.1cxx11abiFALSE-cp310-cp310-linux_x86_64.whl

# Install other dependencies
pip install -r requirements.txt
```

## 3. Prepare Datasets
We use the SEVIR-LR dataset for training and evaluation.

## 4. Training and Testing
We provide training and testing scripts in the ./scripts folder.

### Training
```shell
python scripts/train.py --config configs/dualcast_default.yaml
```
### Testing
```shell
python scripts/test.py --checkpoint path/to/your/best_model.pth
```
# 🙌🏻 Acknowledgement:
1. We acknowledge the wonderful work of Mamba and Diffusion Models.
2. The implementation of TDM is inspired by the Brownian Bridge diffusion process.
3. The training pipeline is adapted from standard PyTorch practices.
