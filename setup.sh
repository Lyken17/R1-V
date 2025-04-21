# /bin/bash
set -e 
# Install the packages in r1-v .
cd src/r1-v 
pip install uv 

pip install lighteval@git+https://github.com/huggingface/lighteval.git@4f381b352c0e467b5870a97d41cb66b487a2c503
uv pip install -e ".[dev]"

# Addtional modules
uv pip install wandb==0.18.3
uv pip install tensorboardx
uv pip install qwen_vl_utils torchvision
# uv pip install flash-attn --no-build-isolation
# pin version for python=3.10 + torch=2.5
pip install https://github.com/Dao-AILab/flash-attention/releases/download/v2.7.4.post1/flash_attn-2.7.4.post1+cu12torch2.5cxx11abiFALSE-cp310-cp310-linux_x86_64.whl

# vLLM support 
uv pip install vllm==0.7.2

# fix transformers version
uv pip install git+https://github.com/huggingface/transformers.git@336dc69d63d56f232a183a3e7f52790429b871ef

# for nvila wrapper
pip install git+https://github.com/bfshi/scaling_on_scales.git


