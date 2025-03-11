# Install the packages in r1-v .
cd src/r1-v 
pip install uv pip --upgrade

uv pip install -e ".[dev]"

# Addtional modules
uv pip install wandb==0.18.3
uv pip install tensorboardx
uv pip install qwen_vl_utils torchvision
uv pip install flash-attn --no-build-isolation

# vLLM support 
uv pip install vllm==0.7.2

# fix transformers version
uv pip install git+https://github.com/huggingface/transformers.git@336dc69d63d56f232a183a3e7f52790429b871ef
