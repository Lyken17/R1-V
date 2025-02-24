export http_proxy="http://star-proxy.oa.com:3128"
export https_proxy="http://star-proxy.oa.com:3128"

conda create -n r1v --clone base -y
source $(conda info --base)/etc/profile.d/conda.sh
conda activate r1v

# Install the packages in r1-v .
cd src/r1-v 
pip install -e ".[dev]"

# Addtional modules
# fix transformers version
pip install git+https://github.com/huggingface/transformers.git@336dc69d63d56f232a183a3e7f52790429b871ef
pip install wandb==0.18.3
pip install tensorboardx    # not neccessary
pip install qwen_vl_utils
pip insall ujson
# pip install flash-attn --no-build-isolation
# pip install git+https://github.com/huggingface/transformers.git # correct deepspeed support

# vLLM support
pip install vllm==0.7.2

