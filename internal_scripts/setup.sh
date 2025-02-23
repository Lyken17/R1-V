export http_proxy="http://star-proxy.oa.com:3128"
export https_proxy="http://star-proxy.oa.com:3128"

conda create -n r1v --clone base -y
source $(conda info --base)/etc/profile.d/conda.sh
conda activate r1v

# Install the packages in open-r1-multimodal .
cd src/open-r1-multimodal # We edit the grpo.py and grpo_trainer.py in open-r1 repo.
pip install -e ".[dev]"

# Addtional modules
pip install wandb==0.18.3
pip install tensorboardx    # not neccessary
pip install qwen_vl_utils
pip insall ujson
# pip install flash-attn --no-build-isolation
# pip install git+https://github.com/huggingface/transformers.git # correct deepspeed support

# vLLM support
pip install vllm==0.7.2

