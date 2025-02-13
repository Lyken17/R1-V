# 内部分支使用说明
- 基础环境镜像：mirrors.tencent.com/timurhe-ns/hunyuan-cuda12.2-torch2.5.1-flashattn2.7.4.post1-vllm0.7.1-python3.12.2-gcc12:latest
- [使用文档](https://iwiki.woa.com/p/4013620298)
- [实验记录（待更新）](https://iwiki.woa.com/p/4013620373)

## 新特性
### 2025.02.13
1. 为方便管理，内部环境的启动脚本已全部移到 ``src/scripts/internal_scripts/` 下。

2. vllm_grpo_trainer启动脚本样例：`src/scripts/internal_scripts/run_vllm_grpo_clevr_torchrun_gy.sh`
    - 目前不支持多机vllm
    - 新的Trainer为：`Qwen2VLGRPOVLLMTrainerModified`（在 `src/open-r1-multimodal/src/open_r1/trainer/vllm_grpo_trainer_modified.py`）中，不再使用 `RepeatRandomSampler` 以避免steps倍增问题，在单个原始batch内完成各prompt的多次采样和计算loss，即保持与 `Qwen2VLGRPOTrainer` 一致的逻辑；同时，不再要求num_generations必须保持与num_gpus一致
    - 新的Trainer已替代原 `Qwen2VLGRPOVLLMTrainer` 在 `src/open-r1-multimodal/src/open_r1/grpo.py` 加载
    - vllm采样逻辑已修正