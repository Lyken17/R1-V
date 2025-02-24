import os
import json
import tempfile
import torch
from vllm import LLM, SamplingParams
from datasets import load_dataset
from transformers import (
    AutoProcessor,
)
from qwen_vl_utils import process_vision_info
import copy


# model_name = "/apdcephfs_gy2/share_302735770/stephenruan/code/src/Qwen2-VL-2B-Instruct"
model_name = "/apdcephfs_gy2/share_302735770/stephenruan/code/src/Qwen2.5-VL-3B-Instruct"
dataset_name = "/apdcephfs_gy2/share_302735770/stephenruan/data/leonardPKU___geoqa_r1_v_train_8_k"
dataset = load_dataset(dataset_name)
QUESTION_TEMPLATE = "{Question}  Output the thinking process in <think> </think> and final answer (number) in <answer> </answer> tags."
def make_conversation_image(example):
    return {
        "prompt": [
            {
                "role": "user",
                "content": [
                    {"type": "image"},
                    {"type": "text", "text": QUESTION_TEMPLATE.format(Question=example["problem"])},
                ],
            },
        ],
    }
dataset = dataset.map(make_conversation_image)  # Utilize multiprocessing for faster mapping
dataset = dataset["train"]

min_pixels = 4*28*28
max_pixels = 640*28*28
llm = LLM(
    model=model_name,
    device="cuda:0",
    gpu_memory_utilization=0.8,
    dtype=torch.bfloat16,
    # Automatic Prefix Caching caches the KV cache of existing queries, so that a new query can
    # directly reuse the KV cache if it shares the same prefix with one of the existing queries.
    # This is particularly useful here because we generate completions from the same prompts.
    enable_prefix_caching=True,
    enforce_eager=True,
    max_model_len=8192,
    trust_remote_code=True,
    mm_processor_kwargs=(
        {
            "max_pixels": max_pixels,
            "min_pixels": min_pixels,
        }
    )
)
batch_size = 4
num_generations = 8
sampling_params = SamplingParams(
    temperature=1,
    max_tokens=2048,
    n=num_generations
)
processing_class = AutoProcessor.from_pretrained(model_name)
batch_inputs = []
for d in dataset:
    if len(batch_inputs) < batch_size:
        print("len(batch_inputs): ", len(batch_inputs))
        # 临时文件处理
        with tempfile.NamedTemporaryFile(suffix=".png", delete=False) as tmp_file:
            image_path = tmp_file.name
            d['image'].save(image_path)
        # 构建消息结构
        messages = [
            {"role": "system", "content": "You are a helpful assistant."},
            {
                "role": "user",
                "content": [
                    {
                        "type": "image",
                        "image": image_path,
                        "min_pixels": min_pixels,
                        "max_pixels": max_pixels
                    },
                    {"type": "text", "text": d['problem']}
                ]
            }
        ]
        # 生成文本prompt
        prompt = processing_class.apply_chat_template(
            messages,
            tokenize=False,
            add_generation_prompt=True,
        )
        # 处理视觉输入
        image_inputs, video_inputs = process_vision_info(
            messages, 
        )
        # 构建多模态输入
        mm_data = {}
        if image_inputs is not None:
            mm_data["image"] = image_inputs

        inputs = {
            "prompt": prompt,
            "multi_modal_data": mm_data,
            # "mm_processor_kwargs": video_kwargs,
        }
        batch_inputs.append(inputs)
    else:
        outputs = llm.generate(
            batch_inputs,
            sampling_params=sampling_params,
            use_tqdm=False,
        )
        # 提取生成的文本
        for i, output in enumerate(outputs):
            for completion in output.outputs:
                print(completion.text)
                print(f"----- output_{i} -----")
        batch_inputs = []

        break
