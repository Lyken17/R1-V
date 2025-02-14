import os
import json
import tempfile
import torch
from vllm import LLM, SamplingParams
from datasets import load_dataset, load_from_disk
from trl.data_utils import (
    apply_chat_template,
    is_conversational,
    maybe_apply_chat_template,
)
from transformers import (
    AriaForConditionalGeneration,
    AriaProcessor,
    AutoModelForCausalLM,
    AutoModelForSequenceClassification,
    AutoProcessor,
    AutoTokenizer,
    GenerationConfig,
    PreTrainedModel,
    PreTrainedTokenizerBase,
    Qwen2VLForConditionalGeneration,
    Qwen2_5_VLForConditionalGeneration,
    Trainer,
    TrainerCallback,
    is_wandb_available,
)
from qwen_vl_utils import process_vision_info


model_name = "/apdcephfs_gy2/share_302735770/stephenruan/code/src/Qwen2-VL-2B-Instruct"
dataset_name="/apdcephfs_gy2/share_302735770/stephenruan/data/lmms-lab___multimodal-open-r1-8k-verified"

# Load the dataset
dataset = load_dataset(dataset_name)
# Format into conversation
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
# print("has image in dataset")
dataset = dataset.map(make_conversation_image)  # Utilize multiprocessing for faster mapping

# print(dataset)

dataset = dataset["train"]
# for d in dataset:
#     print(d)
#     break

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
)
sampling_params = SamplingParams(
    temperature=1,
    max_tokens=2048,
)
processing_class = AutoProcessor.from_pretrained(model_name)
for d in dataset:
    all_inputs = []
    for i in range(4):
        # print(d)
        # prompt = d["prompt"]
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
                        "min_pixels": 224*224,
                        "max_pixels": 1280*28*28
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
        
        # prompts_text = [
        #     maybe_apply_chat_template(d, processing_class)["prompt"]
        # ]
        # image = d['image']
        # inputs = {"prompt": prompts_text, "multi_modal_data": {"image": image}}

        inputs = {
            "prompt": prompt,
            "multi_modal_data": mm_data,
            # "mm_processor_kwargs": video_kwargs,
        }
        all_inputs.append(inputs)

    outputs = llm.generate(
        all_inputs,
        sampling_params=sampling_params,
        use_tqdm=False,
    )
    # print(outputs[0].outputs[0].text)
    for o in outputs:
        print(o.outputs[0].text)
    # print(outputs)

    break

