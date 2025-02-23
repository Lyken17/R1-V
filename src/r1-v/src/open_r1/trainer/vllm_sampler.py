import os
import copy

import torch
import torch.nn as nn
import torch.utils.data
import torch.distributed as dist

import transformers
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
from transformers.integrations.deepspeed import is_deepspeed_zero3_enabled
from trl.import_utils import is_vllm_available
from trl.data_utils import (
    apply_chat_template,
    is_conversational,
    maybe_apply_chat_template,
)
from trl.trainer.grpo_config import GRPOConfig

if is_vllm_available():
    from vllm import LLM, SamplingParams

if is_wandb_available():
    import wandb


class VLLMSampler:
    def __init__(self, 
        model,
        args: GRPOConfig = None,
        processing_class: Optional[PreTrainedTokenizerBase] = None,
        max_pixels: Optional[int] = 12845056,
        min_pixels: Optional[int] = 3136,
        attn_implementation: str = "flash_attention_2",
    ):
        # model, processor and args
        if isinstance(model, str):
            if args is None:
                model_name = model if isinstance(model, str) else model.config._name_or_path
                model_name = model_name.split("/")[-1]
                args = GRPOConfig(f"{model_name}-GRPO")
        else:

        # Processing class
        model_id = model
        if processing_class is None:
            if "Qwen" in model_id or "Aria" in model_id:
                processing_class = AutoProcessor.from_pretrained(model_id)
                pad_token_id = processing_class.tokenizer.pad_token_id
                processing_class.pad_token_id = pad_token_id
                processing_class.eos_token_id = processing_class.tokenizer.eos_token_id
                if "Qwen" in model_id:
                    processing_class.image_processor.max_pixels = max_pixels
                    processing_class.image_processor.min_pixels = min_pixels
            else:
                processing_class = AutoTokenizer.from_pretrained(
                    model.config._name_or_path, padding_side="left"
                )
                pad_token_id = processing_class.pad_token_id
        else:
            model_id = model.config._name_or_path
            if args.model_init_kwargs is not None:
                raise ValueError(
                    "You passed `model_init_kwargs` to the `GRPOConfig`, but your model is already instantiated. "
                    "This argument can only be used when the `model` argument is a string."
                )

        # sampling args
        self.max_prompt_length = args.max_prompt_length
        self.max_completion_length = (
            args.max_completion_length
        )  # = |o_i| in the GRPO paper
        self.num_generations = args.num_generations  # = G in the GRPO paper

        

        self.llm = LLM(model_path)

        dist.init_process_group(backend="nccl")
    
    def _receive_state_dict_from_trainers(self):
        # 从训练机器接收权重
        state_dict = {}
        for rank in range(self.accelerator.num_processes):
            if rank != self.accelerator.process_index:
                state_dict.update(dist.recv(src=rank))
        return state_dict

    def _receive_prompts_from_trainers(self):
        # 从训练机器接收prompts
        all_prompts_text = []
        for rank in range(self.accelerator.num_processes):
            if rank != self.accelerator.process_index:
                prompts = dist.recv(src=rank))
                all_prompts_text.extend(prompts)
        return all_prompts_text

    def _receive_images_from_trainers(self):
        # 从训练机器接收images
        all_images = []
        for rank in range(self.accelerator.num_processes):
            if rank != self.accelerator.process_index:
                images = dist.recv(src=rank))
                all_images.extend(images)
        return all_images

    def _send_completions_to_trainers(self, completions):
        # 将completions发送回训练机器
        for rank in range(self.accelerator.num_processes):
            if rank != self.accelerator.process_index:
                dist.send(completions, dst=rank)
            
    def _infer(self, inputs):
        # Generate completions using vLLM: gather all prompts and use them in a single call in the main process
        images = [x["image"] for x in inputs]
        prompts_text = [
            maybe_apply_chat_template(example, self.processing_class)["prompt"]
            for example in inputs
        ]
        all_prompts_text = gather_object(prompts_text)
        all_images = gather_object(images)
        # group into pairs
        all_multimodal_inputs = []
        for prompt, image in zip(all_prompts_text, all_images):
            for _ in range(self.num_generations):
                all_multimodal_inputs.append({"prompt": prompt, "multi_modal_data": {"image": image}})

        # # NOTE: The sampling should be divided into `num_generations` batches, 
        # # otherwise the sampling of each prompt will be the same
        # all_completion_ids = [None] * len(all_multimodal_inputs)
        # for i in range(self.num_generations):
        #     # Get the inputs for the current batch
        #     batch_inputs = [all_multimodal_inputs[j] for j in range(i, len(all_multimodal_inputs), self.num_generations)]
        #     if self.accelerator.is_main_process:
        #         outputs = self.llm.generate(
        #             batch_inputs,
        #             sampling_params=self.sampling_params,
        #             use_tqdm=False,
        #         )
        #         batch_completion_ids = [out.token_ids for completions in outputs for out in completions.outputs]
        #     else:
        #         batch_completion_ids = [None] * len(batch_inputs)
        #     # Place the results back into their original positions
        #     for idx, completion_id in enumerate(batch_completion_ids):
        #         all_completion_ids[i + idx * self.num_generations] = completion_id
        # completion_ids = all_completion_ids

        # Initialize the list to store completion IDs
        all_completion_ids = [None] * len(all_multimodal_inputs)
        for i in range(self.num_generations):
            # Get the inputs for the current batch
            batch_inputs = [all_multimodal_inputs[j] for j in range(i, len(all_multimodal_inputs), self.num_generations)]
            if self.accelerator.is_main_process:
                # Generate output for the current input
                outputs = self.llm.generate(
                    input_data,  # Pass the current input as a single-element list
                    sampling_params=self.sampling_params,
                    use_tqdm=False,
                )
                # Extract the completion IDs from the output
                completion_ids = [out.token_ids for completions in outputs for out in completions.outputs]
                all_completion_ids[i] = completion_ids[0]  # Store the result
            else:
                all_completion_ids[i] = None  # Non-main processes store None

        # Final completion IDs
        completion_ids = all_completion_ids
        
        completion_ids = broadcast_object_list(completion_ids, from_process=0)
        process_slice = slice(
            self.accelerator.process_index * len(prompts) * self.num_generations,
            (self.accelerator.process_index + 1) * len(prompts) * self.num_generations,
        )
        completion_ids = completion_ids[process_slice]
        # Pad the completions, and concatenate them with the prompts
        completion_ids = [torch.tensor(ids, device=device) for ids in completion_ids]
        completion_ids = pad(
            completion_ids, padding_value=self.processing_class.pad_token_id
        )
        prompt_ids = prompt_ids.repeat_interleave(self.num_generations, dim=0)
        prompt_completion_ids = torch.cat([prompt_ids, completion_ids], dim=1)

        prompt_length = prompt_ids.size(1)
        prompt_ids = prompt_completion_ids[:, :prompt_length]
        completion_ids = prompt_completion_ids[:, prompt_length:]
        prompt_mask = prompt_mask.repeat_interleave(self.num_generations, dim=0)
    
    def _update_model_weights(self, state_dict):
        llm_model = (
            self.llm.llm_engine.model_executor.driver_worker.model_runner.model
        )
        llm_model.load_weights(state_dict.items())
        # self._last_loaded_step = self.state.global_step

    def run(self):
        while True:
            # 1. Receive weights from training node
            state_dict = {}
            for key in self.llm.state_dict().keys():
                tensor = torch.empty_like(self.llm.state_dict()[key])
                dist.recv(tensor, src=0)
                state_dict[key] = tensor
            self.llm.load_state_dict(state_dict)

            # 2. Receive prompts/images from training node
            inputs = torch.empty(...)
            dist.recv(inputs, src=0)

            # 3. Generate completions
            outputs = self.llm.generate(inputs)

            # 4. Send completions back
            dist.send(outputs, dst=0)