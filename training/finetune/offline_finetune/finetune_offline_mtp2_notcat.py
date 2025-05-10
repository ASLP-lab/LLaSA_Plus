from logging import config
import os
import json
import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader
from transformers import (
    AutoTokenizer,
    AutoConfig,
    AutoModelForCausalLM,
    Trainer,
    TrainingArguments,
    HfArgumentParser,
    default_data_collator,
)
from dataclasses import dataclass, field
from typing import Optional
import sys
import transformers
import wandb
from transformers.trainer_pt_utils import LabelSmoother
import numpy as np
import random
from datasets import load_dataset
from functools import partial
import torch.nn.functional as F
import torch
import torch.nn as nn
import math
from transformers import Trainer

# class CustomTrainer(Trainer):
#     def compute_loss(self, model, inputs, return_outputs=False):
#         """
#         How the loss is computed by Trainer. By default, all models return the loss in the first element.

#         Subclass and override for custom behavior.
#         """

#         mtp_1_loss, mtp_2_loss = model(**inputs)
#         loss = mtp_1_loss + mtp_2_loss
#         self.log({"loss1": mtp_1_loss.item(), "loss2": mtp_2_loss.item(), "loss_total": loss.item()})
#         return loss
    
    
def freeze_model(model):
    for name, param in model.named_parameters():
        param.requires_grad = False
@dataclass
class ModelArguments:
    llm_model_name_or_path: Optional[str] = field(default="meta-llama/Llama-3.2-1B-Instruct")
    cache_dir: Optional[str] = field(default=None, metadata={"help": "Cache directory for the model."})

 
@dataclass
class DataArguments:
    data_path: str = field(default=None, metadata={"help": "Root path to the memmap data."})

 
@dataclass
class CustomTrainingArguments(TrainingArguments):
    optim: str = field(default="adamw_torch_fused")
    model_max_length: int = field(
        default=2048,
        metadata={"help": "Maximum sequence length"},
    )
    logging_steps: int = field(default=100, metadata={"help": "Log every X updates"})
    report_to: Optional[str] = field(
        default=None, metadata={"help": "The integration to report the results and logs to."}
    )
    run_name: Optional[str] = field(
        default='/zhuxinfa/work_space/llasa/output', metadata={"help": "The name of the run for logging."}
    )
    gradient_checkpointing: bool = field(default=True)
    lr_scheduler_type: str = field(default="cosine", metadata={"help": "The learning rate scheduler to use."})
  
class TTSDataset(Dataset):
    def __init__(self, data_path, split, tokenizer):
 
        memmap_path = os.path.join(data_path, f'{split}_input_ids.memmap')
        shape_path = os.path.join(data_path, f'{split}_input_ids_shape.npy')

        self.input_ids = np.memmap(memmap_path, dtype='int32', mode='r', shape=tuple(np.load(shape_path)))
        self.length = self.input_ids.shape[0]
        self.pad_token_id = tokenizer.pad_token_id   
        self.tokenizer = tokenizer

   
        self.speech_generation_start_id = tokenizer.convert_tokens_to_ids('<|SPEECH_GENERATION_START|>')
        self.speech_generation_end_id = tokenizer.convert_tokens_to_ids('<|SPEECH_GENERATION_END|>')
        self.text_generation_start_id = tokenizer.convert_tokens_to_ids('<|TEXT_GENERATION_START|>')
        self.text_generation_end_id = tokenizer.convert_tokens_to_ids('<|TEXT_GENERATION_END|>')
        self.text_understanding_start_id = tokenizer.convert_tokens_to_ids('<|TEXT_UNDERSTANDING_START|>')
        self.text_understanding_end_id = tokenizer.convert_tokens_to_ids('<|TEXT_UNDERSTANDING_END|>')
        self.speech_understanding_start_id = tokenizer.convert_tokens_to_ids('<|SPEECH_UNDERSTANDING_START|>')
        self.speech_understanding_end_id = tokenizer.convert_tokens_to_ids('<|SPEECH_UNDERSTANDING_END|>')
 
        self.max_length = 2048
        self.ignore_index = -100  

    def __len__(self):
        return self.length

    def replace_tagged_token(self, token_list, target_token, new_sequence):
        idx = token_list.index(target_token)
        return token_list[:idx] + list(new_sequence) + token_list[idx+1:]

    def pad_sequence(self, sequence, max_length, value=0):
        if len(sequence) >= max_length:
            return sequence[:max_length]
        else:
            padding = torch.full((max_length - len(sequence),), value, dtype=sequence.dtype)
            return torch.cat([sequence, padding], dim=0)

    def __getitem__(self, idx):
        input_ids = torch.tensor(self.input_ids[idx], dtype=torch.long)

 
        labels = torch.full_like(input_ids, self.ignore_index)

 
        speech_gen_positions = (input_ids == self.speech_generation_start_id).nonzero(as_tuple=True)[0]
        text_gen_positions = (input_ids == self.text_generation_start_id).nonzero(as_tuple=True)[0]
 
        speech_gen_idx = speech_gen_positions[0].item()
        try:
            speech_gen_end_idx = (input_ids == self.speech_generation_end_id).nonzero(as_tuple=True)[0].item()
        except Exception as e:
            print(f"maybe Error in speech_gen_end_idx: {e}")
            speech_gen_end_idx = 2048
 
        text_sequence = input_ids[:speech_gen_idx]
        speech_sequence = input_ids[speech_gen_idx : speech_gen_end_idx + 1]
 
        chat = [
            {"role": "user", "content": "Convert the text to speech:<|TEXT_UNDERSTANDING_START|>"},
            {"role": "assistant", "content": "<|SPEECH_GENERATION_START|>"}
        ]
        ids = self.tokenizer.apply_chat_template(chat, tokenize=True)

    
        ids = self.replace_tagged_token(ids, self.text_understanding_start_id, text_sequence)
        ids = self.replace_tagged_token(ids, self.speech_generation_start_id, speech_sequence)



        input_ids = torch.tensor(ids, dtype=torch.long)
        labels = torch.full_like(input_ids, self.ignore_index)

        try:
 
            speech_gen_idx_in_input = (input_ids == self.speech_generation_start_id).nonzero(as_tuple=True)[0].item()
            labels[speech_gen_idx_in_input:] = input_ids[speech_gen_idx_in_input:]
        except Exception as e:
            print(f"maybe Error in speech_gen_idx_in_input: {e}")
            # speech_gen_idx_in_input = len(input_ids) - 1
 
            labels  = input_ids 

 
        attention_mask = (input_ids != self.pad_token_id).long()
 
        labels[input_ids == self.pad_token_id] = self.ignore_index

 
        
        input_ids = self.pad_sequence(input_ids, self.max_length, value=self.pad_token_id)
        attention_mask = self.pad_sequence(attention_mask, self.max_length, value=0)
        labels = self.pad_sequence(labels, self.max_length, value=self.ignore_index)

        return {
            'input_ids': list(input_ids),
            'labels': list(labels),
            'attention_mask': list(attention_mask)
        }


    
class CustomModel(nn.Module):
    def __init__(self, model_name_or_path, config, cache_dir=None):
        super(CustomModel, self).__init__()
        # 加载预训练的 AutoModelForCausalLM
        self.llm = AutoModelForCausalLM.from_pretrained(
            model_name_or_path,
            torch_dtype='auto',
            cache_dir=cache_dir,
        )
        # 冻结模型的所有参数
        for param in self.llm.parameters():
            param.requires_grad = False        # 包括 lm head 冻结

        # 获取模型的隐藏层大小
        self.hidden_size = self.llm.config.hidden_size
        self.config = config
        self.ignore_index = -100
        # /root/anaconda3/envs/xcodec2/lib/python3.9/site-packages/transformers/models/llama/modeling_llama.py
        from transformers.models.llama.modeling_llama import LlamaDecoderLayerLlasa, LlamaRMSNorm
        self.mtp_1_block = LlamaDecoderLayerLlasa(config=config, layer_idx=17)
        self.mtp_1_proj = nn.Linear(in_features=2048, out_features=2048)
        self.mtp_1_local_RMSNorm = LlamaRMSNorm(hidden_size = 2048)
        self.mtp_2_block = LlamaDecoderLayerLlasa(config=config, layer_idx=18)
        self.mtp_2_proj = nn.Linear(in_features=2048, out_features=2048)
        self.mtp_2_local_RMSNorm = LlamaRMSNorm(hidden_size = 2048)

    def forward(self, input_ids, attention_mask, labels):
        # 通过原始模型获取输出
        device = input_ids.device  
        t = input_ids.shape[1]

        past_seen_tokens = 0
        cache_position = torch.arange(
            past_seen_tokens, past_seen_tokens + t, device=device
        )
        position_ids = cache_position.unsqueeze(0)
            
        # lm_outputs = self.llm(input_ids=input_ids, attention_mask=attention_mask, labels=labels)
        lm_outputs = self.llm(input_ids=input_ids, attention_mask=attention_mask)
        # print('lm_outputs', lm_outputs.loss)
        
        # #lm head全部冻结，然后需要先过lm head再计算loss, 如果不冻结，则原始的backbone的效果会下降
        
        # 最后一个时刻是t+1，应该 mask 或者去掉 
        mtp_1_features = self.mtp_1_proj(self.mtp_1_local_RMSNorm(lm_outputs.hidden_states[-1][:, : -1, :])) # 2048 -》 2047 输入是1~t  target 2~t+1
        mtp_1_outputs = self.mtp_1_block(mtp_1_features, attention_mask=attention_mask[:, 1:], position_ids=position_ids[:, :-1], cache_position=cache_position[:-1])[0]
        mtp_1_logits = self.llm.lm_head(mtp_1_outputs)
        
        # position id 都是从0计算的！！！
        mtp_2_features = self.mtp_2_proj(self.mtp_2_local_RMSNorm(mtp_1_outputs[:, : -1, :])) # 2047 -》 2046   输入是2~t+2  输入是2~t  target 3~t+1
        mtp_2_outputs = self.mtp_2_block(mtp_2_features, attention_mask=attention_mask[:, 2:], position_ids=position_ids[:, :-2], cache_position=cache_position[:-2])[0]
        mtp_2_logits = self.llm.lm_head(mtp_2_outputs)

        
        mtp_1_loss = F.cross_entropy(mtp_1_logits[:, : -1, :].transpose(1,2), labels[:, 2:], ignore_index=self.ignore_index)        
        mtp_2_loss = F.cross_entropy(mtp_2_logits[:, : -1, :].transpose(1,2), labels[:, 3:], ignore_index=self.ignore_index)
        
        return  [ mtp_1_loss+mtp_2_loss ]



def main():
    # Parse arguments
    parser = transformers.HfArgumentParser(
        (ModelArguments, DataArguments, CustomTrainingArguments))
    if len(sys.argv) > 1 and sys.argv[1].endswith(".json"):
        # Load arguments from the specified JSON file
        (
            model_args,
            data_args,
            training_args,
        ) = parser.parse_json_file(json_file=os.path.abspath(sys.argv[1]))
    else:
        # Attempt to load arguments from the default 'config.json' file
        default_config_file = 'config.json'
        (
            model_args,
            data_args,
            training_args,
        ) = parser.parse_json_file(json_file=os.path.abspath(default_config_file))
     
    is_main_process = training_args.local_rank in [-1, 0]
 
    if training_args.report_to == "wandb" and is_main_process:
        wandb.init(
            project="llm_tts",  
            config=training_args.to_sanitized_dict(),
            name=training_args.run_name
        )

 
    tokenizer = AutoTokenizer.from_pretrained(
        model_args.llm_model_name_or_path,
        model_max_length=training_args.model_max_length,
        padding_side="right",
    )
    tokenizer.pad_token = tokenizer.eos_token
    config = transformers.AutoConfig.from_pretrained(model_args.llm_model_name_or_path)
    model = CustomModel(model_name_or_path=model_args.llm_model_name_or_path,
                        config=config,
                        cache_dir=model_args.cache_dir)

   
 
    train_dataset = TTSDataset(
        data_path=data_args.data_path,
        split='train',
        tokenizer=tokenizer
    )
    train_dataset[0]
    eval_dataset = TTSDataset(
        data_path=data_args.data_path,
        split='val',
        tokenizer=tokenizer
    ) if os.path.exists(os.path.join(data_args.data_path, 'val_input_ids.memmap')) else None
 
    data_collator = default_data_collator
    trainer = Trainer(
        model=model,
        tokenizer=tokenizer,
        args=training_args,
        train_dataset=train_dataset,
        eval_dataset=eval_dataset,
        data_collator=data_collator,
    )
    if is_main_process:
        trainer.add_callback(transformers.integrations.WandbCallback())
 
    trainer.train()
 
    trainer.save_model(training_args.output_dir)
    tokenizer.save_pretrained(training_args.output_dir)

if __name__ == "__main__":
    main()
