from transformers import AutoTokenizer, AutoModelForCausalLM
import torch
import soundfile as sf
import os, sys
import transformers
from tqdm import tqdm
import json
# llasa_1b ='HKUSTAudio/Llasa-1B'
llasa_1b_my = "/home/work_nfs9/wjtian/llasa_opensource/model.safetensors"
device = "cuda"
output_dir = f'./output'
import librosa
os.makedirs(output_dir, exist_ok=True)
import torch.nn.functional as F
import torch.nn as nn
class CustomModel(nn.Module):
    def __init__(self, model_name_or_path, config, cache_dir=None):
        super(CustomModel, self).__init__()
        # 加载预训练的 AutoModelForCausalLM
        self.llm = AutoModelForCausalLM.from_pretrained(
            model_name_or_path,
            torch_dtype='auto',
            cache_dir=cache_dir,
        )
                
        # 打印出self.llm的第一层参数
        for name, param in self.llm.named_parameters():
            print(f"Parameter name: {name}")
            print(f"Parameter shape: {param.shape}")
            print(f"Parameter shape: {param}")
            break
        
        
        # 冻结模型的所有参数
        for param in self.llm.parameters():
            param.requires_grad = False        # 包括 lm head 冻结

        # 获取模型的隐藏层大小
        self.hidden_size = self.llm.config.hidden_size
        self.config = config
        self.ignore_index = -100
        # /root/anaconda3/envs/xcodec2/lib/python3.9/site-packages/transformers/models/llama/modeling_llama.py
        from llama_dir.modeling_llama import LlamaDecoderLayerLlasa, LlamaRMSNorm
        self.mtp_1_block = LlamaDecoderLayerLlasa(config=config, layer_idx=17)
        self.mtp_1_proj = nn.Linear(in_features=2048, out_features=2048)
        self.mtp_1_local_RMSNorm = LlamaRMSNorm(hidden_size = 2048)
        self.mtp_2_block = LlamaDecoderLayerLlasa(config=config, layer_idx=18)
        self.mtp_2_proj = nn.Linear(in_features=2048, out_features=2048)
        self.mtp_2_local_RMSNorm = LlamaRMSNorm(hidden_size = 2048)
    
    def inference(self, input_ids, eos_token_id):
        # 通过原始模型获取输出
        outputs = self.llm.generate(
            input_ids,
            max_length=2048,  # We trained our model with a max length of 2048
            eos_token_id= eos_token_id ,
            do_sample=True,    
            top_p=5,           #  Adjusts the diversity of generated content
            temperature=0.8,   #  Controls randomness in output
        )
        return  outputs
    

    def inference_speedup_backbone_and_mtp1mtp2_correct_checkeos_0313_correct(self, 
                                    input_ids,
                                    eos_token_id, 
                                    top_k = 50,
                                    top_p = None,
                                    temperature = 1,
                                    max_length = 500,
                                    eos_topk = 50,
                                    ):
        current_input = input_ids.clone()  # 当前输入从初始输入开始
        device = input_ids.device
        
        mtp1_next_token_check_todo = None
        mtp2_next_token_check_todo = None
        
        total_num = 0
        check_count_success_mtp1 = 0  # mtp1 校验成功次数
        check_count_success_mtp2 = 0  # mtp2 校验成功次数

        for index in tqdm(range(max_length)):  # 设定最大生成长度
            with torch.no_grad():  # 确保在推理时不会进行梯度计算
                total_num += 1
                t = current_input.shape[1]
                attention_mask = torch.ones_like(current_input).to(device)

                # 第一步：通过 backbone 生成 token1
                lm_outputs = self.llm(input_ids=current_input)  # 获取当前输入的输出
                logits = lm_outputs.logits

                # 校验上一步的 token1 是否在 mtp1 的 top-50 内
                if mtp1_next_token_check_todo:
                    if mtp1_next_token_check_todo == eos_token_id:
                        backbone_topk_tokens = torch.topk(logits[:, -3, :], top_k, dim=-1).indices

                    else:
                        backbone_topk_tokens = torch.topk(logits[:, -3, :], top_k, dim=-1).indices
                    
                    # 校验不通过，去掉上一个 mtp1 token，重新用 backbone 预测上一个 token
                    if mtp1_next_token_check_todo not in backbone_topk_tokens:
                        mtp1_next_token = self.get_next_token_nottime(logits[:, -3, :] / temperature, top_k=top_k, top_p=top_p)
                        # if mtp1_next_token.item() == eos_token_id:
                        #     break
                        current_input = torch.cat([current_input[:, :-2], mtp1_next_token], dim=1)
                        
                        # 重置进行 token 预测，因为 hidden 需要重新得到
                        mtp1_next_token_check_todo = None
                        continue
                    else:
                        check_count_success_mtp1 += 1  # mtp1 校验成功
                        # print(f'check success mtp1_next_token_check_todo: {mtp1_next_token_check_todo}')
                        if mtp1_next_token_check_todo == eos_token_id:
                            current_input = current_input[:, :-2]
                            break
                        
                        # 校验上一步的 token2 是否在 mtp2 的 top-50 内
                        if mtp2_next_token_check_todo:
                            
                            if mtp2_next_token_check_todo == eos_token_id:
                                backbone_topk_tokens = torch.topk(logits[:, -2, :], top_k, dim=-1).indices

                            else:
                                backbone_topk_tokens = torch.topk(logits[:, -2, :], top_k, dim=-1).indices
                        
                        
                            # 校验不通过，去掉上一个 mtp2 token，重新用 backbone 预测上一个 token
                            if mtp2_next_token_check_todo not in backbone_topk_tokens:
                                mtp2_next_token = self.get_next_token_nottime(logits[:, -2, :] / temperature, top_k=top_k, top_p=top_p)
                                # if mtp2_next_token.item() == eos_token_id:
                                #     break
                                current_input = torch.cat([current_input[:, :-1], mtp2_next_token], dim=1)
                                
                                # 重置进行 token 预测，因为 hidden 需要重新得到
                                mtp2_next_token_check_todo = None
                                mtp1_next_token_check_todo = None
                                continue
                            else:
                                check_count_success_mtp2 += 1  # mtp2 校验成功
                                # print(f'check success mtp2_next_token_check_todo: {mtp2_next_token_check_todo}')
                                if mtp2_next_token_check_todo == eos_token_id:
                                    current_input = current_input[:, :-1]
                                    break
                                
                                
                # 如果不校验，则直接拼接 backbone token，hidden 可以继续使用
                backbone_next_token = self.get_next_token(logits, temperature=temperature, top_k=top_k, top_p=top_p)
                if backbone_next_token.item() == eos_token_id:
                    # if index == 0:
                    #     while(backbone_next_token.item() == eos_token_id):
                    #         backbone_next_token = self.get_next_token(logits, temperature=temperature, top_k=top_k, top_p=top_p)
                    # else:
                    break
                current_input = torch.cat([current_input, backbone_next_token], dim=1)

                # 第二步：通过 mtp1 生成 token2
                hid = lm_outputs.hidden_states[-1]
                past_seen_tokens = 0
                cache_position = torch.arange(
                    past_seen_tokens, past_seen_tokens + t, device=device
                )
                position_ids = cache_position.unsqueeze(0)

                inputs_embeds = self.llm.model.embed_tokens(current_input)  
                mtp_1_features = self.mtp_1_proj( self.mtp_1_local_RMSNorm(hid.float()) )
                mtp_1_outputs = self.mtp_1_block(mtp_1_features, attention_mask=attention_mask, position_ids=position_ids, cache_position=cache_position)[0]
                mtp_1_logits = self.llm.lm_head(mtp_1_outputs.to(dtype=torch.bfloat16))
                mtp1_next_token = self.get_next_token(mtp_1_logits, temperature=temperature, top_k=top_k, top_p=top_p)

                # if mtp1_next_token.item() == eos_token_id:
                #     break
                current_input = torch.cat([current_input, mtp1_next_token], dim=1)
                mtp1_next_token_check_todo = mtp1_next_token.item()  # to check token

                # 第三步：通过 mtp2 生成 token3
                inputs_embeds = self.llm.model.embed_tokens(current_input)     
                mtp_2_features = self.mtp_2_proj(self.mtp_2_local_RMSNorm( mtp_1_outputs.float()) )
                mtp_2_outputs = self.mtp_2_block(mtp_2_features, attention_mask=attention_mask, position_ids=position_ids, cache_position=cache_position)[0]
                mtp_2_logits = self.llm.lm_head(mtp_2_outputs.to(dtype=torch.bfloat16))
                mtp2_next_token = self.get_next_token(mtp_2_logits, temperature=temperature, top_k=top_k, top_p=top_p)

                # if mtp2_next_token.item() == eos_token_id:
                #     break
                
                current_input = torch.cat([current_input, mtp2_next_token], dim=1)
                mtp2_next_token_check_todo = mtp2_next_token.item()  # to check token


        return current_input, check_count_success_mtp1, check_count_success_mtp2, total_num

    def get_next_token(self, logits, temperature=None, top_k=None, top_p=None):
        logits = logits[:, -1:, :] / temperature  # [1, k, 53000]
        if top_k is not None:
            v, _ = torch.topk(logits, min(top_k, logits.size(-1)))
            logits[logits < v[:, :, [-1]]] = -float('Inf')
        if top_p is not None:
            sorted_logits, sorted_indices = torch.sort(logits, descending=True)
            cumulative_probs = torch.cumsum(F.softmax(sorted_logits, dim=-1), dim=-1)
            sorted_indices_to_remove = cumulative_probs > top_p
            sorted_indices_to_remove[..., 1:] = sorted_indices_to_remove[..., :-1].clone()
            sorted_indices_to_remove[..., 0] = 0
            for batch in range(sorted_logits.size(0)):
                for beam in range(sorted_logits.size(1)):
                    indices_to_remove = sorted_indices[batch, beam][sorted_indices_to_remove[batch, beam]]
                    logits[batch, beam, indices_to_remove] = -float('Inf')
        probs = F.softmax(logits, dim=-1).squeeze(1)
        idx_next = torch.multinomial(probs, num_samples=1)# b, d -> b, 1

        return idx_next


    def get_next_token_nottime(self, logits, temperature=None, top_k=None, top_p=None):
        if top_k is not None:
            v, _ = torch.topk(logits, min(top_k, logits.size(-1)))
            logits[logits < v[:, [-1]]] = -float('Inf')
        if top_p is not None:
            sorted_logits, sorted_indices = torch.sort(logits, descending=True)
            cumulative_probs = torch.cumsum(F.softmax(sorted_logits, dim=-1), dim=-1)
            sorted_indices_to_remove = cumulative_probs > top_p
            sorted_indices_to_remove[..., 1:] = sorted_indices_to_remove[..., :-1].clone()
            sorted_indices_to_remove[..., 0] = 0
            for batch in range(sorted_logits.size(0)):
                for beam in range(sorted_logits.size(1)):
                    indices_to_remove = sorted_indices[batch, beam][sorted_indices_to_remove[batch, beam]]
                    logits[batch, beam, indices_to_remove] = -float('Inf')
        probs = F.softmax(logits, dim=-1).squeeze(1)
        idx_next = torch.multinomial(probs, num_samples=1)# b, d -> b, 1

        return idx_next
def ids_to_speech_tokens(speech_ids):
 
    speech_tokens_str = []
    for speech_id in speech_ids:
        speech_tokens_str.append(f"<|s_{speech_id}|>")
    return speech_tokens_str

def extract_speech_ids(speech_tokens_str):
 
    speech_ids = []
    for token_str in speech_tokens_str:
        if token_str.startswith('<|s_') and token_str.endswith('|>'):
            num_str = token_str[4:-2]

            num = int(num_str)
            speech_ids.append(num)
        else:
            print(f"Unexpected token: {token_str}")
    return speech_ids

tokenizer = AutoTokenizer.from_pretrained("HKUSTAudio/Llasa-1B")
config = transformers.AutoConfig.from_pretrained("HKUSTAudio/Llasa-1B")
model = CustomModel(model_name_or_path="HKUSTAudio/Llasa-1B",
                    config=config,
                    )


import torch
from safetensors.torch import load_file
state_dict = load_file(llasa_1b_my)
model.load_state_dict(state_dict)
model.eval()  # 设置为评估模式
print(f' loading {llasa_1b_my}')
for name, param in model.llm.named_parameters():
    print(f"Parameter name: {name}")
    print(f"Parameter shape: {param.shape}")
    print(f"Parameter shape: {param}")
    break

model.to('cuda')
sys.path.append("./xcodec2/")
from modeling_xcodec import XCodec2Model
model_path = "ckpts/finetuning.ckpt"  
Codec_model = XCodec2Model(ckpt_path=model_path, device=device)
Codec_model.eval().cuda()   

# input_text = 'Dealing with family secrets is never easy. Yet, sometimes, omission is a form of protection, intending to safeguard some from the harsh truths. One day, I hope you understand the reasons behind my actions. Until then, Anna, please, bear with me.'
# input_text = '突然，身边一阵笑声。我看着他们，意气风发地挺直了胸膛，甩了甩那稍显肉感的双臂，轻笑道："我身上的肉，是为了掩饰我爆棚的魅力，否则，岂不吓坏了你们呢？"'
import sys
# infile=sys.argv[1]

topk = int(50)
    #TTS start!
from tqdm import tqdm
os.makedirs(f"{output_dir}/mtp2-audioprompt-topk{topk}", exist_ok=True)

outputdddd = f"{output_dir}/mtp2-audioprompt-topk{topk}"
            
with torch.no_grad():
    speech_end_id = tokenizer.convert_tokens_to_ids('<|SPEECH_GENERATION_END|>')
            
    # 按顺序分配变量
    filename = "test"
    prompt_text = "收到好友从远方寄来的生日礼物，那份意外的惊喜与深深的祝福让我心中充满了甜蜜的快乐，笑容如花儿般绽放。"
    prompt_audio = "./asset/zero_shot_prompt.wav"
    text_to_synthesize = "哈哈哈，你好，我是来自火星的生命，希望你以后能够做的比我还好呦。"
    ground_truth_audio = "./gt.wav"
    
    if ground_truth_audio:
        print(f"Ground Truth Audio: {ground_truth_audio}")

    idfilename = os.path.splitext(os.path.splitext(os.path.basename(filename))[0])[0]

    target_text = text_to_synthesize
    
    prompt_wav, sr = librosa.load(prompt_audio, sr=16000)
    # prompt_wav, sr = sf.read(item.replace('.normalized.txt', '.wav').strip()) # 16k only
    prompt_wav = torch.from_numpy(prompt_wav).float().unsqueeze(0)  
    prompt_text = prompt_text
    vq_code_prompt = Codec_model.encode_code(input_waveform=prompt_wav)
    vq_code_prompt = vq_code_prompt[0,0,:]
    speech_ids_prefix = ids_to_speech_tokens(vq_code_prompt)
    
    
    input_text = prompt_text + ' ' + target_text
    formatted_text = f"<|TEXT_UNDERSTANDING_START|>{input_text}<|TEXT_UNDERSTANDING_END|>"
    # Tokenize the text and the speech prefix
    chat = [
        {"role": "user", "content": "Convert the text to speech:" + formatted_text},
        {"role": "assistant", "content": "<|SPEECH_GENERATION_START|>" + ''.join(speech_ids_prefix)}
    ]

    input_ids = tokenizer.apply_chat_template(
        chat, 
        tokenize=True, 
        return_tensors='pt', 
        continue_final_message=True
    )
    input_ids = input_ids.to('cuda')
    outputs, check_count_success_mtp1, check_count_success_mtp2, total_num = model.inference_speedup_backbone_and_mtp1mtp2_correct_checkeos_0313_correct(
        input_ids,
        eos_token_id= speech_end_id ,
        top_k=topk
    )
    generated_ids = outputs[0][input_ids.shape[1]-len(speech_ids_prefix):]
    speech_tokens = tokenizer.batch_decode(generated_ids, skip_special_tokens=True)   
    speech_tokens = extract_speech_ids(speech_tokens)
    speech_tokens = torch.tensor(speech_tokens).cuda().unsqueeze(0).unsqueeze(0)
    try:
        gen_wav = Codec_model.decode_code(speech_tokens) 
    except:
        print(f"{idfilename} decode error")
    sf.write(f"{outputdddd}/{idfilename}.wav", gen_wav[0, 0, 320 * len(speech_ids_prefix):].cpu().numpy(), 16000)
                
