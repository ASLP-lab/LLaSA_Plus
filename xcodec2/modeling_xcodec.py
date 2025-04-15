import torch
import torch.nn as nn
from transformers import AutoFeatureExtractor, Wav2Vec2BertModel
from vq.codec_encoder import CodecEncoder
from vq.codec_decoder_vocos import CodecDecoderVocos
from vq.module import SemanticEncoder
import torch.nn.functional as F

class XCodec2Model(nn.Module):
    def __init__(self, ckpt_path, device='cuda'):
        super().__init__()
        self.device = device
        self.load_ckpt(ckpt_path)

    def load_ckpt(self, ckpt_path):
        print(f'Loading checkpoint from {ckpt_path}')
        ckpt = torch.load(ckpt_path, map_location='cpu')
        state_dict = ckpt['state_dict'] if 'state_dict' in ckpt else ckpt

        # 过滤权重
        filtered_state_dict_codec = {}
        filtered_state_dict_semantic_encoder = {}
        filtered_state_dict_gen = {}
        filtered_state_dict_fc_post_a = {}
        filtered_state_dict_fc_prior = {}

        for key, value in state_dict.items():
            if key.startswith('CodecEnc.'):
                filtered_state_dict_codec[key[len('CodecEnc.'):]] = value
            elif key.startswith('generator.'):
                filtered_state_dict_gen[key[len('generator.'):]] = value
            elif key.startswith('fc_post_a.'):
                filtered_state_dict_fc_post_a[key[len('fc_post_a.'):]] = value
            elif key.startswith('fc_prior.'):
                filtered_state_dict_fc_prior[key[len('fc_prior.'):]] = value
            elif key.startswith('SemanticEncoder_module.'):
                filtered_state_dict_semantic_encoder[key[len('SemanticEncoder_module.'):]] = value

        # 模块实例化
        self.semantic_model = Wav2Vec2BertModel.from_pretrained(
            "/home/work_nfs14/code/hkxie/TTS/MaskGCT/ckpt/w2v-bert-2.0",
            output_hidden_states=True
        ).eval().to(self.device)

        self.SemanticEncoder_module = SemanticEncoder(1024, 1024, 1024).eval().to(self.device)
        self.SemanticEncoder_module.load_state_dict(filtered_state_dict_semantic_encoder)

        self.CodecEnc = CodecEncoder().eval().to(self.device)
        self.CodecEnc.load_state_dict(filtered_state_dict_codec)

        self.decoder = CodecDecoderVocos().eval().to(self.device)
        self.decoder.load_state_dict(filtered_state_dict_gen)

        self.fc_post_a = nn.Linear(2048, 1024).eval().to(self.device)
        self.fc_post_a.load_state_dict(filtered_state_dict_fc_post_a)

        self.fc_prior = nn.Linear(2048, 2048).eval().to(self.device)
        self.fc_prior.load_state_dict(filtered_state_dict_fc_prior)

        self.feature_extractor = AutoFeatureExtractor.from_pretrained(
            "/home/work_nfs14/code/hkxie/TTS/MaskGCT/ckpt/w2v-bert-2.0"
        )

    def encode_code(self, input_waveform, sample_rate=16000):
        """
        将输入的音频编码为代码表示。

        参数：
          input_waveform: [batch_size, waveform_length]
          sample_rate: 默认 16000
        返回：
          编码后的代码 (Tensor)
        """
        with torch.no_grad():
    
            wav = input_waveform 
            pad_for_wav = (320 - (wav.shape[1] % 320))
    
            wav = torch.nn.functional.pad(wav, (0, pad_for_wav))

            input_features = self.feature_extractor(F.pad(wav[0,:].cpu(), (160, 160)), sampling_rate=16000, return_tensors="pt") .data['input_features'].to(self.device)  # [batch, frames, feat_dim]

            # 2) 语义层
            semantic_output = self.semantic_model(input_features)
            semantic_hidden_16 = semantic_output.hidden_states[16]  # 取第16层
            semantic_hidden_16 = semantic_hidden_16.transpose(1, 2)  # [batch, hidden_dim, frames]
            semantic_encoded = self.SemanticEncoder_module(semantic_hidden_16)

            # 3) codec encoder
            wav = wav.to(self.device)  # shape: [batch, 1, time]
            vq_emb = self.CodecEnc(wav.unsqueeze(1))  # [batch, time//down, 1024] 只是示例
            vq_emb = vq_emb.transpose(1, 2)  # -> [batch, 1024, frames]

            # 4) 拼接
            concat_emb = torch.cat([semantic_encoded, vq_emb], dim=1)  # [batch, 2048, frames]

            # 5) fc_prior
            concat_emb = self.fc_prior(concat_emb.transpose(1, 2)).transpose(1, 2)

            # 6) decoder 的量化部分，获取code
            _, vq_code, _ = self.decoder(concat_emb, vq=True)
            # vq_code: [batch, frames]
            return vq_code

    def decode_code(self, vq_code):
        """
        将编码后的代码解码回音频。

        参数：
          vq_code: 编码后的代码 (Tensor) [batch, frames]
        返回：
          解码后的音频 (Tensor) [batch, waveform_length]
        """
        with torch.no_grad():
            # 获取量化后的嵌入
            vq_post_emb = self.decoder.quantizer.get_output_from_indices(vq_code.transpose(1, 2))
            vq_post_emb = vq_post_emb.transpose(1, 2)  # [batch, 1024, frames]

            # 7) fc_post_a
            vq_post_emb = self.fc_post_a(vq_post_emb.transpose(1, 2)).transpose(1, 2)  # [batch, 1024, frames]

            # 8) 最后解码成波形
            recon_audio = self.decoder(vq_post_emb.transpose(1, 2), vq=False)[0]  # [batch, time]
            return recon_audio