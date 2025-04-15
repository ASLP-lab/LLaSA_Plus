import torch
import soundfile as sf
from transformers import AutoConfig
from argparse import ArgumentParser
from modeling_xcodec import XCodec2Model

if __name__ == '__main__':
    parser = ArgumentParser()
    parser.add_argument('--input-dir', type=str, default='test_audio/input_test')
    parser.add_argument('--ckpt', type=str, default='ckpt/fintune.ckpt')
    parser.add_argument('--output-dir', type=str, default='test_audio/output_test')
    parser.add_argument('--device', type=str, default='1')
    args = parser.parse_args()

    device = torch.device(f'cuda:{args.device}')
    
    model = XCodec2Model(ckpt_path=args.ckpt, device=device)
    
    model.eval()

    wav, sr = sf.read("./sample/ZH_B00029_S02141_W000305.mp3")   
    wav_tensor = torch.from_numpy(wav).float().unsqueeze(0)  # Shape: (1, T)
    
    with torch.no_grad():
        # Only 16khz speech
        # Only supports single input. For batch inference, please refer to the link below.
        vq_code = model.encode_code(input_waveform=wav_tensor)
        print("Code:", vq_code)  

        recon_wav = model.decode_code(vq_code).cpu()       # Shape: (1, 1, T')
    
    sf.write("reconstructed.wav", recon_wav[0, 0, :].numpy(), sr)
    print("Done! Check reconstructed.wav")