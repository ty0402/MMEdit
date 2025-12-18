import torch
import torch.nn as nn
import librosa
import numpy as np
from transformers import AutoProcessor, Qwen2AudioForConditionalGeneration
import os
# 暂未使用，原始应该是生成的pre
QWEN_AUDIO_PREFIX = '''Given a user prompt and an audio clip, generate an "Enhanced prompt" that provides detailed descriptions suitable for audio generation. Evaluate the audio and user prompt:
- If the prompt is simple, focus on adding specifics about tones, instruments, rhythms, tempos, and audio characteristics to create vivid and concrete audio descriptions.
- If the prompt is already detailed, refine and enhance the existing details slightly without overcomplicating.\n
Here are examples of how to transform or refine prompts:
- User Prompt: Piano music -> Enhanced: A gentle, melancholic piano piece with delicate arpeggios in a minor key, featuring subtle reverb that creates a sense of space and intimacy.
- User Prompt: City sounds -> Enhanced: A bustling urban soundscape with distant traffic noise, occasional car horns, footsteps on concrete sidewalks, and the murmur of crowd conversations, with subtle pigeons cooing in the background.\n
Please generate only the enhanced description for the audio and prompt below and avoid including any additional commentary or evaluations:
User Prompt:'''

class Qwen2AudioEmbedder(nn.Module):
    def __init__(self, model_path, embed_dim=256, max_length=320, dtype=torch.float, device="cuda"):
        super().__init__()
        self.max_length = max_length
        self.device = device
        self.embed_dim = embed_dim

        self.model = Qwen2AudioForConditionalGeneration.from_pretrained(
            model_path,
            torch_dtype=dtype,
            device_map={"": int(os.environ.get("LOCAL_RANK", 0))}
        )
        # 禁止梯度回传
        self.model.requires_grad_(False)
        self.model.eval()
        self.processor = AutoProcessor.from_pretrained(model_path)
        
        # 添加投影层，从模型隐藏层维度(4096)映射到指定的embed_dim
        # 按理来说这一层也是会加入训练的呀
        self.proj = nn.Linear(4096, embed_dim, device=device, dtype=dtype)
        self.prefix = QWEN_AUDIO_PREFIX

    def forward(self, text, audio_data):
        """
        Args:
            text: 文本描述列表
            audio_data: 音频数据列表,每个元素是numpy数组
        Returns:
            字典包含 "output": 嵌入张量, "mask": 掩码张量
        """
        output, mask = self.encode(text, audio_data)
        output = self.projection(output)
        return {"output": output, "mask": mask}

    def encode(self, text, audio_data):
        """编码文本和音频到嵌入空间"""
        """编码文本和音频到嵌入空间"""
        batch_size = len(text)
        
        # 统一转换采样率 (如果需要的话) - 这一步应该在外部或这里批量处理
        processed_audios = []
        for audio in audio_data:
            if isinstance(audio, torch.Tensor):
                audio = audio.cpu().numpy()
            # 添加librosa.resample 操作
            audio=librosa.resample(audio, orig_sr=24000, target_sr=16000)
            processed_audios.append(audio)

        # 批量构建对话文本
        conversations = []
        for txt in text:
            conversation = [
                {"role": "user", "content": [
                    # 注意：此处audio字段先用None占位，后面再由processor处理
                    {"type": "audio", "audio": None}, 
                    {"type": "text", "text": txt}
                ]}
            ]
            # 使用 apply_chat_template 转换文本
            formatted_text = self.processor.apply_chat_template(conversation, add_generation_prompt=True, tokenize=False)
            conversations.append(formatted_text)

        with torch.no_grad():
            # 一次性批量处理整个batch的文本和音频
            # processor会自动对音频数据进行填充
            # padding的话是这里padding
            inputs = self.processor(
                text=conversations,
                audio=processed_audios,
                return_tensors="pt",
                sampling_rate=16000,
                padding=True,
                truncation=True  # 确保不会超过模型最大长度
            )

            # 将输入移动到设备
            inputs = {k: v.to(self.device) for k, v in inputs.items()}

            # 获取模型输出
            outputs = self.model(
                input_ids=inputs["input_ids"],
                attention_mask=inputs["attention_mask"],
                input_features=inputs["input_features"],
                feature_attention_mask=inputs["feature_attention_mask"],
                output_hidden_states=True,
            )

            # 提取最后一层隐藏状态
            hidden_states_full = outputs.hidden_states[-1]
            
            # 裁剪到最大长度
            # 批量处理后，所有样本的长度都已对齐，所以可以直接切片
            # embs = hidden_states_full[:, :self.max_length, :]
            # masks = inputs["attention_mask"][:, :self.max_length].bool() # attention_mask可以直接作为布尔掩码使用


            # --- 核心修改：确保输出长度固定为 self.max_length ---
            
            # 1. 截断或填充隐藏状态
            current_len = hidden_states_full.shape[1]
            if current_len > self.max_length:
                embs = hidden_states_full[:, :self.max_length, :]
            else:
                pad_width = self.max_length - current_len
                # 创建一个(batch_size, pad_width, hidden_size)的零张量用于填充
                padding = torch.zeros(
                    hidden_states_full.shape[0], 
                    pad_width, 
                    hidden_states_full.shape[2],
                    device=self.device, 
                    dtype=hidden_states_full.dtype
                )
                embs = torch.cat([hidden_states_full, padding], dim=1)

            # 2. 截断或填充掩码
            attention_mask = inputs["attention_mask"]
            if current_len > self.max_length:
                masks = attention_mask[:, :self.max_length].bool()
            else:
                pad_width = self.max_length - current_len
                # 创建一个(batch_size, pad_width)的False掩码
                mask_padding = torch.zeros(
                    attention_mask.shape[0], 
                    pad_width, 
                    device=self.device, 
                    dtype=torch.bool
                )
                masks = torch.cat([attention_mask.bool(), mask_padding], dim=1)
                
        return embs, masks

    def projection(self, x):
        """将嵌入映射到指定维度"""
        return self.proj(x)




if __name__ == "__main__":
    import argparse
    
    parser = argparse.ArgumentParser(description="Test Qwen Audio Encoder")
    parser.add_argument("--model_path", type=str, default="/mnt/petrelfs/taoye/workspace/model/qwen25audio", 
                        help="Path to Qwen Audio model")
    parser.add_argument("--embed_dim", type=int, default=4096,
                        help="Target embedding dimension after projection")
    args = parser.parse_args()
    
    print(f"Loading model from {args.model_path}...")
    
    # 初始化编码器
    device = "cuda" if torch.cuda.is_available() else "cpu"
    embedder = Qwen2AudioEmbedder(
        model_path=args.model_path,
        embed_dim=args.embed_dim,
        max_length=640,
        dtype=torch.float,
        device=device
    )
    
    # 准备测试批次
    captions = [
        "Describe this audio",
        "What musical instruments are being played in this recording?"
    ]
    
    # 直接加载音频数据
    audio_path = "/mnt/petrelfs/taoye/workspace/editing/data/add/add_fore_audio_caps_begin_1/audio/edit/syn_5.wav"
    audio_data = []
    for _ in range(len(captions)):
        waveform, sr = librosa.load(audio_path,sr=24000)
        # print(sr)
        audio_data.append(waveform)
    
    # 获取嵌入
    with torch.no_grad():
        output = embedder(captions, audio_data)
    
    # 打印结果
    print("模型输出的字典：")
    print(f"包含keys: {list(output.keys())}")
    
    print("\n输出张量的形状：")
    print(output['output'].shape)
    
    print("\n掩码张量的形状：")
    print(output['mask'].shape)
    
    # 验证嵌入维度是否符合预期
    assert output['output'].shape[-1] == args.embed_dim, f"输出维度 {output['output'].shape[-1]} 不等于预期维度 {args.embed_dim}"
    print(f"\n成功验证：输出维度 = {args.embed_dim}")
    
    # 显示样本嵌入值
    print(f"样本嵌入值:\n{output['output'][0, :5, :5]}")
    print(f"非零掩码位置数量: {output['mask'][0,:]}")  
    # 显示第一个样本中非零掩码位置的数量
    print(f"第一个样本的非零掩码位置数量: {output['mask'][0].sum().item()}")


