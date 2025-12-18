import logging
from dataclasses import dataclass
from typing import Optional

import torch
import torch.nn as nn
import torch.nn.functional as F

# 假设这些是你原来的导入
from .mmdit_layers_new import compute_rope_rotations
from .mmdit_layers_new import TimestepEmbedder
from .mmdit_layers_new import MLP, ChannelLastConv1d, ConvMLP
from .mmdit_layers_new import (FinalBlock, MMDitSingleBlock, JointBlock_AT)

log = logging.getLogger()


@dataclass
class PreprocessedConditions:
    text_f: torch.Tensor
    text_f_c: torch.Tensor


class MMAudio(nn.Module):
    """
    一个修改版的 MMAudio 接口尽量和LayerFusionAudioDiT一致。
    """
    def __init__(self,
                 *,
                 latent_dim: int,
                 text_dim: int,
                 hidden_dim: int,
                 depth: int,
                 fused_depth: int,
                 num_heads: int,
                 mlp_ratio: float = 4.0,
                 latent_seq_len: int,
                 text_seq_len: int = 640,
                 # --- 新增参数，对齐 LayerFusionAudioDiT ---
                 ta_context_dim: int,
                 ta_context_fusion: str = 'add', # 'add' or 'concat'
                 ta_context_norm: bool = False,
                 # --- 其他原有参数 ---
                 empty_string_feat: Optional[torch.Tensor] = None,
                 v2: bool = False) -> None:
        super().__init__()

        self.v2 = v2
        self.latent_dim = latent_dim
        self._latent_seq_len = latent_seq_len
        self._text_seq_len = text_seq_len
        self.hidden_dim = hidden_dim
        self.num_heads = num_heads
        
        # --- 1. time_aligned_context 的投影层 ---
        # 我们在这里定义一个投影层，而不是在每个 block 里都定义一个。
        # 这样更高效，也符合你代码注释中的想法：“现在是每一层proj，改为不映射”。
        # 我们的方案是：只映射一次，然后传递给所有层。
        self.ta_context_fusion = ta_context_fusion
        self.ta_context_norm_flag = ta_context_norm
        
        if self.ta_context_fusion == "add":
            # 如果是相加融合，将 ta_context 投射到和 latent 一样的维度 (hidden_dim)
            self.ta_context_projection = nn.Linear(ta_context_dim, hidden_dim, bias=False)
            self.ta_context_norm = nn.LayerNorm(ta_context_dim) if self.ta_context_norm_flag else nn.Identity()
        elif self.ta_context_fusion == "concat":
            # 如果是拼接融合，在 block 内部处理，这里不需要主投影层
            # 但你的原始代码在concat后也有一个projection，我们可以在 block 内部实现
            # 为了简化，这里先假设主要的融合逻辑在 block 内部
            self.ta_context_projection = nn.Identity()
            self.ta_context_norm = nn.Identity()
        else:
            raise ValueError(f"Unknown ta_context_fusion type: {ta_context_fusion}")


        # --- 原有的输入投影层 (基本不变) ---
        # 现在我的输入要变为editing，需要变为latent*2
        self.audio_input_proj = nn.Sequential(
            ChannelLastConv1d(latent_dim*2, hidden_dim, kernel_size=7, padding=3),
            nn.SELU(),
            ConvMLP(hidden_dim, hidden_dim * 4, kernel_size=7, padding=3),
        )
        self.text_input_proj = nn.Sequential(
            nn.Linear(text_dim, hidden_dim),
            MLP(hidden_dim, hidden_dim * 4),
        )
            
        self.text_cond_proj = nn.Linear(hidden_dim, hidden_dim)
        self.global_cond_mlp = MLP(hidden_dim, hidden_dim * 4)
        

        # 
        self.t_embed = TimestepEmbedder(hidden_dim, frequency_embedding_size=256, max_period=10000)
            
        # --- Transformer Blocks (基本不变) ---
        # **重要**: 你需要修改 JointBlock_AT 和 MMDitSingleBlock 的 forward 定义来接收 `time_aligned_context`
        self.joint_blocks = nn.ModuleList([
            JointBlock_AT(hidden_dim, num_heads, mlp_ratio=mlp_ratio, pre_only=(i == depth - fused_depth - 1))
            for i in range(depth - fused_depth)
        ])
        self.fused_blocks = nn.ModuleList([
            MMDitSingleBlock(hidden_dim, num_heads, mlp_ratio=mlp_ratio, kernel_size=3, padding=1)
            for i in range(fused_depth)
        ])
        
        # --- 输出层 (不变) ---
        self.final_layer = FinalBlock(hidden_dim, latent_dim)

        
        if empty_string_feat is None:
            empty_string_feat = torch.zeros((text_seq_len, text_dim))
            
        self.empty_string_feat = nn.Parameter(empty_string_feat, requires_grad=False)
        
        self.initialize_weights()
        self.initialize_rotations()

        # 初始化旋转矩阵
    # 这里的旋转矩阵是为了实现 RoPE
    def initialize_rotations(self):
        base_freq = 1.0


        latent_rot = compute_rope_rotations(self._latent_seq_len,
                                            self.hidden_dim // self.num_heads,
                                            10000,
                                            freq_scaling=base_freq,
                                            device="cuda" if torch.cuda.is_available() else "cpu")

        self.register_buffer('latent_rot', latent_rot, persistent=False)
        # self.clip_rot = nn.Buffer(clip_rot, persistent=False)

    def update_seq_lengths(self, latent_seq_len: int, clip_seq_len: int, sync_seq_len: int) -> None:
        self._latent_seq_len = latent_seq_len
        self._clip_seq_len = clip_seq_len
        self._sync_seq_len = sync_seq_len
        self.initialize_rotations()

    def initialize_weights(self):

        def _basic_init(module):
            if isinstance(module, nn.Linear):
                torch.nn.init.xavier_uniform_(module.weight)
                if module.bias is not None:
                    nn.init.constant_(module.bias, 0)

        self.apply(_basic_init)

        # Initialize timestep embedding MLP:
        nn.init.normal_(self.t_embed.mlp[0].weight, std=0.02)
        nn.init.normal_(self.t_embed.mlp[2].weight, std=0.02)

        # Zero-out adaLN modulation layers in DiT blocks:兼容性保护
        for block in self.joint_blocks:
            nn.init.constant_(block.latent_block.adaLN_modulation[-1].weight, 0)
            nn.init.constant_(block.latent_block.adaLN_modulation[-1].bias, 0)
            nn.init.constant_(block.text_block.adaLN_modulation[-1].weight, 0)
            nn.init.constant_(block.text_block.adaLN_modulation[-1].bias, 0)
        for block in self.fused_blocks:
            nn.init.constant_(block.adaLN_modulation[-1].weight, 0)
            nn.init.constant_(block.adaLN_modulation[-1].bias, 0)

        # Zero-out output layers:
        nn.init.constant_(self.final_layer.adaLN_modulation[-1].weight, 0)
        nn.init.constant_(self.final_layer.adaLN_modulation[-1].bias, 0)
        nn.init.constant_(self.final_layer.conv.weight, 0)
        nn.init.constant_(self.final_layer.conv.bias, 0)


    
    def preprocess_conditions(self, text_f: torch.Tensor) -> PreprocessedConditions:
        # 预处理文本条件
        # assert text_f.shape[1] == self._text_seq_len, f'{text_f.shape=} {self._text_seq_len=}'
        bs = text_f.shape[0]

        # 这里固定外部的llm_embedding
        text_f = self.text_input_proj(text_f)
        # 全局的条件
        text_f_c = self.text_cond_proj(text_f.mean(dim=1))
        return PreprocessedConditions(text_f=text_f, text_f_c=text_f_c)

    def predict_flow(self, x: torch.Tensor, timesteps: torch.Tensor, conditions: PreprocessedConditions, time_aligned_context: torch.Tensor, x_mask=None, context_mask=None) -> torch.Tensor:
        """
        核心的预测流程，现在加入了 time_aligned_context

        to do :加入这个padding的mask部分
        """
        assert x.shape[2] == self._latent_seq_len, f'{x.shape=} {self._latent_seq_len=}'
        
        # 1. 预处理各种输入
        text_f = conditions.text_f
        text_f_c = conditions.text_f_c
        
        timesteps = timesteps.to(x.dtype)  # 保持和输入张量同 dtype

        global_c = self.global_cond_mlp(text_f_c)  # (B, D)
        
        # 2. 融合 timestep
        global_c = self.t_embed(timesteps).unsqueeze(1) + global_c.unsqueeze(1) # (B, 1, D)
        extended_c = global_c # 这个将作为 AdaLN 的条件
        """
        这里决定了x的形状,需要debug
        """
        # 3. **处理 time_aligned_context** 这里第一种方式是直接和latent进行融合，然后投影 
        # 从128->256 
        x = torch.cat([x.transpose(1, 2), time_aligned_context], dim=-1)
        latent = self.audio_input_proj(x)  # (B, N, D)

        # 4. 依次通过 Transformer Blocks
        for block in self.joint_blocks:
            # **你需要修改 JointBlock_AT.forward**
            latent, text_f = block(latent, text_f, global_c, extended_c,
                                           self.latent_rot, latent_mask=x_mask, text_mask=context_mask) 

        for block in self.fused_blocks:
            # 这里的x_mask由于只有一个模态都是这个
            latent = block(latent, extended_c, self.latent_rot,key_padding_mask=x_mask,query_mask=x_mask)

        # 5. 通过输出层
        flow = self.final_layer(latent, global_c)
        return flow

    def forward(self, 
                x: torch.Tensor, 
                timesteps: torch.Tensor,
                context: torch.Tensor,
                time_aligned_context: torch.Tensor,
                # 其他参数可以根据需要添加，我现在先添加mask信息进入
                x_mask=None,
                context_mask=None,
               ) -> torch.Tensor:
        """
        模型主入口，接口已对齐 LayerFusionAudioDiT。
        - x: 噪声 latent, shape (B, N_latent, latent_dim)
        - timesteps: 时间步, shape (B,)
        - context: 文本条件, shape (B, N_text, text_dim)
        - time_aligned_context: 时间对齐的条件, shape (B, N_ta, ta_context_dim)
        """
        # 兼容整数时间步
        if timesteps.dim() == 0:
            timesteps = timesteps.expand(x.shape[0]).to(x.device, dtype=torch.long)
        
        # 预处理文本条件
        # 注意：在原始 `MMAudio` 中，`text_f` 是 forward 的直接输入。
        # 为了接口对齐，我们现在接受原始的 `context` 并在这里处理。
        text_conditions = self.preprocess_conditions(context)
        
        # 调用核心预测流
        flow = self.predict_flow(x, timesteps, text_conditions, time_aligned_context, x_mask, context_mask)
        
        # 这里的flow添加一个1，2维的转置
        # 使其符合 (B, latent_dim，n) 的格式
        flow = flow.transpose(1, 2)

        # 需要和x_mask的长度对齐，只需要保留x_mask为1的部分


        return flow

    # ... (get_empty_string_sequence, get_empty_conditions, ode_wrapper 等方法基本可以复用) ...
    # 注意：如果 ode_wrapper 或 get_empty_conditions 被使用，
    # 它们也需要处理新的 time_aligned_context，可能需要一个空的 time_aligned_context。
    # 这里为了简化，暂时不修改它们。
    # 暂时删除device

    @property
    def latent_seq_len(self) -> int:
        return self._latent_seq_len
    

# 我的latent对应的是b,500，128

def small_16k(**kwargs) -> MMAudio:
    num_heads = 16
    return MMAudio(latent_dim=128,
                   text_dim=1024,
                   hidden_dim=64 * num_heads,
                   depth=12,
                   fused_depth=8,
                   num_heads=num_heads,
                   latent_seq_len=500,
                   **kwargs)




if __name__ == '__main__':
    # 1. 设置测试参数
    batch_size = 4
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")

    # 模型配置，特别是新增的 ta_context_dim
    # 根据你的逻辑 `latent*2`，ta_context_dim 应该和 latent_dim 一样
    config = {
        "ta_context_dim": 128,
        "ta_context_fusion": "concat", # 你的逻辑是 concat，所以这里设为 concat
        "ta_context_norm": False
    }

    # 2. 实例化模型
    try:
        model = small_16k(**config).to(device)
        model.eval() # 使用评估模式
        print("Model instantiated successfully!")
    except Exception as e:
        print(f"Error during model instantiation: {e}")
        exit()


    # print the number of parameters in terms of millions
    num_params = sum(p.numel() for p in model.parameters()) / 1e6
    print(f'Number of parameters: {num_params:.2f}M')

    # 3. 创建符合模型输入的虚拟数据
    # 从 small_16k 和你的 forward 方法定义中获取维度信息
    latent_dim = 128
    latent_seq_len = 500
    text_dim = 1024
    # 
    text_seq_len = 640
    ta_context_dim = config["ta_context_dim"]

    dummy_x = torch.randn(batch_size,latent_dim, latent_seq_len, device=device)
    dummy_timesteps = torch.randint(0, 1000, (batch_size,), device=device)
    dummy_context = torch.randn(batch_size, text_seq_len, text_dim, device=device)
    
    # 这里的 time_aligned_context 形状需要和 x 一致，以便在特征维度上拼接
    dummy_ta_context = torch.randn(batch_size, latent_seq_len, ta_context_dim, device=device)

    print("\n--- Input Shapes ---")
    print(f"x (latent):           {dummy_x.shape}")
    print(f"timesteps:            {dummy_timesteps.shape}")
    print(f"context (text):       {dummy_context.shape}")
    print(f"time_aligned_context: {dummy_ta_context.shape}")
    print("--------------------\n")
    
    # 4. 执行前向传播
    try:
        with torch.no_grad(): # 在验证时不需要计算梯度
            output = model(
                x=dummy_x,
                timesteps=dummy_timesteps,
                context=dummy_context,
                time_aligned_context=dummy_ta_context
            )
        print("✅ Forward pass successful!")
        print(f"Output shape: {output.shape}")

        # 5. 验证输出形状
        expected_shape = (batch_size, latent_seq_len, latent_dim)
        assert output.shape == expected_shape, \
            f"Output shape mismatch! Expected {expected_shape}, but got {output.shape}"
        print("✅ Output shape is correct!")

    except Exception as e:
        print(f"❌ Error during forward pass: {e}")
        import traceback
        traceback.print_exc()