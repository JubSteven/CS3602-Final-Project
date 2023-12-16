import torch
import torch.nn as nn
import torch.nn.functional as F


class BertHiddenGroupFusion(nn.Module):
    def __init__(self):
        super(BertHiddenGroupFusion, self).__init__()
        self.fusion_mlp = nn.Sequential(
            nn.Linear(768 * 4, 1024),
            nn.GELU(),  # GELU是Bert中使用的激活函数，这样一致性可能会更好
            nn.LayerNorm(1024),
            nn.Linear(1024, 768),
            nn.GELU(),
            nn.LayerNorm(768)
        )
        self.self_attention = nn.MultiheadAttention(embed_dim=768, num_heads=3)
        self.residual = nn.Linear(768, 768)

    def forward(self, x):
        batch_size = x.size(0)
        output = torch.zeros(batch_size, 3, 26, 768, device=x.device)

        for i in range(3):
            group = x[:, i * 4:(i + 1) * 4, :, :]
            group = group.view(batch_size, 26, -1)
            fused = self.fusion_mlp(group)

            # 残差连接
            residual = self.residual(fused)
            fused = fused + residual

            # 自注意力机制
            fused = fused.permute(1, 0, 2)  # 调整维度以适应自注意力模块
            attn_output, _ = self.self_attention(fused, fused, fused)
            attn_output = attn_output.permute(1, 0, 2)  # 恢复原始维度

            output[:, i, :, :] = attn_output

        return output


# 创建模型实例
model = BertHiddenGroupFusion()

# 创建一个示例输入张量，大小为 [batch_size, 32, 26, 768]
# 假设batch_size为1
example_input = torch.randn(4, 12, 26, 768)

# 通过模型传递输入并获取输出
output = model(example_input)

# 检查输出张量的形状是否正确
print(output.shape)
