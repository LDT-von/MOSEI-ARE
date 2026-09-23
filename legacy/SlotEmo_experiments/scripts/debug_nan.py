"""
手算gumbel_noise
"""
import torch
torch.manual_seed(42)
device = 'cuda'

# 原始实现
def gumbel_noise_buggy(t):
    noise = torch.rand_like(t)
    return -torch.log(-torch.log(noise.clamp(min=1e-20)).clamp(min=1e-20)).clamp(min=1e-20)

# 修复: -log(noise) 是 u = -log(noise), 然后 -log(u) = gumbel
# 但 noise本身是(0,1)的随机数, log(noise)是负无穷大
# 直接用 Gumbel(0,1) 公式: -log(-log(U))
# 更稳定的实现
def gumbel_noise_fixed(t):
    noise = torch.rand_like(t)
    # 防止 noise=0
    noise = noise + 1e-20
    noise = noise.clamp(min=1e-20, max=1-1e-7)
    return -torch.log(-torch.log(noise))

# 测试
logits = torch.randn(2, 16).to(device) * 0.001

print("Original:")
n1 = gumbel_noise_buggy(logits)
print(f"  range: [{n1.min():.4f}, {n1.max():.4f}], NaN: {torch.isnan(n1).any()}")

print("Fixed:")
n2 = gumbel_noise_fixed(logits)
print(f"  range: [{n2.min():.4f}, {n2.max():.4f}], NaN: {torch.isnan(n2).any()}")
print(f"  sample: {n2[0, :5]}")
