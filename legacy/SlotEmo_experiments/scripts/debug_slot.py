"""
终极debug: 检查slot gate为什么均匀
"""
import sys
import os
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import torch
import numpy as np
from src.slot_modules import MoESlotDecoder

torch.manual_seed(42)
device = 'cuda'

# 模拟训练后的slot
slots = torch.randn(1, 16, 128).to(device) * 0.05

decoder = MoESlotDecoder(dim=128, num_slots=16, num_classes=3, temperature=0.5).to(device)
decoder.train()

# 训练前
result = decoder(slots)
print("训练前:")
print(f"keep_score: {result['keep_score'][0].detach().cpu().numpy()}")
print(f"softmax values (before mask): {torch.softmax(result['keep_score']/0.5, dim=-1)[0].detach().cpu().numpy()}")
print(f"keep_mask: {result['keep_mask'][0].detach().cpu().numpy().astype(int)}")
print(f"slot_gate: {result['slot_gate'][0].detach().cpu().numpy()}")
