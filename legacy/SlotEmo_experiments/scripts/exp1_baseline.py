"""
实验1: SlotEmo基线验证
目标:
1. 验证Slot Attention在情感数据上的有效性
2. 验证模型能学到有意义的情感slot
3. 评估TopK选slot的可解释性
"""
import sys
import os
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import torch
import torch.nn as nn
import numpy as np
import pickle
import argparse
import time
from sklearn.metrics import accuracy_score, f1_score
from scipy.stats import pearsonr

from src.config import MODEL_CONFIG, TRAIN_CONFIG, ATTACHMENT2_PATH, OUTPUT_ROOT, EXPERIMENTS_ROOT, SEED
from src.slot_model import SlotEmo


def set_seed(seed):
    import random
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def normalize_features(data, mean_audio, std_audio, mean_vision, std_vision):
    """对audio和vision特征做标准化"""
    data = dict(data)  # 复制

    for split in ['train', 'valid', 'test']:
        d = dict(data[split])
        d['audio'] = (data[split]['audio'] - mean_audio) / (std_audio + 1e-8)
        d['vision'] = (data[split]['vision'] - mean_vision) / (std_vision + 1e-8)
        # 限制在合理范围内防止极端值
        d['audio'] = np.clip(d['audio'], -5, 5)
        d['vision'] = np.clip(d['vision'], -5, 5)
        data[split] = d

    return data


def load_data():
    """加载附件2数据, 并做标准化"""
    pkl_path = os.path.join(ATTACHMENT2_PATH, 'aligned_50.pkl')
    with open(pkl_path, 'rb') as f:
        data = pickle.load(f)

    # 计算训练集的均值和标准差
    audio_train = data['train']['audio']
    vision_train = data['train']['vision']

    mean_audio = audio_train.mean()
    std_audio = audio_train.std()
    mean_vision = vision_train.mean()
    std_vision = vision_train.std()

    print(f"audio: mean={mean_audio:.4f}, std={std_audio:.4f}")
    print(f"vision: mean={mean_vision:.4f}, std={std_vision:.4f}")

    # 标准化
    data = normalize_features(data, mean_audio, std_audio, mean_vision, std_vision)

    return data


def prepare_batch(data, indices, device):
    """准备batch数据"""
    text = torch.FloatTensor(data['text'][indices]).to(device)
    audio = torch.FloatTensor(data['audio'][indices]).to(device)
    vision = torch.FloatTensor(data['vision'][indices]).to(device)
    cls_labels = torch.LongTensor(data['classification_labels'][indices]).to(device)
    reg_labels = torch.FloatTensor(data['regression_labels'][indices]).to(device)
    return text, audio, vision, cls_labels, reg_labels


def train_epoch(model, data, optimizer, batch_size, device, sim_missing_ratio=0.1):
    """训练一个epoch"""
    model.train()
    total_loss = 0
    total_cls_loss = 0
    total_reg_loss = 0
    n_samples = 0

    indices = np.random.permutation(len(data['text']))
    n_batches = (len(indices) + batch_size - 1) // batch_size

    cls_criterion = nn.CrossEntropyLoss()
    reg_criterion = nn.MSELoss()

    for i in range(n_batches):
        batch_idx = indices[i*batch_size:(i+1)*batch_size]
        text, audio, vision, cls_labels, reg_labels = prepare_batch(data, batch_idx, device)

        # 模拟模态缺失(随机生成mask)
        B, T = text.shape[0], text.shape[1]
        mask_text = torch.ones(B, T, device=device)
        mask_audio = torch.ones(B, T, device=device)
        mask_vision = torch.ones(B, T, device=device)

        if sim_missing_ratio > 0:
            # 随机选择样本进行模态缺失模拟
            n_missing = int(B * sim_missing_ratio)
            if n_missing > 0:
                missing_idx = np.random.choice(B, n_missing, replace=False)
                # 随机选择缺失的模态
                missing_modality = np.random.choice(['audio', 'vision'])
                missing_len = int(T * 0.3)
                missing_start = np.random.randint(0, T - missing_len + 1)

                if missing_modality == 'audio':
                    mask_audio[missing_idx, missing_start:missing_start+missing_len] = 0
                else:
                    mask_vision[missing_idx, missing_start:missing_start+missing_len] = 0

        # 前向传播
        outputs = model(text, audio, vision, mask_text, mask_audio, mask_vision)

        # 计算损失
        cls_loss = cls_criterion(outputs['logits'], cls_labels)
        reg_loss = reg_criterion(outputs['regression'], reg_labels)
        # 限制数值稳定性
        if torch.isnan(cls_loss) or torch.isnan(reg_loss):
            print(f"NaN detected, skipping batch")
            continue

        loss = cls_loss + 0.5 * reg_loss

        # 反向传播
        optimizer.zero_grad()
        loss.backward()
        # 严格梯度裁剪
        torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=0.5)
        optimizer.step()

        total_loss += loss.item() * B
        total_cls_loss += cls_loss.item() * B
        total_reg_loss += reg_loss.item() * B
        n_samples += B

    return {
        'loss': total_loss / max(n_samples, 1),
        'cls_loss': total_cls_loss / max(n_samples, 1),
        'reg_loss': total_reg_loss / max(n_samples, 1),
    }


@torch.no_grad()
def evaluate(model, data, batch_size, device):
    """评估"""
    model.eval()

    all_preds = []
    all_labels = []
    all_reg_preds = []
    all_reg_labels = []

    indices = np.arange(len(data['text']))
    n_batches = (len(indices) + batch_size - 1) // batch_size

    for i in range(n_batches):
        batch_idx = indices[i*batch_size:(i+1)*batch_size]
        text, audio, vision, cls_labels, reg_labels = prepare_batch(data, batch_idx, device)

        B, T = text.shape[0], text.shape[1]
        mask_text = torch.ones(B, T, device=device)
        mask_audio = torch.ones(B, T, device=device)
        mask_vision = torch.ones(B, T, device=device)

        outputs = model(text, audio, vision, mask_text, mask_audio, mask_vision)

        preds = outputs['logits'].argmax(dim=-1).cpu().numpy()
        all_preds.extend(preds)
        all_labels.extend(cls_labels.cpu().numpy())
        all_reg_preds.extend(outputs['regression'].cpu().numpy())
        all_reg_labels.extend(reg_labels.cpu().numpy())

    # 分类指标
    acc = accuracy_score(all_labels, all_preds)
    f1 = f1_score(all_labels, all_preds, average='weighted')

    # 回归指标
    all_reg_preds = np.array(all_reg_preds)
    all_reg_labels = np.array(all_reg_labels)
    mae = np.mean(np.abs(all_reg_preds - all_reg_labels))
    if np.std(all_reg_preds) > 1e-8 and np.std(all_reg_labels) > 1e-8:
        pearson_corr, _ = pearsonr(all_reg_preds, all_reg_labels)
    else:
        pearson_corr = 0.0

    return {
        'accuracy': acc,
        'f1': f1,
        'mae': mae,
        'pearson': pearson_corr,
    }


def analyze_slot_interpretability(model, data, device, n_samples=5):
    """分析Slot的可解释性"""
    model.eval()

    print("\n" + "=" * 60)
    print("Slot 可解释性分析")
    print("=" * 60)

    # 随机选几个样本
    indices = np.random.choice(len(data['text']), n_samples, replace=False)

    with torch.no_grad():
        for idx in indices:
            text, audio, vision, cls_labels, reg_labels = prepare_batch(
                data, [idx], device
            )

            B, T = text.shape[0], text.shape[1]
            mask_text = torch.ones(B, T, device=device)
            mask_audio = torch.ones(B, T, device=device)
            mask_vision = torch.ones(B, T, device=device)

            outputs = model(text, audio, vision, mask_text, mask_audio, mask_vision)
            explanations = outputs['explanations']

            print(f"\n样本 {idx} (真实标签: {cls_labels.item()}, 强度: {reg_labels.item():.2f}):")
            pred = outputs['logits'].argmax(dim=-1).item()
            pred_reg = outputs['regression'].item()
            print(f"  预测: {pred}, 预测强度: {pred_reg:.2f}")

            # 模态权重
            modality_weights = explanations['modality_weights'][0].cpu().numpy()
            print(f"  模态权重: text={modality_weights[0]:.3f}, audio={modality_weights[1]:.3f}, vision={modality_weights[2]:.3f}")

            # 各模态的Top-K slot
            for mod_name in ['text', 'audio', 'vision']:
                slot_info = explanations[f'{mod_name}_slot_info']
                gate = slot_info['slot_gate'][0].cpu().numpy()
                keep_mask = slot_info['keep_mask'][0].cpu().numpy()
                top_k_idx = np.argsort(gate)[-3:][::-1]  # Top-3 slot

                gate_str = [f'{gate[i]:.3f}' for i in top_k_idx]
                print(f"  {mod_name} Top-3 slots: {top_k_idx.tolist()}, gates={gate_str}")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--epochs', type=int, default=20)
    parser.add_argument('--batch_size', type=int, default=32)
    parser.add_argument('--lr', type=float, default=3e-4)
    parser.add_argument('--exp_name', type=str, default='exp1_slot_baseline')
    args = parser.parse_args()

    set_seed(SEED)
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    print(f"使用设备: {device}")

    # 加载数据
    print("加载数据...")
    data = load_data()
    print(f"训练集: {len(data['train']['text'])} 样本")
    print(f"验证集: {len(data['valid']['text'])} 样本")

    # 创建模型 - 使用更稳定的配置
    config = {**MODEL_CONFIG}
    config['use_cross_modal'] = False
    config['use_recon'] = False
    config['temperature'] = 0.5  # 情感任务使用较温和的温度
    model = SlotEmo(config).to(device)

    n_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print(f"模型参数量: {n_params:,}")

    # 优化器 - 降低学习率
    optimizer = torch.optim.AdamW(
        model.parameters(),
        lr=args.lr,
        weight_decay=TRAIN_CONFIG['weight_decay']
    )
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=args.epochs)

    os.makedirs(OUTPUT_ROOT, exist_ok=True)
    os.makedirs(EXPERIMENTS_ROOT, exist_ok=True)

    # 训练循环
    print(f"\n开始训练 ({args.epochs} epochs)...")
    best_f1 = 0
    best_epoch = 0

    for epoch in range(args.epochs):
        start_time = time.time()

        # 训练
        train_metrics = train_epoch(
            model, data['train'], optimizer, args.batch_size, device,
            sim_missing_ratio=0.1
        )

        # 验证
        valid_metrics = evaluate(model, data['valid'], args.batch_size, device)

        scheduler.step()

        elapsed = time.time() - start_time

        print(f"Epoch {epoch+1}/{args.epochs} [{elapsed:.1f}s]")
        print(f"  Train - loss: {train_metrics['loss']:.4f}, cls_loss: {train_metrics['cls_loss']:.4f}")
        print(f"  Valid - acc: {valid_metrics['accuracy']:.4f}, f1: {valid_metrics['f1']:.4f}, "
              f"mae: {valid_metrics['mae']:.4f}, pearson: {valid_metrics['pearson']:.4f}")

        if valid_metrics['f1'] > best_f1:
            best_f1 = valid_metrics['f1']
            best_epoch = epoch + 1
            save_path = os.path.join(
                OUTPUT_ROOT,
                f'{args.exp_name}_best.pt'
            )
            torch.save(model.state_dict(), save_path)

    print(f"\n最佳F1: {best_f1:.4f} (epoch {best_epoch})")

    # 加载最佳模型进行可解释性分析
    save_path = os.path.join(
        OUTPUT_ROOT,
        f'{args.exp_name}_best.pt'
    )
    if os.path.exists(save_path):
        model.load_state_dict(torch.load(save_path))

    analyze_slot_interpretability(model, data['valid'], device)

    # 最终测试
    test_metrics = evaluate(model, data['test'], args.batch_size, device)
    print(f"\n最终测试结果:")
    print(f"  Accuracy: {test_metrics['accuracy']:.4f}")
    print(f"  F1: {test_metrics['f1']:.4f}")
    print(f"  MAE: {test_metrics['mae']:.4f}")
    print(f"  Pearson: {test_metrics['pearson']:.4f}")

    # 保存结果
    results_path = os.path.join(
        EXPERIMENTS_ROOT,
        f'{args.exp_name}_results.txt'
    )
    with open(results_path, 'w', encoding='utf-8') as f:
        f.write(f"实验: {args.exp_name}\n")
        f.write(f"最佳F1: {best_f1:.4f} (epoch {best_epoch})\n")
        f.write(f"\n测试集结果:\n")
        f.write(f"  Accuracy: {test_metrics['accuracy']:.4f}\n")
        f.write(f"  F1: {test_metrics['f1']:.4f}\n")
        f.write(f"  MAE: {test_metrics['mae']:.4f}\n")
        f.write(f"  Pearson: {test_metrics['pearson']:.4f}\n")

    print(f"\n结果已保存到: {results_path}")


if __name__ == '__main__':
    main()
