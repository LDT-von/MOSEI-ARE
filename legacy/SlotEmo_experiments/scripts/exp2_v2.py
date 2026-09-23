"""
实验2: SlotEmo v2 优化版
改进:
1. 加入sparse loss让slot专门化
2. 加dropout防止过拟合
3. 支持关键证据定位(返回attention)
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
from src.slot_model_v2 import SlotEmoV2


def set_seed(seed):
    import random
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def normalize_features(data):
    """对audio和vision特征做标准化"""
    audio_train = data['train']['audio']
    vision_train = data['train']['vision']
    mean_audio, std_audio = audio_train.mean(), audio_train.std()
    mean_vision, std_vision = vision_train.mean(), vision_train.std()

    print(f"audio: mean={mean_audio:.4f}, std={std_audio:.4f}")
    print(f"vision: mean={mean_vision:.4f}, std={std_vision:.4f}")

    data = dict(data)
    for split in ['train', 'valid', 'test']:
        d = dict(data[split])
        d['audio'] = np.clip((data[split]['audio'] - mean_audio) / (std_audio + 1e-8), -5, 5)
        d['vision'] = np.clip((data[split]['vision'] - mean_vision) / (std_vision + 1e-8), -5, 5)
        data[split] = d

    return data


def load_data():
    pkl_path = os.path.join(ATTACHMENT2_PATH, 'aligned_50.pkl')
    with open(pkl_path, 'rb') as f:
        data = pickle.load(f)
    return normalize_features(data)


def prepare_batch(data, indices, device):
    text = torch.FloatTensor(data['text'][indices]).to(device)
    audio = torch.FloatTensor(data['audio'][indices]).to(device)
    vision = torch.FloatTensor(data['vision'][indices]).to(device)
    cls_labels = torch.LongTensor(data['classification_labels'][indices]).to(device)
    reg_labels = torch.FloatTensor(data['regression_labels'][indices]).to(device)
    return text, audio, vision, cls_labels, reg_labels


def train_epoch(model, data, optimizer, batch_size, device, sim_missing_ratio=0.1, sparse_weight=0.1):
    model.train()
    total_loss = 0
    total_cls_loss = 0
    total_reg_loss = 0
    total_sparse_loss = 0
    n_samples = 0

    indices = np.random.permutation(len(data['text']))
    n_batches = (len(indices) + batch_size - 1) // batch_size

    cls_criterion = nn.CrossEntropyLoss()
    reg_criterion = nn.MSELoss()

    for i in range(n_batches):
        batch_idx = indices[i*batch_size:(i+1)*batch_size]
        text, audio, vision, cls_labels, reg_labels = prepare_batch(data, batch_idx, device)

        B, T = text.shape[0], text.shape[1]
        mask_text = torch.ones(B, T, device=device)
        mask_audio = torch.ones(B, T, device=device)
        mask_vision = torch.ones(B, T, device=device)

        if sim_missing_ratio > 0:
            n_missing = int(B * sim_missing_ratio)
            if n_missing > 0:
                missing_idx = np.random.choice(B, n_missing, replace=False)
                missing_modality = np.random.choice(['audio', 'vision'])
                missing_len = int(T * 0.3)
                missing_start = np.random.randint(0, T - missing_len + 1)

                if missing_modality == 'audio':
                    mask_audio[missing_idx, missing_start:missing_start+missing_len] = 0
                else:
                    mask_vision[missing_idx, missing_start:missing_start+missing_len] = 0

        outputs = model(text, audio, vision, mask_text, mask_audio, mask_vision, return_attention=False)

        cls_loss = cls_criterion(outputs['logits'], cls_labels)
        reg_loss = reg_criterion(outputs['regression'], reg_labels)
        sparse_loss = outputs['sparse_loss']

        loss = cls_loss + 0.5 * reg_loss + sparse_weight * sparse_loss

        if torch.isnan(loss):
            continue

        optimizer.zero_grad()
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=0.5)
        optimizer.step()

        total_loss += loss.item() * B
        total_cls_loss += cls_loss.item() * B
        total_reg_loss += reg_loss.item() * B
        total_sparse_loss += sparse_loss.item() * B
        n_samples += B

    return {
        'loss': total_loss / max(n_samples, 1),
        'cls_loss': total_cls_loss / max(n_samples, 1),
        'reg_loss': total_reg_loss / max(n_samples, 1),
        'sparse_loss': total_sparse_loss / max(n_samples, 1),
    }


@torch.no_grad()
def evaluate(model, data, batch_size, device):
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

        outputs = model(text, audio, vision, mask_text, mask_audio, mask_vision, return_attention=False)

        preds = outputs['logits'].argmax(dim=-1).cpu().numpy()
        all_preds.extend(preds)
        all_labels.extend(cls_labels.cpu().numpy())
        all_reg_preds.extend(outputs['regression'].cpu().numpy())
        all_reg_labels.extend(reg_labels.cpu().numpy())

    acc = accuracy_score(all_labels, all_preds)
    f1 = f1_score(all_labels, all_preds, average='weighted')

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


def analyze_slot_interpretability_v2(model, data, device, n_samples=3):
    """分析Slot的可解释性 v2 - 加入attention分析"""
    model.eval()

    print("\n" + "=" * 60)
    print("Slot 可解释性分析 (V2)")
    print("=" * 60)

    indices = np.random.choice(len(data['text']), n_samples, replace=False)

    with torch.no_grad():
        for idx in indices:
            text, audio, vision, cls_labels, reg_labels = prepare_batch(data, [idx], device)

            B, T = text.shape[0], text.shape[1]
            mask_text = torch.ones(B, T, device=device)
            mask_audio = torch.ones(B, T, device=device)
            mask_vision = torch.ones(B, T, device=device)

            outputs = model(text, audio, vision, mask_text, mask_audio, mask_vision, return_attention=True)
            explanations = outputs['explanations']

            raw_text = data['raw_text'][idx] if 'raw_text' in data else f"sample {idx}"
            if len(str(raw_text)) > 200:
                raw_text = str(raw_text)[:200] + "..."

            print(f"\n样本 {idx}")
            print(f"  文本: {raw_text}")
            print(f"  真实: 标签={['Negative', 'Neutral', 'Positive'][cls_labels.item()]}, 强度={reg_labels.item():.2f}")
            pred = outputs['logits'].argmax(dim=-1).item()
            print(f"  预测: 标签={['Negative', 'Neutral', 'Positive'][pred]}, 强度={outputs['regression'].item():.2f}")

            # 模态权重
            modality_weights = explanations['modality_weights'][0].cpu().numpy()
            print(f"  模态贡献度: text={modality_weights[0]:.3f}, audio={modality_weights[1]:.3f}, vision={modality_weights[2]:.3f}")

            # 各模态的Top-3 slot
            for mod_name in ['text', 'audio', 'vision']:
                slot_info = explanations[f'{mod_name}_slot_info']
                gate = slot_info['slot_gate'][0].cpu().numpy()
                top_k_idx = np.argsort(gate)[-3:][::-1]
                gate_str = [f'{gate[i]:.3f}' for i in top_k_idx]
                print(f"  {mod_name} Top-3 slots: {top_k_idx.tolist()}, gates={gate_str}")

                # 关键时间片段
                if f'{mod_name}_time_importance' in explanations:
                    time_imp = explanations[f'{mod_name}_time_importance'][0].cpu().numpy()
                    top_time = np.argsort(time_imp)[-5:][::-1]
                    print(f"    关键时间步: {top_time.tolist()}")
                    print(f"    重要性值: {[f'{time_imp[t]:.3f}' for t in top_time]}")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--epochs', type=int, default=30)
    parser.add_argument('--batch_size', type=int, default=32)
    parser.add_argument('--lr', type=float, default=2e-4)
    parser.add_argument('--dropout', type=float, default=0.3)
    parser.add_argument('--sparse_weight', type=float, default=0.1)
    parser.add_argument('--temperature', type=float, default=2.0)
    parser.add_argument('--exp_name', type=str, default='exp2_slot_v2')
    args = parser.parse_args()

    set_seed(SEED)
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    print(f"使用设备: {device}")

    print("加载数据...")
    data = load_data()
    print(f"训练集: {len(data['train']['text'])} 样本")
    print(f"验证集: {len(data['valid']['text'])} 样本")

    # v2 配置
    config = {**MODEL_CONFIG}
    config['use_cross_modal'] = False
    config['use_recon'] = False
    config['temperature'] = args.temperature
    config['dropout'] = args.dropout

    model = SlotEmoV2(config).to(device)
    n_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print(f"模型参数量: {n_params:,}")

    optimizer = torch.optim.AdamW(
        model.parameters(),
        lr=args.lr,
        weight_decay=1e-3,  # 增加正则化
    )
    scheduler = torch.optim.lr_scheduler.CosineAnnealingWarmRestarts(optimizer, T_0=10, T_mult=2)

    os.makedirs(OUTPUT_ROOT, exist_ok=True)
    os.makedirs(EXPERIMENTS_ROOT, exist_ok=True)
    print(f"\n开始训练 ({args.epochs} epochs)...")
    print(f"配置: dropout={args.dropout}, sparse_weight={args.sparse_weight}, temperature={args.temperature}")

    best_f1 = 0
    best_epoch = 0
    patience = 8
    patience_counter = 0

    for epoch in range(args.epochs):
        start_time = time.time()

        train_metrics = train_epoch(
            model, data['train'], optimizer, args.batch_size, device,
            sim_missing_ratio=0.1, sparse_weight=args.sparse_weight
        )

        valid_metrics = evaluate(model, data['valid'], args.batch_size, device)

        scheduler.step()

        elapsed = time.time() - start_time

        print(f"Epoch {epoch+1}/{args.epochs} [{elapsed:.1f}s]")
        print(f"  Train - loss: {train_metrics['loss']:.4f}, cls: {train_metrics['cls_loss']:.4f}, "
              f"sparse: {train_metrics['sparse_loss']:.4f}")
        print(f"  Valid - acc: {valid_metrics['accuracy']:.4f}, f1: {valid_metrics['f1']:.4f}, "
              f"mae: {valid_metrics['mae']:.4f}, pearson: {valid_metrics['pearson']:.4f}")

        if valid_metrics['f1'] > best_f1:
            best_f1 = valid_metrics['f1']
            best_epoch = epoch + 1
            patience_counter = 0
            save_path = os.path.join(
                OUTPUT_ROOT,
                f'{args.exp_name}_best.pt'
            )
            torch.save(model.state_dict(), save_path)
        else:
            patience_counter += 1
            if patience_counter >= patience:
                print(f"早停触发 (patience={patience})")
                break

    print(f"\n最佳F1: {best_f1:.4f} (epoch {best_epoch})")

    save_path = os.path.join(
        OUTPUT_ROOT,
        f'{args.exp_name}_best.pt'
    )
    if os.path.exists(save_path):
        model.load_state_dict(torch.load(save_path))

    # 可解释性分析
    analyze_slot_interpretability_v2(model, data['valid'], device)

    # 测试
    test_metrics = evaluate(model, data['test'], args.batch_size, device)
    print(f"\n最终测试结果:")
    print(f"  Accuracy: {test_metrics['accuracy']:.4f}")
    print(f"  F1: {test_metrics['f1']:.4f}")
    print(f"  MAE: {test_metrics['mae']:.4f}")
    print(f"  Pearson: {test_metrics['pearson']:.4f}")

    results_path = os.path.join(
        EXPERIMENTS_ROOT,
        f'{args.exp_name}_results.txt'
    )
    with open(results_path, 'w', encoding='utf-8') as f:
        f.write(f"实验: {args.exp_name}\n")
        f.write(f"配置: dropout={args.dropout}, sparse_weight={args.sparse_weight}, temp={args.temperature}\n")
        f.write(f"最佳F1: {best_f1:.4f} (epoch {best_epoch})\n")
        f.write(f"\n测试集结果:\n")
        f.write(f"  Accuracy: {test_metrics['accuracy']:.4f}\n")
        f.write(f"  F1: {test_metrics['f1']:.4f}\n")
        f.write(f"  MAE: {test_metrics['mae']:.4f}\n")
        f.write(f"  Pearson: {test_metrics['pearson']:.4f}\n")

    print(f"\n结果已保存到: {results_path}")


if __name__ == '__main__':
    main()
