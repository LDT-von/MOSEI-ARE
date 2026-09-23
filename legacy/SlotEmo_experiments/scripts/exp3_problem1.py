"""
实验3: 历史SlotEmo探索，并非正式问题1原视频解决方案
基于SlotSPE的核心思想分析附件2特征:
- 特征提取过程的数学建模
- 时序对齐机制的分析
- 基于附件2特征的预测

关键改进:
1. 回退到稳定的v2配置 (temperature=0.5, 不强制专门化)
2. 重点分析Slot对原始输入的attention (= 时序对齐矩阵)
3. 提供可解释的特征提取过程
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
import json
from sklearn.metrics import accuracy_score, f1_score, classification_report
from scipy.stats import pearsonr

from src.config import MODEL_CONFIG, TRAIN_CONFIG, ATTACHMENT2_PATH, ATTACHMENT1_PATH, OUTPUT_ROOT, EXPERIMENTS_ROOT, SEED
from src.slot_model_v2 import SlotEmoV2


def set_seed(seed):
    import random
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def normalize_features(data):
    """标准化audio和vision特征"""
    audio_train = data['train']['audio']
    vision_train = data['train']['vision']
    mean_audio, std_audio = audio_train.mean(), audio_train.std()
    mean_vision, std_vision = vision_train.mean(), vision_train.std()

    data = dict(data)
    for split in ['train', 'valid', 'test']:
        d = dict(data[split])
        d['audio'] = np.clip((data[split]['audio'] - mean_audio) / (std_audio + 1e-8), -5, 5)
        d['vision'] = np.clip((data['split' if False else split]['vision'] - mean_vision) / (std_vision + 1e-8), -5, 5)
        data[split] = d

    return data, (mean_audio, std_audio), (mean_vision, std_vision)


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


def train_epoch(model, data, optimizer, batch_size, device, sparse_weight=0.01):
    """简化训练 - 降低sparse_weight避免干扰"""
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

        B, T = text.shape[0], text.shape[1]
        mask_text = torch.ones(B, T, device=device)
        mask_audio = torch.ones(B, T, device=device)
        mask_vision = torch.ones(B, T, device=device)

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
        n_samples += B

    return {
        'loss': total_loss / max(n_samples, 1),
        'cls_loss': total_cls_loss / max(n_samples, 1),
        'reg_loss': total_reg_loss / max(n_samples, 1),
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
    }, all_preds, all_labels


def analyze_temporal_alignment(model, data, device, n_samples=3):
    """
    分析时序对齐机制
    Slot attention的核心机制: 每个slot"绑定"到特定的时间步
    这就是天然的"时序对齐矩阵"
    """
    model.eval()

    print("\n" + "=" * 60)
    print("时序对齐分析 (基于Slot Attention)")
    print("=" * 60)

    indices = np.random.choice(len(data['text']), n_samples, replace=False)

    alignment_results = []

    with torch.no_grad():
        for idx in indices:
            text, audio, vision, cls_labels, reg_labels = prepare_batch(data, [idx], device)

            B, T = text.shape[0], text.shape[1]
            mask_text = torch.ones(B, T, device=device)
            mask_audio = torch.ones(B, T, device=device)
            mask_vision = torch.ones(B, T, device=device)

            outputs = model(text, audio, vision, mask_text, mask_audio, mask_vision, return_attention=True)
            explanations = outputs['explanations']

            label_name = ['Negative', 'Neutral', 'Positive'][cls_labels.item()]
            pred = outputs['logits'].argmax(dim=-1).item()
            pred_name = ['Negative', 'Neutral', 'Positive'][pred]

            print(f"\n样本 {idx}: 真实={label_name}, 预测={pred_name}")

            # 分析每个模态的"对齐矩阵"
            for mod_name in ['text', 'audio', 'vision']:
                if f'{mod_name}_time_importance' in explanations:
                    time_imp = explanations[f'{mod_name}_time_importance'][0].cpu().numpy()
                    top_time = np.argsort(time_imp)[-5:][::-1]

                    # 时序对齐: 找出每个slot对齐到的时间步
                    slot_info = explanations[f'{mod_name}_slot_info']
                    gate = slot_info['slot_gate'][0].cpu().numpy()

                    # 找到每个slot的最主要时间步
                    print(f"  {mod_name}: Top关键时间步={top_time.tolist()}, 重要性={[f'{time_imp[t]:.3f}' for t in top_time]}")

                    alignment_results.append({
                        'sample_idx': idx,
                        'modality': mod_name,
                        'top_time_steps': top_time.tolist(),
                        'time_importance': time_imp.tolist(),
                        'slot_gate': gate.tolist(),
                    })

    return alignment_results


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--epochs', type=int, default=15)
    parser.add_argument('--batch_size', type=int, default=32)
    parser.add_argument('--lr', type=float, default=3e-4)
    parser.add_argument('--temperature', type=float, default=1.0)
    parser.add_argument('--dropout', type=float, default=0.3)
    args = parser.parse_args()

    set_seed(SEED)
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    print(f"使用设备: {device}")

    print("加载数据...")
    data, audio_stats, vision_stats = load_data()
    print(f"训练集: {len(data['train']['text'])} 样本")

    # 配置: 温和的slot专门化 + 合理的dropout
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
        weight_decay=1e-3,
    )
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=args.epochs)

    os.makedirs(OUTPUT_ROOT, exist_ok=True)
    os.makedirs(EXPERIMENTS_ROOT, exist_ok=True)
    print(f"\n开始训练 ({args.epochs} epochs)...")
    best_f1 = 0
    best_epoch = 0

    for epoch in range(args.epochs):
        start_time = time.time()

        train_metrics = train_epoch(
            model, data['train'], optimizer, args.batch_size, device,
            sparse_weight=0.01
        )

        valid_metrics, _, _ = evaluate(model, data['valid'], args.batch_size, device)

        scheduler.step()

        elapsed = time.time() - start_time

        print(f"Epoch {epoch+1}/{args.epochs} [{elapsed:.1f}s]")
        print(f"  Train - loss: {train_metrics['loss']:.4f}, cls: {train_metrics['cls_loss']:.4f}")
        print(f"  Valid - acc: {valid_metrics['accuracy']:.4f}, f1: {valid_metrics['f1']:.4f}, "
              f"mae: {valid_metrics['mae']:.4f}, pearson: {valid_metrics['pearson']:.4f}")

        if valid_metrics['f1'] > best_f1:
            best_f1 = valid_metrics['f1']
            best_epoch = epoch + 1
            save_path = os.path.join(
                OUTPUT_ROOT,
                f'exp3_problem1_best.pt'
            )
            torch.save(model.state_dict(), save_path)

    print(f"\n最佳F1: {best_f1:.4f} (epoch {best_epoch})")

    # 加载最佳模型
    save_path = os.path.join(
        OUTPUT_ROOT,
        f'exp3_problem1_best.pt'
    )
    if os.path.exists(save_path):
        model.load_state_dict(torch.load(save_path))

    # 时序对齐分析
    alignment_results = analyze_temporal_alignment(model, data['valid'], device)

    # 测试集最终评估
    test_metrics, all_preds, all_labels = evaluate(model, data['test'], args.batch_size, device)

    print(f"\n" + "=" * 60)
    print(f"问题1最终测试结果:")
    print(f"  Accuracy: {test_metrics['accuracy']:.4f}")
    print(f"  F1: {test_metrics['f1']:.4f}")
    print(f"  MAE: {test_metrics['mae']:.4f}")
    print(f"  Pearson: {test_metrics['pearson']:.4f}")

    # 详细分类报告
    print("\n分类报告:")
    print(classification_report(all_labels, all_preds,
                                target_names=['Negative', 'Neutral', 'Positive']))

    # 保存结果
    results_path = os.path.join(
        EXPERIMENTS_ROOT,
        'exp3_problem1_results.txt'
    )
    with open(results_path, 'w', encoding='utf-8') as f:
        f.write(f"实验: 问题1 - 基于SlotSPE的特征提取与时序对齐\n")
        f.write(f"\n模型配置:\n")
        f.write(f"  text_dim: {config['text_dim']}, audio_dim: {config['audio_dim']}, vision_dim: {config['vision_dim']}\n")
        f.write(f"  projection_dim: {config['projection_dim']}\n")
        f.write(f"  num_slots_text/audio/vision: {config['num_slots_text']}/{config['num_slots_audio']}/{config['num_slots_vision']}\n")
        f.write(f"  temperature: {args.temperature}, dropout: {args.dropout}\n")
        f.write(f"  参数量: {n_params:,}\n")
        f.write(f"\n训练过程:\n")
        f.write(f"  最佳F1: {best_f1:.4f} (epoch {best_epoch})\n")
        f.write(f"\n测试集结果:\n")
        f.write(f"  Accuracy: {test_metrics['accuracy']:.4f}\n")
        f.write(f"  F1: {test_metrics['f1']:.4f}\n")
        f.write(f"  MAE: {test_metrics['mae']:.4f}\n")
        f.write(f"  Pearson: {test_metrics['pearson']:.4f}\n")
        f.write(f"\n特征归一化:\n")
        f.write(f"  audio mean/std: {audio_stats[0]:.4f}/{audio_stats[1]:.4f}\n")
        f.write(f"  vision mean/std: {vision_stats[0]:.4f}/{vision_stats[1]:.4f}\n")

    # 保存时序对齐结果
    alignment_path = os.path.join(
        EXPERIMENTS_ROOT,
        'exp3_problem1_alignment.json'
    )
    # 转换numpy类型为python原生类型
    def convert_to_native(obj):
        if isinstance(obj, dict):
            return {k: convert_to_native(v) for k, v in obj.items()}
        elif isinstance(obj, list):
            return [convert_to_native(v) for v in obj]
        elif isinstance(obj, (np.integer, np.int32, np.int64)):
            return int(obj)
        elif isinstance(obj, (np.floating, np.float32, np.float64)):
            return float(obj)
        elif isinstance(obj, np.ndarray):
            return convert_to_native(obj.tolist())
        return obj

    alignment_results_native = convert_to_native(alignment_results)
    with open(alignment_path, 'w', encoding='utf-8') as f:
        json.dump(alignment_results_native, f, indent=2, ensure_ascii=False)

    print(f"\n结果已保存到: {results_path}")
    print(f"时序对齐结果: {alignment_path}")


if __name__ == '__main__':
    main()
