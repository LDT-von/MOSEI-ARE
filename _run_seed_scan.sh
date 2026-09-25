#!/bin/bash
# 多种子扫描脚本：跑 Arm C 的多个种子，按 valid selection_score 排序
set -e
cd /data1/E题/MOSEI-ARE
export MOSEI_DATA_ROOT="/data1/E题/E题数据/E题数据"

SEEDS=${1:-"1 7 100 2024"}
for seed in $SEEDS; do
    out="problem2/outputs/_scan_C_seed_${seed}"
    if [ -f "$out/valid_evaluation.json" ]; then
        echo "[skip] seed=$seed 已有 valid_evaluation"
        continue
    fi
    echo "=========== seed=$seed 开始 $(date +%H:%M:%S) ==========="
    rm -rf "$out"
    python3 -m problem2.scripts.run train --arm C --seed $seed --device cpu --output "$out" 2>&1 | tail -8
    echo "[done] seed=$seed $(date +%H:%M:%S)"
done

echo "============ 汇总 ============"
python3 -c "
import json, glob
rows = []
for d in sorted(glob.glob('problem2/outputs/_scan_C_seed_*')):
    try:
        with open(d+'/history.json') as f: h = json.load(f)
        with open(d+'/valid_evaluation.json') as f: v = json.load(f)
        # 取 history 最后一行的 best selection_score
        best_idx = max(range(len(h)), key=lambda i: h[i].get('selection_score', h[i].get('valid_acc', -1)))
        seed = d.split('_')[-1]
        rows.append((
            seed,
            h[best_idx].get('selection_score', h[best_idx].get('valid_acc', 0)),
            h[best_idx].get('epoch', best_idx),
            v.get('clean',{}).get('accuracy',0),
            v.get('clean',{}).get('f1_macro',0),
        ))
    except Exception as e:
        print(f'{d}: ERROR {e}')
rows.sort(key=lambda r: -r[1])
print(f'{\"seed\":<8} {\"sel_score\":<10} {\"best_ep\":<8} {\"clean_acc\":<10} {\"clean_f1\":<10}')
for r in rows: print(f'{r[0]:<8} {r[1]:<10.4f} {r[2]:<8} {r[3]:<10.4f} {r[4]:<10.4f}')
# 加原 arm_C_seed_42 作对比
try:
    with open('problem2/outputs/arm_C_seed_42/history.json') as f: h = json.load(f)
    with open('problem2/outputs/arm_C_seed_42/valid_evaluation.json') as f: v = json.load(f)
    best_idx = max(range(len(h)), key=lambda i: h[i].get('selection_score', h[i].get('valid_acc', -1)))
    print()
    print(f'[ORIG] arm_C_seed_42  sel={h[best_idx].get(\"selection_score\",0):.4f}  ep={best_idx}  acc={v[\"clean\"][\"accuracy\"]:.4f}  f1={v[\"clean\"][\"f1_macro\"]:.4f}')
except: pass
"
