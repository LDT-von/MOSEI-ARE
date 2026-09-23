"""
查看label-100的数据细节
"""
import pandas as pd
import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from src.config import ATTACHMENT1_PATH
sys.stdout.reconfigure(encoding='utf-8')

df = pd.read_excel(Path(ATTACHMENT1_PATH) / 'label-100.xlsx')

# 看几个样本的text和annotation
print('示例样本:')
for i in range(3):
    row = df.iloc[i]
    print(f'\n样本{i}:')
    print(f'  video_id: {row["video_id"]}, clip_id: {row["clip_id"]}')
    print(f'  text: {row["text"][:200]}')
    print(f'  label: {row["label"]}')
    print(f'  annotation: {row["annotation"]}')

# 看数据分布
print('\nannotation分布:')
print(df['annotation'].value_counts())

# 看看label和annotation的关系
print('\n按annotation看label均值:')
print(df.groupby('annotation')['label'].agg(['mean', 'count']))
