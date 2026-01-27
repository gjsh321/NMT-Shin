import os
import json
import numpy as np
import matplotlib.pyplot as plt
from sklearn.manifold import TSNE

# 파일 경로
BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
XAI_PATH = os.path.join(BASE_DIR, "result/full_xai_results.jsonl")
TSNE_PATH = os.path.join(BASE_DIR, "result/full_xai_tsne.png")

# 데이터 로드
y_labels = []
logits_list = []
with open(XAI_PATH, "r", encoding="utf-8") as f:
    for line in f:
        if not line.strip():
            continue
        d = json.loads(line)
        # logits와 label이 모두 존재하는 경우만 사용
        if 'logits' in d and 'label' in d:
            logits = d['logits']
            label = d['label']
            if isinstance(logits, list) and len(logits) == 3:
                logits_list.append(logits)
                y_labels.append(label)

if len(logits_list) < 2:
    print("⚠️ t-SNE를 그릴 데이터가 충분하지 않습니다.")
    exit(1)

logits_np = np.array(logits_list)
y_labels = np.array(y_labels)

# t-SNE
print(f"t-SNE 실행: 샘플 {len(logits_np)}개")
perplexity = min(30, len(logits_np) // 3)
tsne = TSNE(n_components=2, random_state=42, perplexity=perplexity, max_iter=1000)
emb_2d = tsne.fit_transform(logits_np)

# 시각화
plt.figure(figsize=(12, 8))
names = ['Human', 'NMT', 'GPT']
colors = ['#1f77b4', '#ff7f0e', '#2ca02c']
for i in range(3):
    idx = y_labels == i
    if np.any(idx):
        plt.scatter(emb_2d[idx, 0], emb_2d[idx, 1], c=colors[i], label=names[i], alpha=0.6, s=50)
plt.legend(fontsize=12)
plt.title("Full XAI Results t-SNE Visualization", fontsize=14)
plt.xlabel("t-SNE 1")
plt.ylabel("t-SNE 2")
plt.tight_layout()
plt.savefig(TSNE_PATH, dpi=300, bbox_inches='tight')
print(f"✅ t-SNE 저장: {TSNE_PATH}")
