import json
import numpy as np
from sklearn.manifold import TSNE
import matplotlib.pyplot as plt

# 1. 데이터 읽기
embeddings = []
labels = []
agents = []
with open("../data/tsne.input", "r", encoding="utf-8") as f:
    for line in f:
        if line.strip():
            d = json.loads(line)
            emb = d['embedding']
            if isinstance(emb, str):
                emb = json.loads(emb)
            embeddings.append(emb)
            labels.append(d['label'])
            agents.append(d.get('agent', ''))

embeddings = np.array(embeddings)
labels = np.array(labels)
agents = np.array(agents)

# 2. t-SNE 변환
tsne = TSNE(n_components=2, random_state=42, perplexity=30)
emb_2d = tsne.fit_transform(embeddings)

# 3. 시각화
plt.figure(figsize=(10, 8))
colors = ['tab:blue', 'tab:orange']
for label in [0, 1]:
    idx = labels == label
    plt.scatter(emb_2d[idx, 0], emb_2d[idx, 1], 
                s=30, 
                c=colors[label], 
                label=f"label {label}", 
                alpha=0.7)
    # 각 점 위에 agent 표시
    for x, y, agent in zip(emb_2d[idx, 0], emb_2d[idx, 1], agents[idx]):
        plt.text(x, y, agent, fontsize=8, ha='center', va='bottom', alpha=0.8)

plt.legend()
plt.title("t-SNE of Embeddings by Label (with agent)")
plt.xlabel("t-SNE 1")
plt.ylabel("t-SNE 2")
plt.tight_layout()
plt.show()

