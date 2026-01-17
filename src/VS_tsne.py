import torch
import torch.nn as nn
from transformers import AutoModel, AutoTokenizer
from torch.utils.data import DataLoader
from sklearn.manifold import TSNE
import matplotlib.pyplot as plt
import json
import os
import numpy as np
from tqdm import tqdm

# 중요: 서버에서 창을 띄우지 않고 파일로 저장하기 위한 설정
import matplotlib
matplotlib.use('Agg') 

# ---------------------------------------------------------
# 1. 모델 설정 (데이터에서 임베딩을 다시 뽑아야 함)
# ---------------------------------------------------------
class SBertClassifier(nn.Module):
    def __init__(self, model_name, dnn_hidden=128, dropout=0.3):
        super().__init__()
        self.bert = AutoModel.from_pretrained(model_name, trust_remote_code=True)
        self.dropout = nn.Dropout(dropout)
        self.dnn = nn.Sequential(
            nn.Linear(self.bert.config.hidden_size, dnn_hidden),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(dnn_hidden, 1),
            nn.Sigmoid()
        )
    def forward(self, input_ids, attention_mask):
        outputs = self.bert(input_ids=input_ids, attention_mask=attention_mask)
        cls_emb = outputs.last_hidden_state[:, 0] # 문장 전체를 대표하는 벡터
        return cls_emb

# 경로 설정
model_name = "snunlp/KR-SBERT-V40K-klueNLI-augSTS"
model_path = "../model/kr_sbert_dnn.pt"
data_path = "../result/result.jsonl" # 결과 파일 경로
save_img_path = "../result/tsne_plot.png" # 저장될 이미지 이름

# 디바이스 설정
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"사용 장치: {device}")

# ---------------------------------------------------------
# 2. 데이터 로드 및 임베딩 추출
# ---------------------------------------------------------
print("데이터 로딩 중...")
texts = []
labels = []
preds = []

if not os.path.exists(data_path):
    print(f"파일이 없습니다: {data_path}")
    exit()

with open(data_path, "r", encoding="utf-8") as f:
    for line in f:
        if line.strip():
            d = json.loads(line)
            texts.append(d['tr'])
            # 라벨이 없으면 -1, 있으면 사용
            labels.append(d.get('label', 0)) 
            preds.append(d.get('pred', 0))

# 모델 로드
tokenizer = AutoTokenizer.from_pretrained(model_name, trust_remote_code=True)
model = SBertClassifier(model_name).to(device)

if os.path.exists(model_path):
    model.load_state_dict(torch.load(model_path, map_location=device))
else:
    print("학습된 모델이 없어서 기본 모델로 진행합니다.")

model.eval()

# 임베딩 뽑기
print("임베딩 추출 중 (시간이 조금 걸립니다)...")
embeddings = []
batch_size = 32

with torch.no_grad():
    for i in tqdm(range(0, len(texts), batch_size)):
        batch_texts = texts[i:i+batch_size]
        inputs = tokenizer(batch_texts, return_tensors='pt', padding=True, truncation=True, max_length=128).to(device)
        emb = model(inputs['input_ids'], inputs['attention_mask'])
        embeddings.append(emb.cpu().numpy())

embeddings = np.concatenate(embeddings, axis=0)

# ---------------------------------------------------------
# 3. t-SNE 변환 및 시각화
# ---------------------------------------------------------
print(f"t-SNE 변환 시작 (데이터 개수: {len(embeddings)})...")
tsne = TSNE(n_components=2, random_state=42, perplexity=30)
X_embedded = tsne.fit_transform(embeddings)

print("그래프 그리는 중...")
plt.figure(figsize=(10, 8))

# 예측 결과(pred)에 따라 색깔 다르게 찍기 (0: 클래스A, 1: 클래스B)
# 실제 정답(label)으로 보고 싶으면 c=labels 로 변경하세요
scatter = plt.scatter(X_embedded[:, 0], X_embedded[:, 1], c=preds, cmap='coolwarm', alpha=0.6)

plt.colorbar(scatter, label='Prediction (0 vs 1)')
plt.title("t-SNE Visualization of Sentence Embeddings")
plt.xlabel("t-SNE component 1")
plt.ylabel("t-SNE component 2")
plt.grid(True, linestyle='--', alpha=0.5)

# 서버에서는 show() 대신 savefig() 사용!
plt.savefig(save_img_path)
print(f"완료! 이미지가 저장되었습니다: {save_img_path}")