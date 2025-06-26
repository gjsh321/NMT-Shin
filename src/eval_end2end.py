import os
import json
import torch
from torch import nn
from torch.utils.data import Dataset, DataLoader
from transformers import AutoTokenizer, AutoModel
from tqdm import tqdm

# 하이퍼파라미터 및 경로
model_name = "snunlp/KR-SBERT-V40K-klueNLI-augSTS"
batch_size = 32
max_len = 256
model_save_path = "../model/kr_sbert_dnn.pt"
test_path = "../data/test_emb.jsonl"
result_path = "../result/result.jsonl"

# 데이터셋 (tr, label만 있음)
class MyDataset(Dataset):
    def __init__(self, jsonl_path):
        self.data = []
        with open(jsonl_path, "r", encoding="utf-8") as f:
            for line in f:
                if line.strip():
                    d = json.loads(line)
                    self.data.append(d)
    def __len__(self):
        return len(self.data)
    def __getitem__(self, idx):
        d = self.data[idx]
        # label은 있을 수도 있고 없을 수도 있음
        return d['tr'], d.get('label', -1)

def collate_fn(batch, tokenizer, max_len=256):
    texts, labels = zip(*batch)
    enc = tokenizer(list(texts), return_tensors="pt", padding=True, truncation=True, max_length=max_len)
    return enc, labels

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
        cls_emb = outputs.last_hidden_state[:, 0]
        x = self.dropout(cls_emb)
        return self.dnn(x).squeeze(-1)

# 토크나이저, 데이터로더
tokenizer = AutoTokenizer.from_pretrained(model_name, trust_remote_code=True)
test_dataset = MyDataset(test_path)
test_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False,
                        collate_fn=lambda b: collate_fn(b, tokenizer, max_len))

# 모델 로드
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model = SBertClassifier(model_name=model_name).to(device)
model.load_state_dict(torch.load(model_save_path, map_location=device))
model.eval()

# 평가 및 결과 파일 생성
results = []
with torch.no_grad():
    idx = 0
    with open(test_path, "r", encoding="utf-8") as fin:
        original_lines = [json.loads(line) for line in fin if line.strip()]
    for batch in tqdm(test_loader, desc="Evaluating"):
        enc, _ = batch
        for k in enc:
            enc[k] = enc[k].to(device)
        outputs = model(enc['input_ids'], enc['attention_mask'])
        preds = (outputs.cpu().numpy() > 0.5).astype(int)
        # 예측값을 원본 dict에 pred로 넣어줌
        for p, prob in zip(preds, outputs.cpu().numpy()):
            original_lines[idx]['pred'] = int(p)
            original_lines[idx]['prob'] = float(prob)   # <--- 확률값 추가!
            original_lines[idx]['embedding'] = ''
            idx += 1

   # 파일로 저장
    os.makedirs(os.path.dirname(result_path), exist_ok=True)
    with open(result_path, "w", encoding="utf-8") as fout:
        for item in original_lines:
            fout.write(json.dumps(item, ensure_ascii=False) + "\n")

print(f"Saved predictions with 'pred' field to {result_path}")

