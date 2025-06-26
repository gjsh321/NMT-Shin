import os
import json
import torch
from torch import nn
from torch.utils.data import Dataset, DataLoader
from transformers import AutoTokenizer, AutoModel
from tqdm import tqdm
from sklearn.metrics import classification_report, accuracy_score

# 하이퍼파라미터
model_name = "snunlp/KR-SBERT-V40K-klueNLI-augSTS"  # 한국어 SBERT!
batch_size = 32
lr = 2e-5
epochs = 10
max_len = 256
save_dir = "../model/"
os.makedirs(save_dir, exist_ok=True)
model_save_path = os.path.join(save_dir, "kr_sbert_dnn.pt")

# 데이터셋
class MyDataset(Dataset):
    def __init__(self, jsonl_path):
        self.data = []
        with open(jsonl_path, "r", encoding="utf-8") as f:
            for line in f:
                if line.strip():
                    d = json.loads(line)
                    if d.get('tr') and d.get('label') is not None:
                        self.data.append((d['tr'], int(d['label'])))
    def __len__(self):
        return len(self.data)
    def __getitem__(self, idx):
        return self.data[idx]

def collate_fn(batch, tokenizer, max_len=64):
    texts, labels = zip(*batch)
    enc = tokenizer(list(texts), return_tensors="pt", padding=True, truncation=True, max_length=max_len)
    return enc, torch.tensor(labels)

# SBERT + DNN 분류기
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

# 토크나이저/데이터로더
tokenizer = AutoTokenizer.from_pretrained(model_name, trust_remote_code=True)
train_dataset = MyDataset("../data/train_emb.jsonl")
test_dataset = MyDataset("../data/test_emb.jsonl")
train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True,
                         collate_fn=lambda b: collate_fn(b, tokenizer, max_len))
test_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False,
                        collate_fn=lambda b: collate_fn(b, tokenizer, max_len))

# 모델 준비
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model = SBertClassifier(model_name=model_name).to(device)
criterion = nn.BCELoss()
#optimizer = torch.optim.AdamW(model.parameters(), lr=lr)
optimizer = torch.optim.Adam(model.parameters(), lr=lr, weight_decay=1e-5)


# 학습
for epoch in range(epochs):
    model.train()
    total_loss = 0
    for batch in tqdm(train_loader, desc=f"Epoch {epoch+1}"):
        enc, labels = batch
        for k in enc:
            enc[k] = enc[k].to(device)
        labels = labels.to(device).float()
        optimizer.zero_grad()
        outputs = model(enc['input_ids'], enc['attention_mask'])
        loss = criterion(outputs, labels)
        loss.backward()
        optimizer.step()
        total_loss += loss.item() * labels.size(0)
    print(f"Epoch {epoch+1} loss: {total_loss / len(train_dataset):.4f}")

# 평가
model.eval()
all_preds, all_labels = [], []
with torch.no_grad():
    for batch in tqdm(test_loader, desc="Test"):
        enc, labels = batch
        for k in enc:
            enc[k] = enc[k].to(device)
        outputs = model(enc['input_ids'], enc['attention_mask'])
        preds = (outputs.cpu().numpy() > 0.5).astype(int)
        all_preds.extend(preds)
        all_labels.extend(labels.numpy())
print("정확도:", accuracy_score(all_labels, all_preds))
print(classification_report(all_labels, all_preds, digits=4))

# 모델 저장
torch.save(model.state_dict(), model_save_path)
print(f"모델 저장 완료: {model_save_path}")

