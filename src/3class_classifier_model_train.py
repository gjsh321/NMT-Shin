import os
import json
import torch
from torch import nn
from torch.utils.data import Dataset, DataLoader
from transformers import AutoTokenizer, AutoModelForSequenceClassification, get_linear_schedule_with_warmup
from tqdm import tqdm
from sklearn.metrics import classification_report, accuracy_score
from sklearn.model_selection import train_test_split
from torch.optim import AdamW

# 1. 하이퍼 파라미터 최적화 (단순화)
model_name = "klue/roberta-large"
batch_size = 16  # Gradient Accumulation 없이 배치 사이즈 확대 (메모리 허용 시)
lr = 2e-5
epochs = 5      # 데이터가 많으면 에폭은 줄여도 충분합니다
max_len = 128
save_dir = "./model/"
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# 2. 데이터셋 클래스 (동일)
class MyDataset(Dataset):
    def __init__(self, jsonl_path):
        self.data = []
        with open(jsonl_path, "r", encoding="utf-8") as f:
            for line in f:
                if line.strip():
                    d = json.loads(line)
                    self.data.append((d['tr'], int(d['label'])))
    def __len__(self): return len(self.data)
    def __getitem__(self, idx): return self.data[idx]

def collate_fn(batch, tokenizer, max_len=128):
    texts, labels = zip(*batch)
    enc = tokenizer(list(texts), return_tensors="pt", padding=True, truncation=True, max_length=max_len)
    return enc, torch.tensor(labels)

# --- 데이터 준비 ---
tokenizer = AutoTokenizer.from_pretrained(model_name)
full_dataset = MyDataset("./data/final_combined_train.jsonl")

# 3. K-Fold 대신 단순 Train/Val/Test 분할 (80/10/10)
train_idx, temp_idx = train_test_split(range(len(full_dataset)), test_size=0.2, random_state=42)
val_idx, test_idx = train_test_split(temp_idx, test_size=0.5, random_state=42)

train_loader = DataLoader(torch.utils.data.Subset(full_dataset, train_idx), batch_size=batch_size, shuffle=True, collate_fn=lambda b: collate_fn(b, tokenizer, max_len))
val_loader = DataLoader(torch.utils.data.Subset(full_dataset, val_idx), batch_size=batch_size, shuffle=False, collate_fn=lambda b: collate_fn(b, tokenizer, max_len))
test_loader = DataLoader(torch.utils.data.Subset(full_dataset, test_idx), batch_size=batch_size, shuffle=False, collate_fn=lambda b: collate_fn(b, tokenizer, max_len))

# 4. 모델 및 학습 설정 (Focal Loss -> 표준 CrossEntropy)
model = AutoModelForSequenceClassification.from_pretrained(model_name, num_labels=3).to(device)
optimizer = AdamW(model.parameters(), lr=lr, weight_decay=0.01)
loss_fn = nn.CrossEntropyLoss() # 데이터가 충분하면 기본 Loss가 가장 강력합니다.

total_steps = len(train_loader) * epochs
scheduler = get_linear_schedule_with_warmup(optimizer, num_warmup_steps=int(total_steps * 0.1), num_training_steps=total_steps)

# 5. 학습 루프 (단일 실행)
best_acc = 0
for epoch in range(epochs):
    model.train()
    for batch in tqdm(train_loader, desc=f"Epoch {epoch+1}"):
        enc, labels = batch
        inputs = {k: v.to(device) for k, v in enc.items()}
        labels = labels.to(device)
        
        outputs = model(**inputs)
        loss = loss_fn(outputs.logits, labels)
        
        loss.backward()
        optimizer.step()
        scheduler.step()
        optimizer.zero_grad()

    # Validation
    model.eval()
    preds, targets = [], []
    with torch.no_grad():
        for batch in val_loader:
            enc, labels = batch
            outputs = model(**{k: v.to(device) for k, v in enc.items()})
            preds.extend(torch.argmax(outputs.logits, dim=1).cpu().numpy())
            targets.extend(labels.numpy())
    
    val_acc = accuracy_score(targets, preds)
    print(f"Epoch {epoch+1} Val Acc: {val_acc:.4f}")
    
    if val_acc > best_acc:
        best_acc = val_acc
        torch.save(model.state_dict(), os.path.join(save_dir, "simple_model.pt"))

# 6. 최종 Test 평가
print("\n🏆 최종 Test 결과:")
model.load_state_dict(torch.load(os.path.join(save_dir, "simple_model.pt"), weights_only=True))
model.eval()
test_preds, test_targets = [], []
with torch.no_grad():
    for batch in test_loader:
        enc, labels = batch
        outputs = model(**{k: v.to(device) for k, v in enc.items()})
        test_preds.extend(torch.argmax(outputs.logits, dim=1).cpu().numpy())
        test_targets.extend(labels.numpy())

print(classification_report(test_targets, test_preds, target_names=['Human', 'NMT', 'GPT'], digits=4))