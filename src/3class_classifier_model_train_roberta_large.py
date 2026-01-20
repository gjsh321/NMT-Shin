import os
import json
import torch
import numpy as np
from torch import nn
from torch.utils.data import Dataset, DataLoader
from transformers import AutoTokenizer, AutoModelForSequenceClassification, get_linear_schedule_with_warmup
from tqdm import tqdm
from sklearn.metrics import classification_report, accuracy_score
from torch.cuda.amp import GradScaler, autocast
from torch.optim import AdamW
from collections import Counter

#하이퍼 파라미터 설정
model_name = "klue/roberta-large"
batch_size = 32
lr = 2e-5
epochs = 10
max_len = 256
warmup_ratio = 0.1
early_stop_patience = 3
dropout = 0.3  # 기본 설정
num_classes = 3
save_dir = "./model/roberta_large/"
os.makedirs(save_dir, exist_ok=True)
model_save_path = os.path.join(save_dir, "best_model.pt")

#데이터셋 3개 클래스로 라벨링
class MyDataset(Dataset):
    def __init__(self, jsonl_path):
        self.data = []
        if not os.path.exists(jsonl_path):
            print(f"⚠️ 파일 없음: {jsonl_path}")
            return
            
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

# --- 데이터 준비 ---
tokenizer = AutoTokenizer.from_pretrained(model_name, trust_remote_code=True)
train_dataset = MyDataset("./data/train_merged.jsonl") 
test_dataset = MyDataset("./data/test_merged.jsonl")

if len(train_dataset) == 0:
    print("❌ 학습 데이터가 로드되지 않았습니다.")
else:
    # Train/Val split (8:2)
    train_size = int(0.8 * len(train_dataset))
    val_size = len(train_dataset) - train_size
    train_split, val_split = torch.utils.data.random_split(train_dataset, [train_size, val_size])
    
    train_loader = DataLoader(train_split, batch_size=batch_size, shuffle=True,
                            collate_fn=lambda b: collate_fn(b, tokenizer, max_len))
    val_loader = DataLoader(val_split, batch_size=batch_size, shuffle=False,
                            collate_fn=lambda b: collate_fn(b, tokenizer, max_len))
    test_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False,
                            collate_fn=lambda b: collate_fn(b, tokenizer, max_len))

    # 학습 준비
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = AutoModelForSequenceClassification.from_pretrained(model_name, num_labels=num_classes, trust_remote_code=True).to(device)
    
    # Dropout 설정 (기본값)
    model.config.hidden_dropout_prob = dropout
    model.config.attention_probs_dropout_prob = dropout
    
    optimizer = AdamW(model.parameters(), lr=lr, weight_decay=0.01)
    
    # 클래스 가중치 계산
    train_labels = [train_dataset.data[i][1] for i in range(len(train_dataset))]
    label_counts = Counter(train_labels)
    total_samples = len(train_labels)
    class_weights = torch.tensor([total_samples / (num_classes * label_counts.get(i, 1)) for i in range(num_classes)], dtype=torch.float).to(device)
    print(f"Class weights: {class_weights}")
    
    # Learning Rate Scheduler
    total_steps = len(train_loader) * epochs
    warmup_steps = int(total_steps * warmup_ratio)
    scheduler = get_linear_schedule_with_warmup(optimizer, num_warmup_steps=warmup_steps, num_training_steps=total_steps)
    
    scaler = GradScaler() 
    
    best_val_accuracy = 0
    patience_counter = 0
    
    print(f"🚀 RoBERTa-Large 모델 학습 시작")
    print(f"Model: {model_name} | LR: {lr} | Batch: {batch_size} | Epochs: {epochs}\n")
    
    # 학습
    for epoch in range(epochs):
        model.train()
        total_loss = 0
        
        for batch in tqdm(train_loader, desc=f"Epoch {epoch+1}/{epochs}"):
            enc, labels = batch
            
            for k in enc:
                enc[k] = enc[k].to(device)
            labels = labels.to(device).long()
            
            optimizer.zero_grad()
            
            with autocast():
                outputs = model(input_ids=enc['input_ids'], attention_mask=enc['attention_mask'], labels=labels)
                loss = outputs.loss
            
            scaler.scale(loss).backward()
            scaler.step(optimizer)
            scaler.update()
            scheduler.step()
            
            total_loss += loss.item() * labels.size(0)
        
        print(f"Epoch {epoch+1} loss: {total_loss / len(train_split):.4f}", end=" | ")
        
        # Validation
        model.eval()
        val_preds, val_labels = [], []
        with torch.no_grad():
            for batch in tqdm(val_loader, desc="Validation", leave=False):
                enc, labels = batch
                for k in enc:
                    enc[k] = enc[k].to(device)
                outputs = model(input_ids=enc['input_ids'], attention_mask=enc['attention_mask'])
                preds = torch.argmax(outputs.logits, dim=1).cpu().numpy()
                val_preds.extend(preds)
                val_labels.extend(labels.numpy())
        
        val_accuracy = accuracy_score(val_labels, val_preds)
        print(f"Val Accuracy: {val_accuracy:.4f}")
        
        # Early Stopping
        if val_accuracy > best_val_accuracy:
            best_val_accuracy = val_accuracy
            patience_counter = 0
            torch.save(model.state_dict(), model_save_path)
            print(f"✅ 최고 성능 모델 저장 (Accuracy: {best_val_accuracy:.4f})")
        else:
            patience_counter += 1
            
            if patience_counter >= early_stop_patience:
                print(f"🛑 Early Stopping! Best Val Accuracy: {best_val_accuracy:.4f}\n")
                break
    
    # Test 평가
    print("="*60)
    print("📊 Test Set 평가")
    print("="*60)
    
    model.load_state_dict(torch.load(model_save_path))
    model.eval()
    test_preds, test_labels = [], []
    with torch.no_grad():
        for batch in tqdm(test_loader, desc="Test"):
            enc, labels = batch
            for k in enc:
                enc[k] = enc[k].to(device)
            outputs = model(input_ids=enc['input_ids'], attention_mask=enc['attention_mask'])
            preds = torch.argmax(outputs.logits, dim=1).cpu().numpy()
            test_preds.extend(preds)
            test_labels.extend(labels.numpy())
    
    test_acc = accuracy_score(test_labels, test_preds)
    print(f"\n🏆 Test Accuracy: {test_acc:.4f}")
    print("\n최종 분류 결과:")
    print(classification_report(test_labels, test_preds, digits=4, target_names=['Human', 'NMT', 'GPT']))
    
    print(f"\n✅ RoBERTa-Large 모델 학습 완료")
    print(f"모델 저장 위치: {model_save_path}")
