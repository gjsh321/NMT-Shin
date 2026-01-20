import os
import json
import torch
import numpy as np
from torch import nn
from torch.utils.data import Dataset, DataLoader
from transformers import AutoTokenizer, AutoModelForSequenceClassification, get_linear_schedule_with_warmup
from tqdm import tqdm
from sklearn.metrics import classification_report, accuracy_score
from sklearn.model_selection import KFold
from torch.cuda.amp import GradScaler, autocast
from torch.optim import AdamW
from collections import Counter
from scipy import stats

#하이퍼 파라미터 설정
model_name = "klue/roberta-base"
batch_size = 16
lr = 1e-5
epochs = 5
max_len = 128
warmup_ratio = 0.1
early_stop_patience = 3
n_splits = 5  # K-fold 분할 수
dropout = 0.4  # Dropout 비율
num_classes = 3
save_dir = "./model/"
os.makedirs(save_dir, exist_ok=True)
model_save_path = os.path.join(save_dir, model_name.replace("/", "_") + ".pt")

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
    test_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False,
                            collate_fn=lambda b: collate_fn(b, tokenizer, max_len))

    # K-Fold 설정
    kfold = KFold(n_splits=n_splits, shuffle=True, random_state=42)
    fold_results = []
    
    print(f"🔄 {n_splits}-Fold 교차 검증 시작\n")
    
    all_fold_preds = []
    all_fold_labels = []
    
    for fold, (train_idx, val_idx) in enumerate(kfold.split(train_dataset), 1):
        print(f"\n{'='*60}")
        print(f"Fold {fold}/{n_splits}")
        print(f"{'='*60}")
        
        # Train/Val split
        train_split = torch.utils.data.Subset(train_dataset, train_idx)
        val_split = torch.utils.data.Subset(train_dataset, val_idx)
        
        train_loader = DataLoader(train_split, batch_size=batch_size, shuffle=True,
                                collate_fn=lambda b: collate_fn(b, tokenizer, max_len))
        val_loader = DataLoader(val_split, batch_size=batch_size, shuffle=False,
                                collate_fn=lambda b: collate_fn(b, tokenizer, max_len))
        
        # 학습 준비
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        model = AutoModelForSequenceClassification.from_pretrained(model_name, num_labels=num_classes, trust_remote_code=True).to(device)
        
        # Dropout 설정 (과적합 방지)
        model.config.hidden_dropout_prob = dropout
        model.config.attention_probs_dropout_prob = dropout
        
        optimizer = AdamW(model.parameters(), lr=lr, weight_decay=0.01)
        
        # 클래스 가중치 계산
        train_labels = [train_dataset.data[i][1] for i in train_idx]
        label_counts = Counter(train_labels)
        total_samples = len(train_labels)
        class_weights = torch.tensor([total_samples / (num_classes * label_counts.get(i, 1)) for i in range(num_classes)], dtype=torch.float).to(device)
        
        # Learning Rate Scheduler
        total_steps = len(train_loader) * epochs
        warmup_steps = int(total_steps * warmup_ratio)
        scheduler = get_linear_schedule_with_warmup(optimizer, num_warmup_steps=warmup_steps, num_training_steps=total_steps)
        
        scaler = GradScaler() 
        
        best_val_accuracy = 0
        patience_counter = 0
        
        print(f"Train samples: {len(train_idx)}, Val samples: {len(val_idx)}\n")
        
        # 각 fold 학습
        for epoch in range(epochs):
            model.train()
            total_loss = 0
            
            for batch in tqdm(train_loader, desc=f"Epoch {epoch+1}/{epochs}", leave=False):
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
                fold_model_path = model_save_path.replace(".pt", f"_fold{fold}.pt")
                torch.save(model.state_dict(), fold_model_path)
            else:
                patience_counter += 1
                
                if patience_counter >= early_stop_patience:
                    print(f"Early Stopping! Best Val Accuracy: {best_val_accuracy:.4f}")
                    break
        
        fold_results.append(best_val_accuracy)
        print(f"Fold {fold} Best Validation Accuracy: {best_val_accuracy:.4f}")
        
        # Test 평가
        fold_model_path = model_save_path.replace(".pt", f"_fold{fold}.pt")
        model.load_state_dict(torch.load(fold_model_path))
        
        model.eval()
        fold_preds, fold_labels = [], []
        with torch.no_grad():
            for batch in tqdm(test_loader, desc="Test", leave=False):
                enc, labels = batch
                for k in enc:
                    enc[k] = enc[k].to(device)
                outputs = model(input_ids=enc['input_ids'], attention_mask=enc['attention_mask'])
                preds = torch.argmax(outputs.logits, dim=1).cpu().numpy()
                fold_preds.extend(preds)
                fold_labels.extend(labels.numpy())
        
        all_fold_preds.append(np.array(fold_preds))
        all_fold_labels.append(np.array(fold_labels))
        
        fold_acc = accuracy_score(fold_labels, fold_preds)
        print(f"Fold {fold} Test Accuracy: {fold_acc:.4f}")
        print(f"\nFold {fold} 분류 결과:")
        print(classification_report(fold_labels, fold_preds, digits=4, target_names=['Human', 'NMT', 'GPT']))
    
    # 최종 결과
    print(f"\n{'='*60}")
    print("📊 K-Fold 교차 검증 최종 결과")
    print(f"{'='*60}")
    print(f"Validation Accuracy (각 fold):")
    for i, acc in enumerate(fold_results, 1):
        print(f"  Fold {i}: {acc:.4f}")
    print(f"  평균: {np.mean(fold_results):.4f} ± {np.std(fold_results):.4f}\n")
    
    # Test 앙상블: 모든 fold predictions 투표
    ensemble_preds = []
    for i in range(len(all_fold_labels[0])):
        # 각 샘플에 대해 5개 fold의 예측값을 모아서 다수결
        votes = [all_fold_preds[fold][i] for fold in range(n_splits)]
        ensemble_pred = np.bincount(votes).argmax()
        ensemble_preds.append(ensemble_pred)
    
    ensemble_preds = np.array(ensemble_preds)
    
    test_acc = accuracy_score(all_fold_labels[0], ensemble_preds)
    print(f"\n🏆 Ensemble Test Accuracy: {test_acc:.4f}")
    print("\n최종 앙상블 분류 결과:")
    print(classification_report(all_fold_labels[0], ensemble_preds, digits=4, target_names=['Human', 'NMT', 'GPT']))
    
    print(f"\n✅ K-Fold 검증 완료")
