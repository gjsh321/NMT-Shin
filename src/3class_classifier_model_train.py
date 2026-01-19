import os
import json
import torch
from torch import nn
from torch.utils.data import Dataset, DataLoader
from transformers import AutoTokenizer, AutoModelForSequenceClassification
from tqdm import tqdm
from sklearn.metrics import classification_report, accuracy_score
from torch.cuda.amp import GradScaler, autocast
from torch.optim import AdamW
#하이퍼 파라미터 설정
model_name = "klue/roberta-base"
batch_size = 32
lr = 2e-5
epochs = 10
max_len = 256
num_classes = 3  # [변경] 클래스 개수 3개 (0:Human, 1:NMT, 2:GPT)
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
                    # label이 0, 1, 2 중 하나여야 함
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

#모델 아키텍처 변경

# --- 데이터 준비 (경로 주의!) ---
# 교수님 코드는 하나의 파일을 읽는 구조이므로, 미리 파일이 합쳐져 있어야 합니다.
# 만약 파일이 분리되어 있다면 아래 '데이터 병합 팁'을 참고하세요.
tokenizer = AutoTokenizer.from_pretrained(model_name, trust_remote_code=True)
train_dataset = MyDataset("./data/train_merged.jsonl") 
test_dataset = MyDataset("./data/test_merged.jsonl")

if len(train_dataset) == 0:
    print("❌ 학습 데이터가 로드되지 않았습니다. 경로를 확인해주세요.")
else:
    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True,
                            collate_fn=lambda b: collate_fn(b, tokenizer, max_len))
    test_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False,
                            collate_fn=lambda b: collate_fn(b, tokenizer, max_len))

    #학습 준비
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = AutoModelForSequenceClassification.from_pretrained(model_name, num_labels=num_classes, trust_remote_code=True).to(device)

    # AutoModelForSequenceClassification은 내부적으로 CrossEntropyLoss 처리 
    optimizer = AdamW(model.parameters(), lr=2e-5, weight_decay=0.01)
    #grad 16비트 설정
    scaler = GradScaler() 
    print(f"🚀 학습 시작 (Device: {device}) - FP16 Mode ON")

#학습
for epoch in range(epochs):
    model.train()
    total_loss = 0
    
    for batch in tqdm(train_loader, desc=f"Epoch {epoch+1}"):
        enc, labels = batch
        
        # 데이터 GPU로 이동
        for k in enc:
            enc[k] = enc[k].to(device)
        labels = labels.to(device).long()
        
        optimizer.zero_grad()
        
        # [핵심 수정] autocast로 감싸서 FP16 연산 수행
        # enabled=True, dtype=torch.float16을 명시하여 FP16 강제 사용
        with autocast():
            outputs = model(input_ids=enc['input_ids'], attention_mask=enc['attention_mask'], labels=labels)
            loss = outputs.loss
        
        # [핵심 수정] Backpropagation 과정도 Scaler를 통해 수행
        scaler.scale(loss).backward()
        scaler.step(optimizer)
        scaler.update()
        
        total_loss += loss.item() * labels.size(0)
    
    print(f"Epoch {epoch+1} loss: {total_loss / len(train_dataset):.4f}")

    #평가 로직 변경 (마지막 epoch에만 결과 출력)
    if epoch == epochs - 1:
        model.eval()
        all_preds, all_labels = [], []
        with torch.no_grad():
            for batch in tqdm(test_loader, desc="Test"):
                enc, labels = batch
                for k in enc:
                    enc[k] = enc[k].to(device)
                
                outputs = model(input_ids=enc['input_ids'], attention_mask=enc['attention_mask'])
                
                # [변경] 0.5보다 크다가 아니라, 가장 높은 점수를 가진 클래스 선택 (argmax)
                preds = torch.argmax(outputs.logits, dim=1).cpu().numpy()
                all_preds.extend(preds)
                all_labels.extend(labels.numpy())

        print("\n🏆 최종 결과")
        print("정확도:", accuracy_score(all_labels, all_preds))
        # target_names 매핑: 0=Human, 1=NMT, 2=GPT
        print(classification_report(all_labels, all_preds, digits=4, target_names=['Human', 'NMT', 'GPT']))

        # 모델 저장
        torch.save(model.state_dict(), model_save_path)
        print(f"모델 저장 완료: {model_save_path}")