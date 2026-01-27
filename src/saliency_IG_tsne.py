import os
import json
import torch
import numpy as np
import matplotlib.pyplot as plt
from transformers import AutoTokenizer, AutoModelForSequenceClassification
from sklearn.manifold import TSNE
from tqdm import tqdm
from captum.attr import IntegratedGradients, Saliency

# 1. 환경 설정
model_name = "klue/roberta-large"
num_classes = 3
max_len = 128
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

BASE_DIR = "/data/class/NLP2025/shh/NMT-Shin"
DATA_PATH = os.path.join(BASE_DIR, "data/final_combined_train.jsonl")
fold_paths = [os.path.join(BASE_DIR, "model", f"klue_roberta-large_fold{i}.pt") for i in range(1, 6)]

# 2. 앙상블 Wrapper 클래스 정의 (Captum 분석용)
class EnsembleWrapper(torch.nn.Module):
    def __init__(self, model_name, fold_paths, num_classes, device):
        super().__init__()
        self.models = torch.nn.ModuleList()
        for path in fold_paths:
            model = AutoModelForSequenceClassification.from_pretrained(model_name, num_labels=num_classes)
            model.load_state_dict(torch.load(path, map_location=device))
            model.to(device)
            model.eval()
            self.models.append(model)
        self.device = device

    def forward(self, inputs_embeds, attention_mask):
        # 모든 모델의 로짓을 합산하여 평균 계산
        ensemble_logits = torch.zeros((inputs_embeds.shape[0], num_classes)).to(self.device)
        for model in self.models:
            # Roberta 구조에 맞게 forward 호출
            outputs = model.roberta(inputs_embeds=inputs_embeds, attention_mask=attention_mask)
            pooled_output = model.classifier(outputs[0])
            ensemble_logits += pooled_output
        return ensemble_logits / len(self.models)

# 3. 모델 및 분석기 준비
print("🚀 앙상블 모델 및 분석기 로딩 중...")
tokenizer = AutoTokenizer.from_pretrained(model_name)
ensemble_model = EnsembleWrapper(model_name, fold_paths, num_classes, device)

# Captum 분석기 설정
ig = IntegratedGradients(ensemble_model)
saliency = Saliency(ensemble_model)

# 4. 데이터 로드 및 앙상블 로짓 추출 (t-SNE용)
all_ensemble_logits = []
all_labels = []
samples_for_viz = []

with open(DATA_PATH, "r", encoding="utf-8") as f:
    lines = f.readlines()
    import random
    random.seed(42)
    sample_lines = random.sample(lines, min(len(lines), 500)) # 분석 속도를 위해 500개 조절

print("🔍 앙상블 결정 공간 추출 및 XAI 분석 시작...")
for line in tqdm(sample_lines):
    d = json.loads(line)
    text, label = d['tr'], int(d['label'])
    
    inputs = tokenizer(text, return_tensors='pt', truncation=True, max_length=max_len).to(device)
    input_ids = inputs['input_ids']
    attention_mask = inputs['attention_mask']
    
    # 임베딩 추출 (IG/Saliency 입력용)
    # 앙상블의 첫 번째 모델을 기준으로 임베딩 층 참조
    input_embeds = ensemble_model.models[0].roberta.embeddings.word_embeddings(input_ids)
    
    with torch.no_grad():
        logits = ensemble_model(input_embeds, attention_mask)
        all_ensemble_logits.append(logits.cpu().numpy())
        all_labels.append(label)
    
    # 상위 몇 개 샘플만 텍스트 분석용으로 저장
    if len(samples_for_viz) < 3:
        # IG 계산
        attributions_ig = ig.attribute(input_embeds, target=label, additional_forward_args=(attention_mask,))
        ig_scores = attributions_ig.sum(dim=-1).squeeze(0).cpu().detach().numpy()
        
        # Saliency 계산
        attributions_slc = saliency.attribute(input_embeds, target=label, additional_forward_args=(attention_mask,))
        slc_scores = attributions_slc.abs().sum(dim=-1).squeeze(0).cpu().detach().numpy()
        
        tokens = tokenizer.convert_ids_to_tokens(input_ids[0])
        samples_for_viz.append({
            'text': text, 'label': label, 'tokens': tokens, 
            'ig_scores': ig_scores, 'slc_scores': slc_scores
        })

# 5. t-SNE 시각화
print("🎨 t-SNE 시각화 생성 중...")
logits_np = np.vstack(all_ensemble_logits)
tsne = TSNE(n_components=2, random_state=42, perplexity=30)
emb_2d = tsne.fit_transform(logits_np)

plt.figure(figsize=(10, 7))
names = ['Human', 'NMT', 'GPT']
colors = ['#1f77b4', '#ff7f0e', '#2ca02c']
for i in range(3):
    idx = np.array(all_labels) == i
    plt.scatter(emb_2d[idx, 0], emb_2d[idx, 1], c=colors[i], label=names[i], alpha=0.5)
plt.legend()
plt.title("Ensemble Model t-SNE (84% Accuracy Space)")
plt.savefig(os.path.join(BASE_DIR, "result/ensemble_tsne.png"))

# 6. XAI 결과 출력
print("\n" + "="*50)
print("📝 앙상블 기반 단어 중요도 분석 (XAI)")
print("="*50)
for i, sample in enumerate(samples_for_viz):
    print(f"\n[Sample {i+1}] 정답 클래스: {names[sample['label']]}")
    print(f"문장: {sample['text'][:60]}...")
    
    # IG 기준 상위 5개 토큰
    top_ig = np.argsort(np.abs(sample['ig_scores']))[-5:][::-1]
    print(f"  🔥 IG 중요 토큰: ", end="")
    for idx in top_ig:
        print(f"{sample['tokens'][idx]}({sample['ig_scores'][idx]:.3f}) ", end="")
    
    # Saliency 기준 상위 5개 토큰
    top_slc = np.argsort(sample['slc_scores'])[-5:][::-1]
    print(f"\n  ⚡ Saliency 중요 토큰: ", end="")
    for idx in top_slc:
        print(f"{sample['tokens'][idx]}({sample['slc_scores'][idx]:.3f}) ", end="")
    print()