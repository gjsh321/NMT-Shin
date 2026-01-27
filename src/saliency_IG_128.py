import os
import json
import torch
import numpy as np
import matplotlib.pyplot as plt
from transformers import AutoTokenizer, AutoModelForSequenceClassification
from sklearn.manifold import TSNE
from tqdm import tqdm
from captum.attr import IntegratedGradients, Saliency
import random
from collections import defaultdict

# 1. 환경 설정
model_name = "klue/roberta-large"
num_classes = 3
max_len = 128
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

BASE_DIR = "/data/class/NLP2025/shh/NMT-Shin"
DATA_PATH = os.path.join(BASE_DIR, "data/final_combined_train.jsonl")
MODEL_PATH = os.path.join(BASE_DIR, "model/simple_model.pt")

# 2. 모델 및 토크나이저 로드
print(f"🚀 Device: {device}")
print("🚀 모델 및 분석기 로딩 중...")
tokenizer = AutoTokenizer.from_pretrained(model_name)
model = AutoModelForSequenceClassification.from_pretrained(model_name, num_labels=num_classes).to(device)

# 학습된 가중치 로드
try:
    model.load_state_dict(torch.load(MODEL_PATH, map_location=device), strict=False)
except Exception as e:
    print(f"⚠️ 모델 로드 경고 (무시 가능): {e}")
model.eval()

# Captum 분석기 설정
ig = IntegratedGradients(model)
saliency = Saliency(model)

# 3. 데이터 로드 및 "층화 추출 (Stratified Sampling)"
# 목표: 각 클래스별 650개씩 -> 총 1950개
SAMPLES_PER_CLASS = 650
data_by_label = defaultdict(list)

print("📂 데이터 로드 및 클래스별 분류 중...")
with open(DATA_PATH, "r", encoding="utf-8") as f:
    for line in f:
        try:
            d = json.loads(line)
            if d.get('tr') and d.get('label') is not None:
                label = int(d['label'])
                data_by_label[label].append(line)
        except:
            continue

# 각 클래스에서 샘플링
sample_lines = []
for label in [0, 1, 2]:
    lines = data_by_label[label]
    count = min(len(lines), SAMPLES_PER_CLASS)
    selected = random.sample(lines, count)
    sample_lines.extend(selected)
    print(f"  👉 Class {label}: {len(lines)}개 중 {count}개 추출 완료")

# 랜덤하게 섞어서 편향 방지
random.seed(42)
random.shuffle(sample_lines)

print(f"\n🔍 총 {len(sample_lines)}개 샘플(클래스 균형)에 대해 XAI 분석 시작...")

# 4. 분석 루프
all_logits = []
all_labels = []
samples_for_viz = []

for line in tqdm(sample_lines):
    try:
        d = json.loads(line)
        text, label = d['tr'], int(d['label'])
        
        # 입력 처리
        inputs = tokenizer(text, return_tensors='pt', truncation=True, max_length=max_len, padding='max_length').to(device)
        input_ids = inputs['input_ids']
        attention_mask = inputs['attention_mask']
        input_embeds = model.roberta.embeddings(input_ids)
        
        # Logit 추출
        with torch.no_grad():
            logits = model(inputs_embeds=input_embeds, attention_mask=attention_mask).logits
            all_logits.append(logits.cpu().numpy())
            all_labels.append(label)
        
        # --- XAI 분석 (IG & Saliency) ---
        # IG 계산 (Tuple 반환 대응)
        attr_ig = ig.attribute(input_embeds, target=label, additional_forward_args=(attention_mask,), n_steps=50)
        if isinstance(attr_ig, tuple): attr_ig = attr_ig[0]
        ig_scores = attr_ig.sum(dim=-1).squeeze(0).cpu().detach().numpy()
        
        # Saliency 계산 (Tuple 반환 대응)
        attr_sal = saliency.attribute(input_embeds, target=label, additional_forward_args=(attention_mask,))
        if isinstance(attr_sal, tuple): attr_sal = attr_sal[0]
        sal_scores = attr_sal.abs().sum(dim=-1).squeeze(0).cpu().detach().numpy()
        
        tokens = tokenizer.convert_ids_to_tokens(input_ids[0])
        samples_for_viz.append({
            'text': text, 'label': label, 'tokens': tokens, 
            'ig_scores': ig_scores.tolist(), # JSON 저장을 위해 리스트 변환
            'sal_scores': sal_scores.tolist()
        })

    except Exception as e:
        # print(f"⚠️ 에러 발생: {e}") 
        continue

print(f"✅ 분석 완료: {len(samples_for_viz)}개 결과 생성")

# 5. t-SNE 시각화 (버전 호환성 적용)
if len(all_logits) > 0:
    print("\n🎨 t-SNE 시각화 생성 중...")
    logits_np = np.vstack(all_logits)
    perplexity = min(30, len(logits_np) // 3)
    
    # TSNE 인자 에러 방지 (n_iter vs max_iter)
    tsne_kwargs = {'n_components': 2, 'random_state': 42, 'perplexity': perplexity}
    try:
        tsne = TSNE(**tsne_kwargs, n_iter=1000)
    except TypeError:
        tsne = TSNE(**tsne_kwargs, max_iter=1000)
        
    emb_2d = tsne.fit_transform(logits_np)
    
    plt.figure(figsize=(12, 8))
    names = ['Human', 'NMT', 'GPT']
    colors = ['#1f77b4', '#ff7f0e', '#2ca02c'] # 파랑, 주황, 초록
    
    for i in range(num_classes):
        idx = np.array(all_labels) == i
        if np.any(idx):
            plt.scatter(emb_2d[idx, 0], emb_2d[idx, 1], c=colors[i], label=names[i], alpha=0.6, s=20)
            
    plt.legend(fontsize=12)
    plt.title(f"Decision Space t-SNE (Sampled: {len(logits_np)})", fontsize=14)
    output_png = os.path.join(BASE_DIR, "result/stratified_tsne.png")
    plt.savefig(output_png, dpi=300, bbox_inches='tight')
    print(f"🖼️ t-SNE 저장 완료: {output_png}")
    plt.close()

# 6. 결과 저장
output_jsonl = os.path.join(BASE_DIR, "result/stratified_xai_results.jsonl")
with open(output_jsonl, 'w', encoding='utf-8') as f:
    for sample in samples_for_viz:
        f.write(json.dumps(sample, ensure_ascii=False) + '\n')

print(f"💾 최종 데이터 저장 완료: {output_jsonl}")