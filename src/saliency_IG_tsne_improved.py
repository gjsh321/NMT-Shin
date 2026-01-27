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

# 1. 환경 설정
model_name = "klue/roberta-large"
num_classes = 3
max_len = 128
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

BASE_DIR = "/data/class/NLP2025/shh/NMT-Shin"
DATA_PATH = os.path.join(BASE_DIR, "data/final_combined_train.jsonl")

# 학습된 모델 경로 찾기 (최신 모델)
model_save_dir = os.path.join(BASE_DIR, "model")
fold_paths = []
for i in range(1, 6):
    path = os.path.join(model_save_dir, f"klue_roberta-large_fold{i}.pt")
    if os.path.exists(path):
        fold_paths.append(path)

if not fold_paths:
    print(f"⚠️ 모델 파일을 찾을 수 없습니다. {model_save_dir} 확인")
    exit()

print(f"✅ 찾은 모델: {len(fold_paths)}개 fold")

# 2. 앙상블 Wrapper 클래스 (개선된 버전)
class EnsembleWrapper(torch.nn.Module):
    def __init__(self, model_name, fold_paths, num_classes, device):
        super().__init__()
        self.models = torch.nn.ModuleList()
        self.num_classes = num_classes
        self.device = device
        
        for path in fold_paths:
            model = AutoModelForSequenceClassification.from_pretrained(
                model_name, num_labels=num_classes, trust_remote_code=True
            )
            state_dict = torch.load(path, map_location=device)
            model.load_state_dict(state_dict)
            model.to(device)
            model.eval()
            self.models.append(model)
        
        print(f"✅ {len(fold_paths)}개 모델 로드 완료")

    def forward(self, inputs_embeds, attention_mask):
        """앙상블 평균 로짓 반환"""
        ensemble_logits = torch.zeros((inputs_embeds.shape[0], self.num_classes)).to(self.device)
        
        for model in self.models:
            with torch.no_grad():
                outputs = model.roberta(inputs_embeds=inputs_embeds, attention_mask=attention_mask)
                pooled_output = outputs.pooler_output
                logits = model.classifier(pooled_output)
                ensemble_logits += logits
        
        return ensemble_logits / len(self.models)

# 3. 모델 및 분석기 준비
print("🚀 앙상블 모델 및 분석기 로딩 중...")
tokenizer = AutoTokenizer.from_pretrained(model_name, trust_remote_code=True)
ensemble_model = EnsembleWrapper(model_name, fold_paths, num_classes, device)

# Captum 분석기 설정
try:
    ig = IntegratedGradients(ensemble_model)
    saliency = Saliency(ensemble_model)
    print("✅ Captum 분석기 준비 완료")
except Exception as e:
    print(f"⚠️ Captum 분석기 초기화 실패: {e}")
    ig, saliency = None, None

# 4. 데이터 로드 및 분석
all_ensemble_logits = []
all_labels = []
samples_for_viz = []

with open(DATA_PATH, "r", encoding="utf-8") as f:
    lines = f.readlines()
    random.seed(42)
    sample_lines = random.sample(lines, min(len(lines), 1000))  # 1000개 샘플 분석

print(f"\n🔍 {len(sample_lines)}개 샘플의 앙상블 결정 공간 추출 중...")
for line in tqdm(sample_lines):
    try:
        d = json.loads(line)
        # None 체크 추가
        if d is None or d.get('tr') is None or d.get('label') is None:
            continue
        text, label = d['tr'], int(d['label'])
        
        inputs = tokenizer(text, return_tensors='pt', truncation=True, 
                          max_length=max_len, padding=True).to(device)
        input_ids = inputs['input_ids']
        attention_mask = inputs['attention_mask']
        
        # 임베딩 추출
        input_embeds = ensemble_model.models[0].roberta.embeddings(input_ids)
        
        with torch.no_grad():
            logits = ensemble_model(input_embeds, attention_mask)
            all_ensemble_logits.append(logits.cpu().numpy())
            all_labels.append(label)
        
        # 상위 3개 샘플 XAI 분석
        if len(samples_for_viz) < 3:
            try:
                # IG 계산
                attributions_ig = ig.attribute(input_embeds, target=label, 
                                              additional_forward_args=(attention_mask,),
                                              n_steps=50)
                ig_scores = attributions_ig.sum(dim=-1).squeeze(0).cpu().detach().numpy()
                
                # Saliency 계산
                attributions_sal = saliency.attribute(input_embeds, target=label,
                                                     additional_forward_args=(attention_mask,))
                sal_scores = attributions_sal.abs().sum(dim=-1).squeeze(0).cpu().detach().numpy()
                
                tokens = tokenizer.convert_ids_to_tokens(input_ids[0])
                samples_for_viz.append({
                    'text': text, 'label': label, 'tokens': tokens, 
                    'ig_scores': ig_scores, 'sal_scores': sal_scores
                })
            except Exception as e:
                print(f"  ⚠️ XAI 분석 실패: {e}")
                
    except Exception as e:
        print(f"⚠️ 샘플 처리 오류: {e}")
        continue

print(f"✅ {len(all_ensemble_logits)}개 샘플 분석 완료")

# 5. t-SNE 시각화
if len(all_ensemble_logits) > 0:
    print("\n🎨 t-SNE 시각화 생성 중...")
    logits_np = np.vstack(all_ensemble_logits)
    
    # perplexity 자동 조정
    perplexity = min(30, len(logits_np) // 3)
    tsne = TSNE(n_components=2, random_state=42, perplexity=perplexity, n_iter=1000)
    emb_2d = tsne.fit_transform(logits_np)
    
    plt.figure(figsize=(12, 8))
    names = ['Human', 'NMT', 'GPT']
    colors = ['#1f77b4', '#ff7f0e', '#2ca02c']
    
    for i in range(num_classes):
        idx = np.array(all_labels) == i
        if np.any(idx):
            plt.scatter(emb_2d[idx, 0], emb_2d[idx, 1], c=colors[i], 
                       label=names[i], alpha=0.6, s=50)
    
    plt.legend(fontsize=12)
    plt.title("Ensemble Model t-SNE Visualization (Decision Space)", fontsize=14)
    plt.xlabel("t-SNE 1")
    plt.ylabel("t-SNE 2")
    plt.tight_layout()
    
    output_path = os.path.join(BASE_DIR, "result/ensemble_tsne_improved.png")
    plt.savefig(output_path, dpi=300, bbox_inches='tight')
    print(f"✅ t-SNE 저장: {output_path}")
    plt.close()

# 6. XAI 결과 출력 및 저장
output_file = os.path.join(BASE_DIR, "result/xai_analysis_results.jsonl")
xai_results = []

print("\n" + "="*70)
print("📝 앙상블 기반 단어 중요도 분석 (XAI)")
print("="*70)

for i, sample in enumerate(samples_for_viz):
    class_name = ['Human', 'NMT', 'GPT'][sample['label']]
    print(f"\n[Sample {i+1}] 정답 클래스: {class_name}")
    print(f"문장: {sample['text'][:80]}...")
    
    result = {
        'sample_id': i,
        'label': sample['label'],
        'class_name': class_name,
        'text': sample['text'],
        'tokens': sample['tokens']
    }
    
    # IG 기준 상위 토큰
    ig_scores = np.abs(sample['ig_scores'])
    top_ig_idx = np.argsort(ig_scores)[-10:][::-1]
    
    print(f"  🔥 IG 중요도 상위 10개:")
    ig_tokens = []
    for idx in top_ig_idx:
        if idx < len(sample['tokens']):
            token = sample['tokens'][idx]
            score = sample['ig_scores'][idx]
            print(f"     {token:15s} ({score:+.4f})", end="  ")
            ig_tokens.append({'token': token, 'score': float(score)})
            if (list(top_ig_idx).index(idx) + 1) % 3 == 0:
                print()
    
    # Saliency 기준 상위 토큰
    sal_scores = sample['sal_scores']
    top_sal_idx = np.argsort(sal_scores)[-10:][::-1]
    
    print(f"\n  ⚡ Saliency 중요도 상위 10개:")
    sal_tokens = []
    for idx in top_sal_idx:
        if idx < len(sample['tokens']):
            token = sample['tokens'][idx]
            score = sample['sal_scores'][idx]
            print(f"     {token:15s} ({score:+.4f})", end="  ")
            sal_tokens.append({'token': token, 'score': float(score)})
            if (list(top_sal_idx).index(idx) + 1) % 3 == 0:
                print()
    
    result['ig_important_tokens'] = ig_tokens
    result['saliency_important_tokens'] = sal_tokens
    xai_results.append(result)

# XAI 결과 저장
with open(output_file, 'w', encoding='utf-8') as f:
    for result in xai_results:
        f.write(json.dumps(result, ensure_ascii=False) + '\n')

print(f"\n✅ XAI 분석 결과 저장: {output_file}")
print("\n" + "="*70)
print(f"📊 분석 완료: {len(all_ensemble_logits)}개 샘플, {len(samples_for_viz)}개 상세 분석")
print("="*70)
