import json
import pandas as pd
from collections import defaultdict

# 데이터 로드
data = []
with open('/data/class/NLP2025/shh/NMT-Shin/result/ensemble_xai_analysis.jsonl', 'r', encoding='utf-8') as f:
    for line in f:
        data.append(json.loads(line))

# 클래스별 토큰 점수 합산용
stats = defaultdict(lambda: defaultdict(list))

for entry in data:
    label = entry['true_label']
    for analysis in entry['analysis']:
        token = analysis['token']
        score = analysis['score']
        stats[label][token].append(score)

# 통계 요약 (평균 점수 기준 상위 5개)
class_names = {0: "Human", 1: "NMT", 2: "GPT"}
for label, tokens in stats.items():
    print(f"\n📊 [{class_names[label]}] 클래스 통계 분석")
    # 토큰별 평균 점수 계산 및 정렬
    avg_scores = {t: sum(s)/len(s) for t, s in tokens.items()}
    sorted_tokens = sorted(avg_scores.items(), key=lambda x: x[1], reverse=True)[:5]
    
    for i, (t, s) in enumerate(sorted_tokens):
        print(f"  {i+1}. {t:<10} | 평균 IG 점수: {s:.4f} | 빈도: {len(tokens[t])}")