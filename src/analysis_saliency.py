import torch
import torch.nn as nn
from transformers import AutoModel, AutoTokenizer
from captum.attr import Saliency
import json
import os
from tqdm import tqdm

# 1. 모델 설정 (학습 때랑 똑같이!)
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
    def forward(self, input_embeds, attention_mask):
        # Captum 분석을 위해 input_ids 대신 input_embeds를 받도록 수정된 forward
        outputs = self.bert(inputs_embeds=input_embeds, attention_mask=attention_mask)
        cls_emb = outputs.last_hidden_state[:, 0]
        x = self.dropout(cls_emb)
        return self.dnn(x).squeeze(-1)

# 2. 경로 및 모델 준비
model_name = "snunlp/KR-SBERT-V40K-klueNLI-augSTS"
model_save_path = "../model/kr_sbert_dnn.pt"  # 저장된 모델 경로 확인
input_path = "../result/result.jsonl"         # 방금 만든 결과 파일
output_path = "../result/result_with_saliency.jsonl" # 새로 만들 파일

max_len = 256
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

tokenizer = AutoTokenizer.from_pretrained(model_name, trust_remote_code=True)
model = SBertClassifier(model_name=model_name).to(device)

if os.path.exists(model_save_path):
    model.load_state_dict(torch.load(model_save_path, map_location=device))
    print(f"모델 로드 완료: {model_save_path}")
else:
    print(f"오류: 모델 파일이 없습니다 ({model_save_path})")
    exit()

model.eval()

# 3. 분석 도구(Saliency) 준비
embedding_layer = model.bert.embeddings.word_embeddings

def forward_func(input_embeds, attention_mask):
    return model(input_embeds, attention_mask)

saliency = Saliency(forward_func)

# 4. 파일 한 줄씩 읽어서 중요도 분석
print(f"분석 시작: {input_path} -> {output_path}")

with open(input_path, "r", encoding="utf-8") as fin, \
     open(output_path, "w", encoding="utf-8") as fout:

    # 라인 수 계산 (tqdm 바를 위해)
    lines = fin.readlines()
    
    for line in tqdm(lines, desc="Processing Saliency"):
        if not line.strip():
            continue
        d = json.loads(line)
        sample = d.get("tr", None)
        
        # 텍스트가 없거나 너무 짧으면 스킵
        if not sample:
            continue

        # (1) 토큰화
        inputs = tokenizer(sample, return_tensors='pt', truncation=True, max_length=max_len)
        for k in inputs:
            inputs[k] = inputs[k].to(device)

        # (2) 임베딩 추출 (Gradient 계산을 위해 필요)
        input_embeds = embedding_layer(inputs['input_ids'])
        input_embeds.requires_grad_()

        # (3) Saliency(기여도) 계산
        # 문장의 어떤 단어가 결과에 영향을 줬는지 계산
        attr = saliency.attribute(
            input_embeds,
            additional_forward_args=(inputs['attention_mask'],),
            abs=True
        )
        # 모든 차원을 합쳐서 단어별 점수 하나로 만듦
        attr = attr.abs().sum(dim=-1).squeeze(0).detach().cpu().numpy()

        # (4) 결과 저장
        tokens = tokenizer.convert_ids_to_tokens(inputs['input_ids'][0])
        
        # 보기 좋게 토큰:점수 쌍으로 저장
        saliency_list = [{"token": tok, "score": float(score)} for tok, score in zip(tokens, attr)]
        d["saliency"] = saliency_list

        fout.write(json.dumps(d, ensure_ascii=False) + "\n")

print("분석 끝! result_with_saliency.jsonl 파일이 생성되었습니다.")