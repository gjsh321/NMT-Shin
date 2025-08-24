import torch
import torch.nn as nn
from transformers import AutoModel, AutoTokenizer
from captum.attr import Saliency
import json
from tqdm import tqdm

# SBertClassifier 정의
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
        outputs = self.bert(inputs_embeds=input_embeds, attention_mask=attention_mask)
        cls_emb = outputs.last_hidden_state[:, 0]
        x = self.dropout(cls_emb)
        return self.dnn(x).squeeze(-1)

# 모델/토크나이저 준비
model_name = "snunlp/KR-SBERT-V40K-klueNLI-augSTS"
model_save_path = "../model/kr_sbert_dnn.pt"
max_len = 256
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
tokenizer = AutoTokenizer.from_pretrained(model_name, trust_remote_code=True)
model = SBertClassifier(model_name=model_name).to(device)
model.load_state_dict(torch.load(model_save_path, map_location=device))
model.eval()

# 임베딩 레이어
embedding_layer = model.bert.embeddings.word_embeddings

# Saliency 객체
def forward_func(input_embeds, attention_mask):
    return model(input_embeds, attention_mask)
saliency = Saliency(forward_func)

# 입력/출력 파일
input_path = "../result/result.jsonl"
output_path = "../result/result_with_saliency.jsonl"

with open(input_path, "r", encoding="utf-8") as fin, \
     open(output_path, "w", encoding="utf-8") as fout:

    for line in tqdm(fin, desc="Processing"):
        if not line.strip():
            continue
        d = json.loads(line)
        sample = d.get("tr", None)
        if sample is None:
            continue

        # 토큰화
        inputs = tokenizer(sample, return_tensors='pt', truncation=True, max_length=max_len)
        for k in inputs:
            inputs[k] = inputs[k].to(device)

        # 임베딩 준비
        input_embeds = embedding_layer(inputs['input_ids'])
        input_embeds.requires_grad_()

        # Saliency 계산
        attr = saliency.attribute(
            input_embeds,
            additional_forward_args=(inputs['attention_mask'],),
            abs=True
        )
        attr = attr.abs().sum(dim=-1).squeeze(0).detach().cpu().numpy()

        # 토큰과 매핑 (리스트 형태)
        tokens = tokenizer.convert_ids_to_tokens(inputs['input_ids'][0])
        saliency_list = [{"token": tok, "score": float(score)} for tok, score in zip(tokens, attr)]

        # saliency 추가
        d["saliency"] = saliency_list

        # 저장
        fout.write(json.dumps(d, ensure_ascii=False) + "\n")

print("result_with_saliency.jsonl 저장됨.")

