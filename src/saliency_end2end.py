import torch
import torch.nn as nn
from transformers import AutoModel, AutoTokenizer
from captum.attr import Saliency
import json
import matplotlib.pyplot as plt

# SBertClassifier 정의 (학습 때와 똑같이!)
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

# 모델/토크나이저
model_name = "snunlp/KR-SBERT-V40K-klueNLI-augSTS"
model_save_path = "../model/kr_sbert_dnn.pt"
max_len = 256
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
tokenizer = AutoTokenizer.from_pretrained(model_name, trust_remote_code=True)
model = SBertClassifier(model_name=model_name).to(device)
model.load_state_dict(torch.load(model_save_path, map_location=device))
model.eval()

# test.jsonl에서 한 샘플
with open("../result/sel.candi", "r", encoding="utf-8") as f:
    for line in f:
        if line.strip():
            d = json.loads(line)
            sample = d['tr']
            break

inputs = tokenizer(sample, return_tensors='pt', truncation=True, max_length=max_len)
for k in inputs:
    inputs[k] = inputs[k].to(device)

# 임베딩 준비
embedding_layer = model.bert.embeddings.word_embeddings
input_embeds = embedding_layer(inputs['input_ids'])
input_embeds.requires_grad_()

def forward_func(input_embeds, attention_mask):
    return model(input_embeds, attention_mask)

saliency = Saliency(forward_func)
attr = saliency.attribute(input_embeds, additional_forward_args=(inputs['attention_mask'],), abs=True)
attr = attr.abs().sum(dim=-1).squeeze(0).detach().cpu().numpy()  # (seq_len,)

tokens = tokenizer.convert_ids_to_tokens(inputs['input_ids'][0])

print("분석할 입력 문장:", sample)
print("토큰별 중요도:")
for token, score in zip(tokens, attr):
    print(f"{token:>10}: {score:.4f}")

# 한글 폰트 설정 (우분투)
plt.rc('font', family='NanumGothic')
plt.figure(figsize=(12,2))
plt.bar(range(len(tokens)), attr)
plt.xticks(range(len(tokens)), tokens, rotation=45, ha='right')
plt.title("Saliency: 입력 토큰별 중요도")
plt.tight_layout()
plt.show()

