#!/usr/bin/env python3
import os
import json
import traceback
import torch
import numpy as np
import matplotlib.pyplot as plt
from transformers import AutoTokenizer, AutoModelForSequenceClassification
from sklearn.manifold import TSNE
from tqdm import tqdm
from captum.attr import IntegratedGradients, Saliency

# ----------------------------
# Config
# ----------------------------
model_name = "klue/roberta-large"
num_classes = 3
max_len = 128

BASE_DIR = "/data/class/NLP2025/shh/NMT-Shin"
DATA_PATH = os.path.join(BASE_DIR, "data/final_combined_train.jsonl")
MODEL_PATH = os.path.join(BASE_DIR, "model/simple_model.pt")
OUT_DIR = os.path.join(BASE_DIR, "result")
OUTPUT_FILE = os.path.join(OUT_DIR, "full_xai_results.jsonl")
TSNE_PNG = os.path.join(OUT_DIR, "full_dataset_tsne.png")

os.makedirs(OUT_DIR, exist_ok=True)

# If you want to test fast, set SAMPLE_LIMIT to a small integer (e.g., 100).
# Set to None to run all samples (be cautious: IG is slow).
SAMPLE_LIMIT = None  # e.g., 200 for quick tests

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"🚀 Device: {device}")

# 결과 파일명 지정
RESULT_PATH = os.path.join(BASE_DIR, "result/full_xai_results.jsonl")

# 기존 파일의 라인 수를 셉니다 (예: 9900줄이 저장되어 있음)
existing_lines = 0
if os.path.exists(RESULT_PATH):
    with open(RESULT_PATH, "r", encoding="utf-8") as f:
        existing_lines = sum(1 for _ in f)
print(f"이미 처리된 {existing_lines}개를 건너뜁니다.")

# ----------------------------
# Helpers
# ----------------------------
def get_embeddings_for_input(model, input_ids):
    """
    Safely get token embeddings from various model attribute patterns.
    Returns embeddings tensor with same device as input_ids.
    """
    # Preferred: model.roberta.embeddings or similar
    if hasattr(model, "roberta") and hasattr(model.roberta, "embeddings"):
        emb_layer = model.roberta.embeddings
        return emb_layer(input_ids)
    # BART/other: base_model.embeddings
    if hasattr(model, "base_model") and hasattr(model.base_model, "embeddings"):
        emb_layer = model.base_model.embeddings
        return emb_layer(input_ids)
    # Fallback: get_input_embeddings()
    emb = model.get_input_embeddings()
    return emb(input_ids)

def forward_with_embeds(inputs_embeds, attention_mask):
    """
    Forward wrapper for Captum: take inputs_embeds and attention_mask,
    return logits tensor (batch, num_labels).
    """
    out = model(inputs_embeds=inputs_embeds, attention_mask=attention_mask)
    if hasattr(out, "logits"):
        return out.logits
    if isinstance(out, tuple) or isinstance(out, list):
        return out[0]
    raise RuntimeError("Unexpected model output type")

# ----------------------------
# Load model & tokenizer
# ----------------------------
tokenizer = AutoTokenizer.from_pretrained(model_name)
model = AutoModelForSequenceClassification.from_pretrained(model_name, num_labels=num_classes).to(device)

try:
    state = torch.load(MODEL_PATH, map_location=device)
    # load with strict=False to allow size/extra key differences; adjust if you want strict load
    model.load_state_dict(state, strict=False)
    print("✅ 모델 가중치 로드 완료")
except Exception as e:
    print(f"⚠️ 모델 로드 경고: {e}")
    traceback.print_exc()

model.eval()

# Initialize Captum
ig = IntegratedGradients(forward_with_embeds)
saliency = Saliency(forward_with_embeds)

# ----------------------------
# Load all lines (JSONL)
# ----------------------------
print("📂 전체 데이터 로딩 중...")
all_lines = []
with open(DATA_PATH, "r", encoding="utf-8") as f:
    for line in f:
        line = line.strip()
        if not line:
            continue
        all_lines.append(line)

total_samples = len(all_lines)
print(f"🔍 총 {total_samples}개 전체 데이터 분석 시작 (실시간 저장 모드)")
print(f"💾 결과 저장 경로: {OUTPUT_FILE}")

# ----------------------------
# XAI loop: process samples and write JSONL incrementally
# ----------------------------
all_logits_for_tsne = []
all_labels_for_tsne = []

# Write results line-by-line
with open(OUTPUT_FILE, "w", encoding="utf-8") as f_out:
    # iterate with optional SAMPLE_LIMIT
    iterator = enumerate(all_lines) if SAMPLE_LIMIT is None else enumerate(all_lines[:SAMPLE_LIMIT])
    for i, line in tqdm(iterator, total=(SAMPLE_LIMIT or total_samples), desc="XAI loop"):
        try:
            d = json.loads(line)
            # basic checks
            if d.get("tr") is None or d.get("label") is None:
                # skip incomplete records
                continue

            text = d["tr"]
            label = int(d["label"])

            # Tokenize -> move tensors to device (do NOT call .to on the dict directly)
            inputs = tokenizer(text, return_tensors="pt", truncation=True, max_length=max_len, padding="max_length")
            inputs = {k: v.to(device) for k, v in inputs.items()}

            input_ids = inputs["input_ids"]            # (1, seq_len)
            attention_mask = inputs["attention_mask"]  # (1, seq_len)

            # Get input embeddings in a robust manner
            input_embeds = get_embeddings_for_input(model, input_ids)
            input_embeds = input_embeds.to(device)
            # For attribution, inputs must require grad
            input_embeds.requires_grad_(True)

            # 1) Inference & collect logits (for t-SNE)
            with torch.no_grad():
                logits = forward_with_embeds(input_embeds, attention_mask)
                all_logits_for_tsne.append(logits.cpu().numpy())
                all_labels_for_tsne.append(label)

            # 2) Prepare baseline for IG (zero baseline is common)
            baseline = torch.zeros_like(input_embeds).to(device)

            # 3) Integrated Gradients
            # Note: n_steps can be lowered for speed (e.g., 20)
            attr_ig = ig.attribute(inputs=input_embeds,
                                   baselines=baseline,
                                   target=label,
                                   additional_forward_args=(attention_mask,),
                                   n_steps=25)
            if isinstance(attr_ig, tuple) or isinstance(attr_ig, list):
                attr_ig = attr_ig[0]
            ig_scores = attr_ig.sum(dim=-1).squeeze(0).cpu().detach().numpy()  # (seq_len,)

            # 4) Saliency
            attr_sal = saliency.attribute(inputs=input_embeds,
                                          target=label,
                                          additional_forward_args=(attention_mask,))
            if isinstance(attr_sal, tuple) or isinstance(attr_sal, list):
                attr_sal = attr_sal[0]
            sal_scores = attr_sal.abs().sum(dim=-1).squeeze(0).cpu().detach().numpy()  # (seq_len,)

            # 5) Tokens and masking (remove/zero padding positions)
            ids_list = input_ids.squeeze(0).cpu().tolist()
            tokens = tokenizer.convert_ids_to_tokens(ids_list)
            mask = attention_mask.squeeze(0).cpu().numpy().tolist()  # 0/1 list

            # Apply mask: set padded positions to 0.0
            ig_scores = [float(s) if m == 1 else 0.0 for s, m in zip(ig_scores.tolist() if hasattr(ig_scores, "tolist") else ig_scores, mask)]
            sal_scores = [float(s) if m == 1 else 0.0 for s, m in zip(sal_scores.tolist() if hasattr(sal_scores, "tolist") else sal_scores, mask)]

            try:
                sep_index = tokens.index("[SEP]") + 1
            except ValueError:
                sep_index = len(tokens)

# 2. 데이터 자르기 (Slicing) - 용량 최적화 핵심!
            trimmed_tokens = tokens[:sep_index]
            trimmed_ig = ig_scores[:sep_index]
            trimmed_sal = sal_scores[:sep_index]

# 3. 최적화된 데이터로 저장할 딕셔너리 생성
            result_line = {
                "tokens": trimmed_tokens,
                "ig_scores": trimmed_ig,
                "sal_scores": trimmed_sal,
                "label": label
            }
# 4. 파일에 쓰기 (한 줄씩 저장)
            f_out.write(json.dumps(result_line, ensure_ascii=False) + "\n")
            f_out.flush()

        except Exception as e:
            # Detailed logging for debugging; continue to next sample
            print(f"⚠️ XAI error at sample idx {i}: {e}")
            traceback.print_exc()
            # Optionally, write a small failure record (uncomment if desired)
            # f_out.write(json.dumps({"error_idx": i, "error": str(e)}, ensure_ascii=False) + "\n")
            continue

print("\n✅ 전체 XAI 데이터 처리 루프 완료")

# ----------------------------
# t-SNE visualization
# ----------------------------
if len(all_logits_for_tsne) > 0:
    print("\n🎨 t-SNE 시각화 생성 중...")
    logits_np = np.vstack(all_logits_for_tsne)  # (N, num_classes)
    # determine perplexity conservatively
    perplexity = 30
    if len(logits_np) < 50:
        perplexity = max(5, len(logits_np) // 5)
    elif len(logits_np) > 1000:
        perplexity = min(50, len(logits_np) // 100)

    tsne_kwargs = {"n_components": 2, "random_state": 42, "perplexity": perplexity}
    try:
        tsne = TSNE(**tsne_kwargs, n_iter=1000)
    except TypeError:
        tsne = TSNE(**tsne_kwargs, max_iter=1000)

    emb_2d = tsne.fit_transform(logits_np)

    plt.figure(figsize=(12, 8))
    names = ["Human", "NMT", "GPT"]
    colors = ["#1f77b4", "#ff7f0e", "#2ca02c"]

    labels_arr = np.array(all_labels_for_tsne)
    for i_class in range(num_classes):
        idx = labels_arr == i_class
        if np.any(idx):
            plt.scatter(emb_2d[idx, 0], emb_2d[idx, 1], c=colors[i_class], label=names[i_class], alpha=0.3, s=5)

    plt.legend(markerscale=5)
    plt.title(f"Full Dataset Decision Space ({len(logits_np)} samples)", fontsize=14)
    plt.savefig(TSNE_PNG, dpi=300, bbox_inches="tight")
    print(f"🖼️ t-SNE 이미지 저장 완료: {TSNE_PNG}")
    plt.close()
else:
    print("ℹ️ t-SNE를 위해 수집된 logits 데이터가 없습니다. XAI 루프에서 오류로 스킵된 항목이 많은지 확인하세요.")