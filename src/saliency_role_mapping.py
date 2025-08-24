import json
from tqdm import tqdm
import stanza

# 한국어 모델 다운로드 (처음만 실행하면 됨)
# stanza.download('ko')

# 한국어 NLP 파이프라인 준비
nlp = stanza.Pipeline('ko', processors='tokenize,pos,lemma,depparse')

input_path = "../result/result_with_saliency.jsonl"
output_path = "../result/result_with_roles.jsonl"

def align_tokens(stanza_words, saliency_tokens):
    """
    토크나이저 결과와 saliency 토큰을 맞추는 함수.
    간단히 substring 매칭 사용.
    (실제 프로젝트에서는 더 정교한 alignment 필요할 수 있음)
    """
    aligned = []
    si = 0
    for word in stanza_words:
        # saliency 토큰 합치면서 word.text 와 매칭
        accum = ""
        tok_list = []
        while si < len(saliency_tokens) and len(accum) < len(word.text):
            tok = saliency_tokens[si]["token"].replace("▁", "")  # SentencePiece '▁' 제거
            accum += tok
            tok_list.append(saliency_tokens[si])
            si += 1
        # word.text와 align된 saliency 평균값 사용
        if tok_list:
            avg_score = sum(t["score"] for t in tok_list) / len(tok_list)
        else:
            avg_score = 0.0
        aligned.append({
            "word": word.text,
            "role": word.deprel,
            "score": avg_score
        })
    return aligned

with open(input_path, "r", encoding="utf-8") as fin, \
     open(output_path, "w", encoding="utf-8") as fout:

    for line in tqdm(fin, desc="Role tagging"):
        if not line.strip():
            continue
        d = json.loads(line)
        text = d.get("tr", None)
        saliency_list = d.get("saliency", [])
        if text is None or not saliency_list:
            continue

        # 구문 분석 실행
        doc = nlp(text)
        if not doc.sentences:
            continue

        words = doc.sentences[0].words
        aligned_roles = align_tokens(words, saliency_list)

        # 새로운 필드 추가
        d["saliency_with_roles"] = aligned_roles

        fout.write(json.dumps(d, ensure_ascii=False) + "\n")

print("완료: result_with_roles.jsonl 저장됨")

