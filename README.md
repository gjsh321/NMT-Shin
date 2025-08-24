## 1. DNN 실행 방법
 python dnn_end2end.py
 
## 2. XGBoost 실행 방법
 python xgb_emb.py

## 3. Saliency 실행 방법
 python saliency_end2end.py

## 4. t-SNE 실행 방법
 python tsne_emb.py

## Data
데이터는 jsonl 형태로 되어있다.
- stype : sentence와 paragraph이 있다.
- agent : M1, M2 와 같이 M으로 시작하는 것은 사람을, gpt 등으로 시작하는 것은 기계번역을 의미한다.
- tr : 번역문
- doc : orginal 문서 이름
- embedding : 번역문을 embedding으로 만든결과이다. ( embedding으로 바꾸기 위해 여러가지 방법을 쓸 수 있다)
