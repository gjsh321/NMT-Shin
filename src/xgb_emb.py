import pandas as pd
import numpy as np
import json
from xgboost import XGBClassifier
from sklearn.metrics import classification_report, accuracy_score

def load_jsonl(filename):
    """jsonl 파일을 DataFrame으로 읽어오기"""
    data = []
    with open(filename, 'r', encoding='utf-8') as f:
        for line in f:
            if line.strip():
                data.append(json.loads(line))
    return pd.DataFrame(data)

# 1. 데이터 읽기
train_df = load_jsonl("../data/train_emb.jsonl")
test_df = load_jsonl("../data/test_emb.jsonl")

# 2. 임베딩과 라벨 추출 (임베딩이 None이 아닌 것만)
train_valid = train_df[train_df["embedding"].notnull()].reset_index(drop=True)
test_valid = test_df[test_df["embedding"].notnull()].reset_index(drop=True)

X_train = np.array(train_valid["embedding"].tolist())
y_train = train_valid["label"].astype(int).values
X_test = np.array(test_valid["embedding"].tolist())
y_test = test_valid["label"].astype(int).values

# 3. XGBoost 분류기 학습
clf = XGBClassifier(
    n_estimators=100,
    max_depth=5,
    learning_rate=0.1,
    random_state=42,
    eval_metric='logloss'
)
clf.fit(X_train, y_train)

# 4. 예측 및 평가
y_pred = clf.predict(X_test)
print("정확도:", accuracy_score(y_test, y_pred))
print("분류 리포트:\n", classification_report(y_test, y_pred, digits=4))

