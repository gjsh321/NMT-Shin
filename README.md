🏆 Ensemble Test Accuracy: 0.7393

최종 앙상블 분류 결과:
              precision    recall  f1-score   support

       Human     0.2414    0.3500    0.2857        40
         NMT     0.8784    0.7222    0.7927       360
         GPT     0.6796    0.8750    0.7650       160

    accuracy                         0.7393       560
   macro avg     0.5998    0.6491    0.6145       560
weighted avg     0.7761    0.7393    0.7486       560

✅ WeightedRandomSampler

배치 구성 시 클래스별 확률 조정
매 배치마다 Human/GPT가 고르게 등장
✅ Focal Loss gamma 강화 (2.0 → 3.0)

어려운 샘플에 더 높은 페널티
희귀 클래스 학습력 증가
