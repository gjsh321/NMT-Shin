🏆 Ensemble Test Accuracy: 0.6464

최종 앙상블 분류 결과:
              precision    recall  f1-score   support

       Human     0.1441    0.4000    0.2119        40
         NMT     0.8987    0.5667    0.6951       360
         GPT     0.6396    0.8875    0.7435       160

    accuracy                         0.6464       560
   macro avg     0.5608    0.6181    0.5501       560
weighted avg     0.7708    0.6464    0.6744       560

오버샘플링 및 ✅ Focal Loss 구현: 어려운 샘플에 더 높은 가중치 → Human 클래스 학습 개선
✅ Class Weight 통합: 오버샘플링과 Focal Loss 병행
✅ Test 데이터 보존: 원본 분포 유지로 공정한 평가
