광운그룹 하계인턴 가상회사 프로그램 : 통신사 고객 이탈 예측 및 다각적 유지 전략 수립
본 프로젝트는 "광운그룹 하계인턴 가상회사 프로그램" 과정 중 수행되었으며, 머신러닝 기반의 이탈 예측과 XAI(Explainable AI)를 결합하여 비즈니스 의사결정을 지원하는 데이터 파이프 라인을 구축 했습니다.

1. 프로젝트 배경 및 목적 (Context)
- 소속: 광운그룹 하계인턴 가상회사 프로그램 (KW - Corporation)
- 목표: 통신 서비스 고객 데이터를 분석하여 이탈 고위험군을 사전에 식별하고, 이탈에 영향을 미치는 핵심 동인을 파악하여 고객 관리 비용 효율화를 위한 전략 도출

2. 주요 분석 프로세스 (Technical Workflow)
- 데이터 전처리 및 탐색 (Cleaning & EDA)
  > 결측치 처리: TotalCharges 등 수치형 데이터의 공백 문자 처리 및 타입 변화 최적화
  > 특성 공학: 범주형 변수의 인코딩 및 학습용 데이터셋 구성 (Label Encoding, One-Hot Encoding)

<img width="1168" height="768" alt="Monthly Charges Distribution by Churn" src="https://github.com/user-attachments/assets/5f96fb3e-8cc4-4921-926c-a08b1fffbc7f" />


- 모델링 및 성능 최적화 (Modeling & Evaluation)
  > 다중 모델 비교: XGBoost, Gradient Boosting, Random Forest, ANN, SVM, KNN, Logistic Regression 등 7종 모델의 성능 비교 분석 수행
  > 최적화: GridSearchCV를 활용한 하이퍼파라미터 튜닝으로 모델의 일반화 성능 확보
  > 평가 지표: Accuracy 뿐만 아니라 비즈니스 관점에서 중요한 Recall(재현율) 과 ROC-AUC를 중점적으로 평가

- 모델 해석 및 인사이트 도출 (XAI & PDP)
  > SHAP (Summary & Force Plot): 각 변수가 개별 고객의 이탈 예측에 기여하는 정도를 시각화하여 '블랙박스' 모델의 해석력 확보
  > PDP (Partial Dependence Plots): 가입 기간(tenure) 및 월 요금(MonthlyCharges) 변화에 따른 이탈 확률의 변화 추이를 분석하여 비선형적 관계 규명

3. 분석 결과 요약 (Key Insights)
- 계약 형태의 영향: 'Month-to-Month' 계약 형태가 이탈의 가장 강력한 지표임을 확인, 장기 계약 전환 프로모션의 필요성 입증
- 요금 임계점 발견: 월 평균 요금이 특정 구간 이상일 때 이탈 확률이 금증하는 경향을 PDP 분석으로 포착
- 기술 지원의 중요성: 온라인 보안 및 기술 지원 서비스 미사용 고객군에서 이탈 징후가 뚜렷하게 나타남

4. 저장소 구성 (File Structure)
- Test_Result.py: 7종 머신러닝 모델 성능 비교 및 ROC 커브 시각화
- Churn_Result.py: XGBoost 기반 SHAP 분석 및 PDP 시각화 스크립트
- Telco_Customer_Churn.py: 데이터 전처리 및 하이퍼파라미터 튜닝 (GridSearch CV)
- Telco_Churn_Visualization.py: EDA 및 주요 변수별 이탈 분포 시각화
- Telco_Customer_Churn.R: R을 활용한 기초 통계 및 데이터 탐색

5. 기대 효과 및 비즈니스 활용
- 전략적 의사결정: 데이터에 근거한 타겟 마케팅으로 마케팅 비용 효율 개선
- 고객 유지율 제고: 이탈 징후 조기 포착을 통한 선제적 대응 체계 마련으로 이탈률 5.0%p 저감 기대

