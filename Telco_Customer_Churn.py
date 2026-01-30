# 1. 라이브러리 불러오기
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.ensemble import GradientBoostingClassifier
from sklearn.metrics import classification_report, confusion_matrix
import shap
import matplotlib.pyplot as plt
import seaborn as sns

# 2. 데이터 불러오기 및 전처리
df = pd.read_csv("WA_Fn-UseC_-Telco-Customer-Churn.csv")
df.columns = df.columns.str.strip()
df['TotalCharges'] = pd.to_numeric(df['TotalCharges'], errors='coerce')
df.dropna(inplace=True)

# 이진 변수로 변환
df['Churn'] = df['Churn'].map({'No': 0, 'Yes': 1})

# 원-핫 인코딩 (범주형 변수 처리)
df_encoded = pd.get_dummies(df.drop('customerID', axis=1), drop_first=True)

# 3. 데이터 분할
X = df_encoded.drop('Churn', axis=1)
y = df_encoded['Churn']

X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.3, stratify=y, random_state=42
)

# 4. Gradient Boosting 기본 모델 학습
gb_model = GradientBoostingClassifier(random_state=42)
gb_model.fit(X_train, y_train)

# 5. 성능 평가
print("=== 기본 모델 성능 (Test Set) ===")
y_pred = gb_model.predict(X_test)
print(classification_report(y_test, y_pred))

# 6. 하이퍼파라미터 튜닝
params = {
    'n_estimators': [100, 200],
    'learning_rate': [0.05, 0.1, 0.2],
    'max_depth': [3, 4, 5],
    'subsample': [0.8, 1.0],
    'min_samples_split': [2, 5],
    'min_samples_leaf': [1, 3]
}

grid = GridSearchCV(
    GradientBoostingClassifier(random_state=42),
    param_grid=params,
    scoring='f1',
    cv=5,
    n_jobs=-1,
    verbose=1
)

grid.fit(X_train, y_train)

print("최적 하이퍼파라미터:", grid.best_params_)


# 7. 최적 모델 성능 확인 (수정: 테스트셋으로 재학습하면 안 됩니다!)
best_model = grid.best_estimator_

# 8. SHAP 해석
# SHAP Explainer 생성
explainer = shap.TreeExplainer(best_model) # Tree 기반 모델이므로 TreeExplainer가 더 정확합니다.
shap_values = explainer.shap_values(X_test)

# SHAP summary plot 생성 및 저장
# show=False를 꼭 넣어줘야 도화지가 비워지지 않습니다!
shap.summary_plot(shap_values, X_test, show=False)

# 저장 전 여백 조정 (글자 잘림 방지)
plt.tight_layout()
plt.savefig("shap_summary.png", dpi=300, bbox_inches='tight')
print("✅ SHAP 그래프가 'shap_summary.png'로 저장되었습니다.")
plt.close() # 메모리 정리

# 9. 성능 평가 (Train/Test 분리 유지)
y_train_pred = best_model.predict(X_train)
print("=== 최적 모델 성능 (Train Set) ===")
print(classification_report(y_train, y_train_pred))

y_test_pred = best_model.predict(X_test)
print("=== 최적 모델 성능 (Test Set) ===")
print(classification_report(y_test, y_test_pred))