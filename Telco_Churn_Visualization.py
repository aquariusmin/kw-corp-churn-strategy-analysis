import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import shap
import xgboost as xgb
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder

# 1. 데이터 불러오기 및 전처리
df = pd.read_csv("WA_Fn-UseC_-Telco-Customer-Churn.csv")
df = df.replace(" ", np.nan).dropna()
df.drop(columns=['customerID'], inplace=True)

# 2. 범주형 변수 인코딩
for col in df.select_dtypes('object').columns:
    df[col] = LabelEncoder().fit_transform(df[col])

# 3. X, y 분리 및 데이터 분할
X = df.drop('Churn', axis=1)
y = df['Churn']
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=42)

# 4. XGBoost 모델 학습
model = xgb.XGBClassifier(use_label_encoder=False, eval_metric='logloss')
model.fit(X_train, y_train)

# 5. SHAP explainer 정의
explainer = shap.Explainer(model, X_train)
shap_values = explainer(X_test)

# 6. summary plot (bar)
plt.figure(figsize=(12, 6))
shap.plots.bar(shap_values, max_display=12, show=False)
plt.title("SHAP Summary - Bar")
plt.tight_layout()
plt.show()

plt.figure(figsize=(12, 6))
shap.plots.beeswarm(shap_values, max_display=12, show=False)
plt.title("SHAP Summary - Beeswarm")
plt.tight_layout()
plt.show()

# 8. dependence plots (개별 생성: figure 자동 생성됨)
features_to_plot = ['tenure', 'MonthlyCharges', 'TotalCharges', 'Contract', 'InternetService', 'PaymentMethod']
for feature in features_to_plot:
    shap.plots.scatter(shap_values[:, feature], color=shap_values, title=f"Dependence Plot - {feature}")

# 9. force plot (JS 기반 단일 예측)
shap.initjs()
shap.plots.force(shap_values[0])