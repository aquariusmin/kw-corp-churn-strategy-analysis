"""
Telco Customer Churn - XGBoost Explainability (SHAP + PDP)
Author: <Your Name>
"""

# -----------------------------
# 0. Imports & Settings
# -----------------------------
import warnings
warnings.simplefilter(action='ignore', category=FutureWarning)

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

# 모델/평가
from sklearn.model_selection import train_test_split
from sklearn.metrics import (
    accuracy_score, precision_score, recall_score,
    roc_auc_score, confusion_matrix, classification_report, roc_curve
)

# XGBoost
import xgboost as xgb

# SHAP
import shap  # pip install shap

# PDP
from sklearn.inspection import PartialDependenceDisplay


# -----------------------------
# 1. Load & Clean Data
# -----------------------------
FILE_PATH = "/Users/sangmin/Desktop/KW/kwco/WA_Fn-UseC_-Telco-Customer-Churn.csv"  # 경로 조정

df = pd.read_csv(FILE_PATH)
df.columns = df.columns.str.strip()

# 공백 문자열 → NaN
df.replace(" ", np.nan, inplace=True)

# 필수 수치형 변환
df['TotalCharges'] = pd.to_numeric(df['TotalCharges'], errors='coerce')

# 결측 제거 (간단 전략; 대안: 중앙값 대체 등)
df.dropna(inplace=True)

# 타겟 인코딩
df['Churn'] = df['Churn'].map({'Yes': 1, 'No': 0})

# SeniorCitizen (0/1 그대로 둬도 됨; 해석용으로는 0/1 유지)
# 다른 이진 문자형 변수는 map
binary_map_cols = ['Partner', 'Dependents', 'PhoneService', 'PaperlessBilling']
for col in binary_map_cols:
    df[col] = df[col].map({'Yes': 1, 'No': 0})

# gender 인코딩
df['gender'] = df['gender'].map({'Female': 1, 'Male': 0})

# customerID 제거
if 'customerID' in df.columns:
    df.drop('customerID', axis=1, inplace=True)

# -----------------------------
# 2. One-Hot Encoding (multi-category)
# -----------------------------
multi_cat_cols = [
    'MultipleLines', 'InternetService', 'OnlineSecurity', 'OnlineBackup',
    'DeviceProtection', 'TechSupport', 'StreamingTV', 'StreamingMovies',
    'Contract', 'PaymentMethod'
]

df = pd.get_dummies(df, columns=multi_cat_cols, drop_first=True)

# -----------------------------
# 3. Split Train/Test (70:30)
# -----------------------------
X = df.drop('Churn', axis=1)
y = df['Churn']

X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.30, stratify=y, random_state=42
)


# -----------------------------
# 4. Train XGBoost
# -----------------------------
# NOTE: use_label_encoder deprecated → omit
xgb_clf = xgb.XGBClassifier(
    eval_metric='logloss',
    random_state=42,
    n_estimators=300,
    learning_rate=0.05,
    max_depth=4,
    subsample=0.8,
    colsample_bytree=0.8,
    reg_lambda=1.0
)
xgb_clf.fit(X_train, y_train)


# -----------------------------
# 5. Evaluate
# -----------------------------
y_pred = xgb_clf.predict(X_test)
y_proba = xgb_clf.predict_proba(X_test)[:, 1]

acc = accuracy_score(y_test, y_pred)
prec = precision_score(y_test, y_pred)
rec = recall_score(y_test, y_pred)
auc = roc_auc_score(y_test, y_proba)
cm = confusion_matrix(y_test, y_pred)

print("=== XGBoost Performance (Test) ===")
print(f"Accuracy : {acc:.4f}")
print(f"Precision: {prec:.4f}")
print(f"Recall   : {rec:.4f}")
print(f"ROC AUC  : {auc:.4f}")
print("\nConfusion Matrix:\n", cm)
print("\nClassification Report:\n", classification_report(y_test, y_pred, digits=4))


# -----------------------------
# 6. Feature Importance (Gain-based from XGB)
# -----------------------------
xgb_importances = pd.DataFrame({
    "Feature": X.columns,
    "Importance": xgb_clf.feature_importances_
}).sort_values(by="Importance", ascending=False)

print("\nTop 10 Features (XGB native importance):")
print(xgb_importances.head(10))

# Optional: bar plot
plt.figure(figsize=(8, 6))
sns.barplot(
    data=xgb_importances.head(10),
    x='Importance', y='Feature', palette='viridis'
)
plt.title('Top 10 Features (XGBoost Importance)')
plt.xlabel('Importance')
plt.ylabel('Feature')
plt.tight_layout()
plt.show()


# -----------------------------
# 7. SHAP Explainability
# -----------------------------
# (a) Explainer
# TreeExplainer is fast for tree-based models
explainer = shap.TreeExplainer(xgb_clf)
shap_values = explainer.shap_values(X_test)

# (b) Summary Plot (beeswarm)
# NOTE: Displays distribution of SHAP values per feature
shap.summary_plot(shap_values, X_test, show=False, plot_size=(10,6))
plt.title("SHAP Summary (Beeswarm) - Test Set", loc='left')
plt.tight_layout()
plt.show()

# (c) Bar plot (mean |SHAP| importance)
shap.summary_plot(shap_values, X_test, plot_type="bar", show=False, plot_size=(8,6))
plt.title("SHAP Feature Importance (mean |SHAP|)", loc='left')
plt.tight_layout()
plt.show()

# -----------------------------
# 8. SHAP Dependence Plots (Top Features)
# -----------------------------
# 안전 체크: 상위 n개 실제 존재 여부
top_feats = xgb_importances['Feature'].head(5).tolist()
for feat in top_feats:
    if feat not in X_test.columns:
        print(f"[WARN] {feat} not in X_test columns (skipped).")

# 첫 3개만 예시
for feat in top_feats[:3]:
    shap.dependence_plot(feat, shap_values, X_test, show=False)
    plt.title(f"SHAP Dependence: {feat}", loc='left')
    plt.tight_layout()
    plt.show()

# 상호작용: tenure vs MonthlyCharges (존재 확인 후)
if 'tenure' in X_test.columns and 'MonthlyCharges' in X_test.columns:
    shap.dependence_plot('tenure', shap_values, X_test, interaction_index='MonthlyCharges', show=False)
    plt.title('SHAP Dependence: tenure × MonthlyCharges', loc='left')
    plt.tight_layout()
    plt.show()


# -----------------------------
# 9. SHAP Force Plot (Single Example)
# -----------------------------
# Force plot은 notebook/HTML에서 인터랙티브. 이미지 저장하려면 shap.save_html 사용.
sample_idx = 0
sample_data = X_test.iloc[sample_idx:sample_idx+1]
sample_shap = shap_values[sample_idx:sample_idx+1]

force_plot = shap.force_plot(
    explainer.expected_value, sample_shap, sample_data, matplotlib=True
)
plt.title(f"SHAP Force Plot (Sample idx={sample_idx})")
plt.show()

# HTML 저장 (추천)
# shap.save_html("shap_force_sample.html",
#                shap.force_plot(explainer.expected_value, sample_shap, sample_data))


# -----------------------------
# 10. Partial Dependence Plots (Sklearn)
# -----------------------------
# PDP는 특성값 변화에 따른 평균 예측 확률 변화를 보여줌.
# Note: binary encoded dummies는 해석에 주의.
# -----------------------------
# 10. Partial Dependence Plots (Sklearn) - 수정 버전
# -----------------------------
# PDP에 적합한 연속형 변수만 선택
pdp_features = []

# 연속형 또는 범주가 충분히 다양한 변수만 선택 (ex: > 2 unique values)
for col in ['tenure', 'MonthlyCharges', 'TotalCharges']:
    if col in X_train.columns and X_train[col].nunique() > 2:
        pdp_features.append(col)

if pdp_features:
    fig, ax = plt.subplots(figsize=(10, 4 * len(pdp_features)))
    PartialDependenceDisplay.from_estimator(
        xgb_clf, X_train, pdp_features,
        kind='average', ax=ax
    )
    fig.suptitle("Partial Dependence Plots (XGBoost)", fontsize=14)
    plt.tight_layout()
    plt.show()
else:
    print("[INFO] No valid features found for PDP.")

# -----------------------------
# 11. (Optional) Save Artifacts
# -----------------------------
# xgb_clf.save_model("xgb_telco_model.json")
# xgb_importances.to_csv("xgb_feature_importance.csv", index=False)