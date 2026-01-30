import warnings
warnings.simplefilter(action='ignore', category=FutureWarning)

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import accuracy_score, precision_score, recall_score, roc_auc_score, roc_curve
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier
from sklearn.neighbors import KNeighborsClassifier
from sklearn.svm import SVC
from sklearn.neural_network import MLPClassifier
import xgboost as xgb

# 데이터 전처리
file_path = '/Users/sangmin/Desktop/KW/kwco/WA_Fn-UseC_-Telco-Customer-Churn.csv'
df = pd.read_csv(file_path)
df.columns = df.columns.str.strip()
df.replace(" ", np.nan, inplace=True)
df['TotalCharges'] = pd.to_numeric(df['TotalCharges'], errors='coerce')
df.dropna(inplace=True)

# 타겟 변수 인코딩
df['Churn'] = df['Churn'].map({'Yes': 1, 'No': 0})

# 이진 변수 인코딩
binary_cols = ['Partner', 'Dependents', 'PhoneService', 'PaperlessBilling']
for col in binary_cols:
    df[col] = df[col].map({'Yes': 1, 'No': 0})
df['gender'] = df['gender'].map({'Female': 1, 'Male': 0})

# ID 컬럼 제거
if 'customerID' in df.columns:
    df.drop('customerID', axis=1, inplace=True)

# 범주형 변수 원-핫 인코딩
cat_cols = df.select_dtypes(include='object').columns
df = pd.get_dummies(df, columns=cat_cols)

# 데이터 분할 (70:30)
X = df.drop('Churn', axis=1)
y = df['Churn']
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=42)

# 스케일링
scaler = StandardScaler()
X_train = scaler.fit_transform(X_train)
X_test = scaler.transform(X_test)

# 모델 정의
models = {
    "Logistic Regression": LogisticRegression(random_state=42),
    "Random Forest": RandomForestClassifier(random_state=42),
    "Grdient Boosting": GradientBoostingClassifier(random_state=42),
    "KNN": KNeighborsClassifier(n_neighbors=5),
    "SVM": SVC(probability=True, random_state=42),
    "ANN": MLPClassifier(random_state=42),
    "XGBoost": xgb.XGBClassifier(random_state=42)
}

# 모델 학습 및 평가
results = []
plt.figure(figsize=(12, 8))

for name, model in models.items():
    # 모델 학습
    model.fit(X_train, y_train)
    
    # 예측
    y_pred = model.predict(X_test)
    y_prob = model.predict_proba(X_test)[:, 1]
    
    # 성능 지표 계산
    results.append({
        "Model": name,
        "Accuracy": accuracy_score(y_test, y_pred),
        "Precision": precision_score(y_test, y_pred),
        "Recall": recall_score(y_test, y_pred),
        "AUC": roc_auc_score(y_test, y_prob)
    })
    
    # ROC 커브
    fpr, tpr, _ = roc_curve(y_test, y_prob)
    plt.plot(fpr, tpr, label=f'{name} (AUC = {roc_auc_score(y_test, y_prob):.3f})')

# ROC 커브 시각화
plt.plot([0, 1], [0, 1], 'k--')
plt.xlabel('FPR')
plt.ylabel('TPR')
plt.title('ROC Curve Comparison (Train:Test = 70:30)')
plt.legend()
plt.grid(True)
plt.savefig('roc_curve_comparison.png')
plt.show()

# 성능 지표 데이터프레임
results_df = pd.DataFrame(results)
print("\n=== Model Comparison ===")
print(results_df)

# 성능 지표 시각화
plt.figure(figsize=(12, 6))
metrics = ['Accuracy', 'Precision', 'Recall', 'AUC']
for i, metric in enumerate(metrics):
    plt.subplot(2, 2, i+1)
    sns.barplot(data=results_df, x='Model', y=metric)
    plt.xticks(rotation=45)
    plt.title(f'Model {metric}')
plt.tight_layout()
plt.savefig('model_comparison_metrics.png')
plt.show()
