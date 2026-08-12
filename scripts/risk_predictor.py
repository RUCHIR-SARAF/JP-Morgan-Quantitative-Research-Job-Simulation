import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import StandardScaler
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import roc_auc_score, roc_curve, log_loss, brier_score_loss
df = pd.read_csv('Task 3 and 4_Loan_Data.csv')
df.head()
df.info()
df.describe()
df['default'].value_counts(normalize = True)
df.corr()['default'].sort_values(ascending = False)
sns.boxplot(x='default', y='fico_score', data = df)
df.isnull().sum()
x = df.drop(columns=['customer_id', 'default'])
y = df['default']
x_train, x_test, y_train, y_test = train_test_split (
    x,y, test_size = 0.2, random_state = 42, stratify = y)
scaler = StandardScaler()
x_train_scaled = scaler.fit_transform(x_train)
x_test_scaled = scaler.transform(x_test)

logreg = LogisticRegression()
logreg.fit(x_train_scaled, y_train)
pd_pred = logreg.predict_proba(x_test_scaled)[:, 1] #probability of class 1
rf = RandomForestClassifier(n_estimators = 200, max_depth = 5, random_state = 42)
rf.fit(x_train, y_train)
pd_pred_rf = rf.predict_proba(x_test)[:, 1]
auc = roc_auc_score(y_test, pd_pred)
print("AUC:", auc)

results = {
    "Logistic Regression":{
        "AUC": roc_auc_score(y_test, pd_pred),
        "LogLoss": log_loss(y_test, pd_pred),
        "brier": brier_score_loss(y_test, pd_pred)
    },
    "Random Forest":{
        "AUC": roc_auc_score(y_test, pd_pred_rf),
        "LogLoss": log_loss(y_test, pd_pred_rf),
        "Brier": brier_score_loss(y_test, pd_pred_rf)
    }
}
pd.DataFrame(results).T
def predict_expected_loss( borrower_feature: dict, model, scaler = None, recovery_rate = 0.10):
    x = pd.DataFrame([borrower_feature])
    if scaler is not None:
        x = scaler.transform(x)
    pd_estimate = model.predict_proba(x)[:,1][0]
    lgd = 1 - recovery_rate #loss given default (lgd)
    ead = borrower_feature['loan_amt_outstanding']#exposure at default (ead)
    expected_loss = pd_estimate*lgd*ead
    return expected_loss, pd_estimate

example_borrower = {
    'credit_lines_outstanding': 5,
    'loan_amt_outstanding': 4000,
    'total_debt_outstanding': 6000,
    'income': 300000,
    'years_employed': 4,
    'fico_score': 695
    }
el, pd_val = predict_expected_loss(example_borrower, logreg, scaler)
print(f"PD:{pd_val:.3f}, Expected Loss: ${el:.2f}")