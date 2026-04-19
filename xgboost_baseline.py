import pandas as pd
import numpy as np
import joblib
from sklearn.metrics import classification_report, accuracy_score, f1_score, matthews_corrcoef
from xgboost import XGBClassifier

# 1. Load Data
data = joblib.load('outputs/models/preprocessed_data.joblib')
X_train, y_train = data['X_train'], data['y_train']
X_val, y_val = data['X_val'], data['y_val']

# 2. XGBoost Baseline
print("Training XGBoost Baseline...")
unique_classes = np.unique(y_train)
class_weights_dict = {c: len(y_train) / (len(unique_classes) * np.sum(y_train == c)) for c in unique_classes}
sample_weights = np.array([class_weights_dict[y] for y in y_train])

xgb = XGBClassifier(
    objective='multi:softprob',
    num_class=3,
    eval_metric='mlogloss',
    max_depth=5,
    learning_rate=0.1,
    n_estimators=100,
    random_state=42
)

xgb.fit(X_train, y_train, sample_weight=sample_weights)

xgb_val_probs = xgb.predict_proba(X_val)
xgb_val_preds = xgb.predict(X_val)

print("XGBoost Val Results:")
print("Accuracy:", accuracy_score(y_val, xgb_val_preds))
print("Macro F1:", f1_score(y_val, xgb_val_preds, average='macro'))
print("MCC:", matthews_corrcoef(y_val, xgb_val_preds))
print("\nRecall per class:\n", classification_report(y_val, xgb_val_preds, target_names=['Fatal', 'Serious', 'Slight']))
