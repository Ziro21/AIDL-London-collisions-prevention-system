import pandas as pd
import numpy as np
import joblib
import torch
from torch import nn
from sklearn.metrics import classification_report, accuracy_score, f1_score, matthews_corrcoef, roc_auc_score
from xgboost import XGBClassifier
import matplotlib.pyplot as plt

# 1. Load Data
data = joblib.load('outputs/models/preprocessed_data.joblib')
X_train, y_train = data['X_train'], data['y_train']
X_val, y_val = data['X_val'], data['y_val']
X_test, y_test = data['X_test'], data['y_test']
class_weights = data['class_weights']
feature_names = data['feature_names']

# 2. Re-create MLP Architecture and Load Best Checkpoint
class CollisionMLP(nn.Module):
    def __init__(self, input_dim, num_classes=3, dropout=0.3):
        super().__init__()
        self.network = nn.Sequential(
            nn.Linear(input_dim, 64),
            nn.BatchNorm1d(64),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(64, 32),
            nn.BatchNorm1d(32),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(32, num_classes)
        )
        
    def forward(self, x):
        return self.network(x)

mlp = CollisionMLP(input_dim=X_train.shape[1], num_classes=3, dropout=0.3)
checkpoint = torch.load('outputs/models/best_arch_mlp_model.pt', map_location='cpu', weights_only=False)
mlp.load_state_dict(checkpoint['model_state_dict'])
mlp.eval()

# Get Val logits
X_val_tensor = torch.FloatTensor(X_val)
y_val_tensor = torch.LongTensor(y_val)
with torch.no_grad():
    val_logits = mlp(X_val_tensor)
    val_probs = torch.softmax(val_logits, dim=1).numpy()

# 3. Calculate Expected Calibration Error (ECE)
def calculate_ece(probs, labels, n_bins=10):
    confidences = np.max(probs, axis=1)
    predictions = np.argmax(probs, axis=1)
    accuracies = predictions == labels
    
    ece = 0
    bin_boundaries = np.linspace(0, 1, n_bins + 1)
    
    for bin_lower, bin_upper in zip(bin_boundaries[:-1], bin_boundaries[1:]):
        in_bin = (confidences > bin_lower) & (confidences <= bin_upper)
        prop_in_bin = in_bin.mean()
        
        if prop_in_bin > 0:
            accuracy_in_bin = accuracies[in_bin].mean()
            avg_confidence_in_bin = confidences[in_bin].mean()
            ece += np.abs(avg_confidence_in_bin - accuracy_in_bin) * prop_in_bin
            
    return ece

uncalibrated_ece = calculate_ece(val_probs, y_val)
print(f"Uncalibrated ECE (Val): {uncalibrated_ece:.4f}")

# 4. Temperature Scaling
class ModelWithTemperature(nn.Module):
    def __init__(self, model):
        super().__init__()
        self.model = model
        self.temperature = nn.Parameter(torch.ones(1) * 1.5) # Start T>1 to soften

    def forward(self, input):
        logits = self.model(input)
        return self.temperature_scale(logits)

    def temperature_scale(self, logits):
        temperature = self.temperature.unsqueeze(1).expand(logits.size(0), logits.size(1))
        return logits / temperature

    def set_temperature(self, val_logits, y_val_tensor):
        nll_criterion = nn.CrossEntropyLoss()
        optimizer = torch.optim.LBFGS([self.temperature], lr=0.01, max_iter=50)
        
        def eval():
            optimizer.zero_grad()
            loss = nll_criterion(self.temperature_scale(val_logits), y_val_tensor)
            loss.backward()
            return loss
            
        optimizer.step(eval)
        print(f"Optimal Temperature: {self.temperature.item():.4f}")

        # Recalculate ECE
        calibrated_logits = self.temperature_scale(val_logits)
        calibrated_probs = torch.softmax(calibrated_logits, dim=1).detach().numpy()
        calibrated_ece = calculate_ece(calibrated_probs, y_val_tensor.numpy())
        print(f"Calibrated ECE (Val): {calibrated_ece:.4f}")
        return self.temperature.item()

scaled_model = ModelWithTemperature(mlp)
optimal_t = scaled_model.set_temperature(val_logits, y_val_tensor)

# 5. XGBoost Baseline
print("\nTraining XGBoost Baseline...")
# Use inverse frequency weights
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

# Save artifacts
joblib.dump({
    'uncalibrated_ece': uncalibrated_ece,
    'optimal_temperature': optimal_t,
    'calibrated_ece': calculate_ece(torch.softmax(scaled_model.temperature_scale(val_logits), dim=1).detach().numpy(), y_val),
    'xgb_model': xgb,
    'xgb_val_preds': xgb_val_preds,
    'xgb_val_probs': xgb_val_probs,
    'xgb_mcc': matthews_corrcoef(y_val, xgb_val_preds)
}, 'outputs/models/production_readiness.joblib')
print("Saved artifacts to outputs/models/production_readiness.joblib")
