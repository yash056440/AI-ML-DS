"""
Logistic Regression: Loan Default Prediction
=============================================
Predicts P(Default = 1) from two features:
    x1 = Debt-to-Income ratio (decimal, e.g. 0.35 = 35%)
    x2 = Credit Score (300-850 range)

Model:
    z = b0 + b1 * x1_scaled + b2 * x2_scaled
    P(default) = 1 / (1 + exp(-z))
"""

import numpy as np
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import (
    accuracy_score, confusion_matrix, classification_report,
    roc_auc_score
)

# ---------------------------------------------------------------
# 1. Training data
# ---------------------------------------------------------------
X = np.array([
    [0.15, 780],
    [0.42, 620],
    [0.28, 700],
    [0.55, 580],
    [0.20, 750],
    [0.48, 600],
    [0.33, 660],
    [0.60, 550],
    [0.25, 720],
    [0.50, 590],
])
y = np.array([0, 1, 0, 1, 0, 1, 0, 1, 0, 1])  # 1 = Default, 0 = No Default

feature_names = ["DTI", "Credit_Score"]

# ---------------------------------------------------------------
# 2. Standardize features (critical: DTI ~0-1, Score ~300-850)
# ---------------------------------------------------------------
scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)

print("Feature means:", dict(zip(feature_names, scaler.mean_)))
print("Feature stds: ", dict(zip(feature_names, scaler.scale_)))

# ---------------------------------------------------------------
# 3. Fit logistic regression
# ---------------------------------------------------------------
model = LogisticRegression()
model.fit(X_scaled, y)

b0 = model.intercept_[0]
b1, b2 = model.coef_[0]

print("\n--- Fitted coefficients (on standardized features) ---")
print(f"Intercept (b0):        {b0:.4f}")
print(f"Coef DTI (b1):         {b1:.4f}")
print(f"Coef Credit Score (b2):{b2:.4f}")

print("\nEquation:")
print(f"z = {b0:.4f} + ({b1:.4f} * DTI_scaled) + ({b2:.4f} * Score_scaled)")
print("P(default) = 1 / (1 + exp(-z))")

# ---------------------------------------------------------------
# 4. Evaluate on training data
# ---------------------------------------------------------------
y_prob = model.predict_proba(X_scaled)[:, 1]
y_pred = model.predict(X_scaled)

print("\n--- Predictions on training data ---")
for xi, yi, p, pr in zip(X, y, y_prob, y_pred):
    print(f"DTI={xi[0]:.2f}, Score={xi[1]:.0f} | actual={yi} | "
          f"P(default)={p:.4f} | predicted={pr}")

print("\nAccuracy:", accuracy_score(y, y_pred))
print("AUC-ROC: ", roc_auc_score(y, y_prob))
print("\nConfusion matrix:\n", confusion_matrix(y, y_pred))
print("\nClassification report:\n", classification_report(y, y_pred))

# ---------------------------------------------------------------
# 5. Reusable prediction function for new applicants
# ---------------------------------------------------------------
def predict_default_probability(dti: float, credit_score: float) -> float:
    """
    Given a new applicant's DTI ratio and credit score,
    return the predicted probability of default.
    """
    x_new = np.array([[dti, credit_score]])
    x_new_scaled = scaler.transform(x_new)
    prob = model.predict_proba(x_new_scaled)[0, 1]
    return prob


def classify_applicant(dti: float, credit_score: float, threshold: float = 0.5):
    """
    Returns (probability, decision) for a new applicant.
    decision = "Default" if prob >= threshold else "No Default"
    """
    prob = predict_default_probability(dti, credit_score)
    decision = "Default" if prob >= threshold else "No Default"
    return prob, decision


# ---------------------------------------------------------------
# 6. Example predictions on new applicants
# ---------------------------------------------------------------
print("\n--- New applicant predictions ---")
new_applicants = [
    (0.38, 640),   # borderline case
    (0.10, 800),   # very safe
    (0.65, 540),   # very risky
    (0.30, 690),
]

for dti, score in new_applicants:
    prob, decision = classify_applicant(dti, score)
    print(f"DTI={dti:.2f}, Score={score} -> "
          f"P(default)={prob:.4f} -> Decision: {decision}")
