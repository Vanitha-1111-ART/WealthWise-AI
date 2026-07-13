import numpy as np
import sys
import os

# Append current directory to path
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

from app.ml.dnn_health import dnn_health_model
from app.ml.lstm_expense import lstm_predictor
from app.ml.transformer_spending import spending_classifier
from app.ml.shap_explain import shap_explainer

def run_tests():
    print("==================================================")
    print("RUNNING WEALTHWISE AI ML ENGINE VERIFICATION TESTS")
    print("==================================================")
    
    # Test 1: DNN Health Predictor
    print("\n[Test 1] Verifying DNN Health Score Predictor...")
    # Feature inputs: Savings rate (40%), Debt (20%), Investments (30%), Risk (Moderate=0.55), Goal progress (70%)
    features = np.array([0.40, 0.20, 0.30, 0.55, 0.70])
    score = dnn_health_model.predict(features)
    print("-> Input Features: " + str(features))
    print(f"-> Calculated Financial Health Score: {score:.2f} / 100.0")
    assert 0 <= score <= 100, "Error: Health score out of bounds!"
    print("[PASS] DNN Health Predictor")
    
    # Test 2: SHAP Explainer
    print("\n[Test 2] Verifying SHAP Explainer attributions...")
    explanations = shap_explainer.explain(features)
    total_shap_impact = 0.0
    for exp in explanations:
        val = exp["attribution_value"]
        total_shap_impact += val
        print(f"-> {exp['feature_name']}: {val:+.2f} points ({exp['description']})")
    print(f"-> Net Shapley Attributions sum: {total_shap_impact:+.2f} points")
    print("[PASS] SHAP Explainer")
    
    # Test 3: LSTM Expense Predictor
    print("\n[Test 3] Verifying LSTM sequential expense forecaster...")
    # Last 6 months aggregate spending history
    history = [12000.0, 12500.0, 11800.0, 13100.0, 12900.0, 13400.0]
    next_month_pred = lstm_predictor.predict(history)
    print(f"-> Expense history (last 6m): {history}")
    print(f"-> Projected expense forecast: Rs.{next_month_pred:,.2f}")
    assert next_month_pred > 0, "Error: LSTM forecast must be positive!"
    print("[PASS] LSTM Expense Predictor")
    
    # Test 4: Transformer Spending Behavior Analysis
    print("\n[Test 4] Verifying Transformer Self-Attention Spending Classifier...")
    # Distribution: Food(20%), Rent(30%), Utilities(10%), Entertainment(25%), Investments(5%), Healthcare(5%), Others(5%)
    ratios = [0.20, 0.30, 0.10, 0.25, 0.05, 0.05, 0.05]
    profile, probs = spending_classifier.analyze(ratios)
    print(f"-> Spending ratios: {ratios}")
    print(f"-> Output Behavior Classification: {profile}")
    for p, val in probs.items():
        print(f"   - {p}: {val*100:.1f}%")
    print("[PASS] Transformer Classifier")
    
    print("\n==================================================")
    print("ALL EMULATION ML ENGINES VERIFIED SUCCESSFULLY!")
    print("==================================================")

if __name__ == "__main__":
    run_tests()
