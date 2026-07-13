import numpy as np
from typing import List, Dict, Any
from app.ml.dnn_health import dnn_health_model

class SHAPExplainer:
    """
    Explainable AI (XAI) using exact Shapley Values.
    Calculates the contributions of each of the 5 financial features 
    towards the Financial Health Score output from the DNN.
    
    Uses the exact cooperative game theory formula:
      phi_i = Sum_[S in N\\{i}] (|S|! * (M - |S| - 1)! / M!) * (f(S U {i}) - f(S))
      where M = 5 features.
    
    Missing features are replaced by baseline median values:
      - Savings Rate: 0.20 (20%)
      - Debt-to-Income: 0.35 (35%)
      - Investment Ratio: 0.10 (10%)
      - Risk Score: 0.50 (Moderate)
      - Goal Progress: 0.30 (30%)
    """
    def __init__(self):
        self.feature_names = [
            "Savings Rate", 
            "Debt-to-Income Ratio", 
            "Investment Ratio", 
            "Risk Profile", 
            "Goal Progress"
        ]
        self.baselines = np.array([0.20, 0.35, 0.10, 0.50, 0.30])
        self.num_features = 5

    def _factorial(self, n: int) -> int:
        if n == 0 or n == 1:
            return 1
        return n * self._factorial(n - 1)

    def explain(self, current_features: np.ndarray) -> List[Dict[str, Any]]:
        """
        Computes exact Shapley values for the given user feature vector.
        Returns a list of dictionaries with feature name, attribution value, and description.
        """
        if len(current_features.shape) == 1:
            current_features = current_features.reshape(1, -1)
            
        shap_values = np.zeros(self.num_features)
        M = self.num_features
        
        # Generate all 2^5 = 32 coalitions
        # We index subsets by binary representations (0 to 31)
        for i in range(M):
            phi_i = 0.0
            # Iterate through all coalitions not containing i
            for s_mask in range(1 << M):
                # Check if feature i is in the coalition
                if (s_mask & (1 << i)) != 0:
                    continue
                    
                # Coalition S size
                S_size = bin(s_mask).count("1")
                
                # Weight = |S|! * (M - |S| - 1)! / M!
                weight = (self._factorial(S_size) * self._factorial(M - S_size - 1)) / self._factorial(M)
                
                # Create vector for S
                x_S = np.copy(self.baselines)
                for j in range(M):
                    if (s_mask & (1 << j)) != 0:
                        x_S[j] = current_features[0, j]
                        
                # Create vector for S U {i}
                x_S_i = np.copy(x_S)
                x_S_i[i] = current_features[0, i]
                
                # Predict differences
                pred_S = dnn_health_model.predict(x_S)
                pred_S_i = dnn_health_model.predict(x_S_i)
                
                # Accumulate weighted contribution
                phi_i += weight * (pred_S_i - pred_S)
                
            shap_values[i] = phi_i
            
        # Compile explanations
        explanations = []
        descriptions = {
            "Savings Rate": "impact of your monthly savings percentage",
            "Debt-to-Income Ratio": "effect of your debt obligations against income",
            "Investment Ratio": "contribution of your stock/mutual fund investments",
            "Risk Profile": "alignment of your current investments with your risk tolerance",
            "Goal Progress": "effect of progress towards your financial goals"
        }
        
        for idx, val in enumerate(shap_values):
            feat_name = self.feature_names[idx]
            
            # Formulate helpful user explanation
            direction = "boosted" if val >= 0 else "reduced"
            impact_adjective = "significantly" if abs(val) > 5 else "slightly"
            desc = f"Your {feat_name.lower()} has {impact_adjective} {direction} your health score by {abs(val):.1f} points. ({descriptions[feat_name]})"
            
            explanations.append({
                "feature_name": feat_name,
                "attribution_value": float(val),
                "description": desc
            })
            
        return explanations

# Global singleton
shap_explainer = SHAPExplainer()
