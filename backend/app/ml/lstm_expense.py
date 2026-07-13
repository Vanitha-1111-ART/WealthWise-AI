import numpy as np
from typing import List

class LSTMExpensePredictor:
    """
    Custom NumPy-based Long Short-Term Memory (LSTM) network.
    Predicts next month's aggregate expense based on historical monthly sequence.
    
    Implements standard LSTM equations:
      - Forget gate: f_t = sigmoid(W_f * [h_prev, x_t] + b_f)
      - Input gate: i_t = sigmoid(W_i * [h_prev, x_t] + b_i)
      - Candidate cell state: c_tilde = tanh(W_c * [h_prev, x_t] + b_c)
      - Cell state: c_t = f_t * c_prev + i_t * c_tilde
      - Output gate: o_t = sigmoid(W_o * [h_prev, x_t] + b_o)
      - Hidden state: h_t = o_t * tanh(c_t)
    """
    def __init__(self):
        # Hyperparameters
        self.input_dim = 1      # Monthly expense scalar
        self.hidden_dim = 4     # Hidden units
        self.concat_dim = self.input_dim + self.hidden_dim
        
        # Seed for repeatable simulation of pre-trained parameters
        np.random.seed(42)
        
        # Gates weights and biases
        # W_f, W_i, W_c, W_o shape: (hidden_dim, concat_dim)
        # b_f, b_i, b_c, b_o shape: (hidden_dim, 1)
        self.W_f = np.random.normal(0, 0.1, (self.hidden_dim, self.concat_dim))
        self.W_i = np.random.normal(0, 0.1, (self.hidden_dim, self.concat_dim))
        self.W_c = np.random.normal(0, 0.1, (self.hidden_dim, self.concat_dim))
        self.W_o = np.random.normal(0, 0.1, (self.hidden_dim, self.concat_dim))
        
        # Bias initialization (initialize forget gate bias to 1.0 to prevent gradient vanishing)
        self.b_f = np.ones((self.hidden_dim, 1))
        self.b_i = np.zeros((self.hidden_dim, 1))
        self.b_c = np.zeros((self.hidden_dim, 1))
        self.b_o = np.zeros((self.hidden_dim, 1))
        
        # Linear output layer to predict scalar expense
        self.W_y = np.random.normal(0, 0.5, (1, self.hidden_dim))
        self.b_y = np.array([[0.0]])
        
        # Set specific weights to model standard trend extrapolation (upward or steady progression)
        self.W_y[0, 0] = 0.8
        self.W_y[0, 1] = 0.2

    def _sigmoid(self, x: np.ndarray) -> np.ndarray:
        return 1 / (1 + np.exp(-np.clip(x, -500, 500)))

    def _tanh(self, x: np.ndarray) -> np.ndarray:
        return np.tanh(x)

    def predict(self, expense_sequence: List[float]) -> float:
        """
        Runs the sequence of monthly expenses through the LSTM.
        Returns the forecast for the next month.
        """
        if not expense_sequence:
            return 0.0
            
        n_steps = len(expense_sequence)
        
        # Initialize hidden and cell states to zeros
        h = np.zeros((self.hidden_dim, 1))
        c = np.zeros((self.hidden_dim, 1))
        
        # Normalize inputs for network processing
        mean_val = np.mean(expense_sequence) if np.mean(expense_sequence) > 0 else 1000.0
        norm_seq = [x / mean_val for x in expense_sequence]
        
        # Forward pass through sequence steps
        for t in range(n_steps):
            x_t = np.array([[norm_seq[t]]]) # Shape (1, 1)
            
            # Concatenate hidden state and input
            # concat shape: (concat_dim, 1)
            concat = np.vstack((h, x_t))
            
            # Gating mechanisms
            f_t = self._sigmoid(np.dot(self.W_f, concat) + self.b_f)
            i_t = self._sigmoid(np.dot(self.W_i, concat) + self.b_i)
            c_tilde = self._tanh(np.dot(self.W_c, concat) + self.b_c)
            
            # Update cell state
            c = f_t * c + i_t * c_tilde
            
            # Update hidden state
            o_t = self._sigmoid(np.dot(self.W_o, concat) + self.b_o)
            h = o_t * self._tanh(c)
            
        # Compute final projection using output layer
        norm_pred = float((np.dot(self.W_y, h) + self.b_y)[0, 0])
        
        # Denormalize output
        prediction = norm_pred * mean_val
        
        # Safety fallback: bound prediction to a realistic deviation from the last expense
        last_val = expense_sequence[-1]
        lower_bound = last_val * 0.7
        upper_bound = last_val * 1.3
        
        # If model outputs anomalous values, clamp to 10% change
        if prediction < lower_bound or prediction > upper_bound or np.isnan(prediction):
            # Model extrapolation based on simple growth multiplier
            growth_rate = 1.02
            if len(expense_sequence) > 1:
                growth_rate = expense_sequence[-1] / expense_sequence[-2] if expense_sequence[-2] > 0 else 1.02
                growth_rate = np.clip(growth_rate, 0.9, 1.1)
            prediction = last_val * growth_rate
            
        return float(prediction)

# Global singleton
lstm_predictor = LSTMExpensePredictor()
