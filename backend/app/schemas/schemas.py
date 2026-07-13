from pydantic import BaseModel, EmailStr, Field
from typing import List, Optional, Dict, Any
from datetime import datetime

# --- Auth Schemas ---
class Token(BaseModel):
    access_token: str
    token_type: str

class TokenData(BaseModel):
    user_id: Optional[str] = None

class UserLogin(BaseModel):
    email: EmailStr
    password: str

class UserRegister(BaseModel):
    email: EmailStr
    password: str
    full_name: str
    risk_tolerance: Optional[str] = "Moderate"
    monthly_income: Optional[float] = 0.0
    monthly_expenses: Optional[float] = 0.0
    investment_goals: Optional[str] = ""

# --- User Schemas ---
class UserProfileUpdate(BaseModel):
    full_name: Optional[str] = None
    risk_tolerance: Optional[str] = None
    monthly_income: Optional[float] = None
    monthly_expenses: Optional[float] = None
    investment_goals: Optional[str] = None

class UserResponse(BaseModel):
    id: int
    email: EmailStr
    full_name: str
    risk_tolerance: str
    monthly_income: float
    monthly_expenses: float
    investment_goals: str
    is_admin: bool
    created_at: datetime

    class Config:
        from_attributes = True

# --- Transaction Schemas ---
class TransactionCreate(BaseModel):
    amount: float = Field(..., gt=0)
    category: str  # Food, Rent, Utilities, Entertainment, Investments, Healthcare, Others
    description: Optional[str] = None
    timestamp: Optional[datetime] = None

class TransactionResponse(BaseModel):
    id: int
    user_id: int
    amount: float
    category: str
    description: Optional[str]
    timestamp: datetime

    class Config:
        from_attributes = True

# --- Asset / Portfolio Schemas ---
class AssetCreate(BaseModel):
    asset_name: str
    asset_type: str  # Cash, Stock, Mutual Fund, FD, Gold
    amount: float = Field(..., ge=0)
    current_value: float = Field(..., ge=0)

class AssetResponse(BaseModel):
    id: int
    user_id: int
    asset_name: str
    asset_type: str
    amount: float
    current_value: float

    class Config:
        from_attributes = True

class PortfolioSummary(BaseModel):
    total_net_worth: float
    asset_mix: Dict[str, float]  # Type -> percentage
    assets: List[AssetResponse]
    risk_alignment: str          # "Aligned", "Rebalancing Recommended", "Misaligned"
    investment_recommendations: List[str]

# --- Goal Schemas ---
class GoalCreate(BaseModel):
    goal_name: str
    target_amount: float = Field(..., gt=0)
    current_amount: Optional[float] = 0.0
    target_date: datetime

class GoalResponse(BaseModel):
    id: int
    user_id: int
    goal_name: str
    target_amount: float
    current_amount: float
    target_date: datetime
    progress_percentage: float

    class Config:
        from_attributes = True

# --- Budget Schemas ---
class BudgetCreate(BaseModel):
    category: str
    amount_limit: float = Field(..., gt=0)

class BudgetResponse(BaseModel):
    id: int
    user_id: int
    category: str
    amount_limit: float
    amount_spent: float
    percentage_spent: float

    class Config:
        from_attributes = True

# --- Alert Schemas ---
class AlertResponse(BaseModel):
    id: int
    user_id: int
    title: str
    message: str
    type: str
    is_read: bool
    created_at: datetime

    class Config:
        from_attributes = True

# --- Machine Learning & Explainable AI (SHAP) ---
class SHAPAttribution(BaseModel):
    feature_name: str
    attribution_value: float  # Positive or negative impact on the health score
    description: str

class MLPredictionResponse(BaseModel):
    health_score: float
    predicted_expenses: float
    spending_behavior: str
    shap_explanation: List[SHAPAttribution]
    recommendations: List[str]
    created_at: datetime

# --- AI Chat & Avatar ---
class ChatMessage(BaseModel):
    message: str
    voice_input: Optional[bool] = False

class ChatResponse(BaseModel):
    message: str
    voice_output_url: Optional[str] = None  # URL to audio binary or Base64 string for voice assistant
    suggestions: List[str]

# --- Admin Panel ---
class SystemStatsResponse(BaseModel):
    total_users: int
    total_assets_tracked: float
    total_goals_set: int
    total_transactions_logged: int
    ml_model_accuracy: Dict[str, float]
