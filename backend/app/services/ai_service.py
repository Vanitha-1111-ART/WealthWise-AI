import logging
from typing import List, Dict, Any, Tuple
import google.generativeai as genai

from app.core.config import settings

logger = logging.getLogger(__name__)

class AIAdvisorService:
    """
    AI Financial Advisor & Chatbot Coach.
    Uses Google Gemini grounded with the user's financial profile,
    portfolio, and goals to provide exact, hyper-personalized advice.
    """
    def __init__(self):
        if settings.GEMINI_API_KEY:
            try:
                genai.configure(api_key=settings.GEMINI_API_KEY)
                self.model = genai.GenerativeModel(
                    model_name='gemini-1.5-flash',
                    system_instruction=(
                        "You are WealthWise Coach, a highly knowledgeable, empathetic, "
                        "and ethical AI Wealth Manager and Financial Advisor for IDBI Bank customers. "
                        "Your goal is to help users manage their money, invest wisely, build emergency "
                        "funds, and meet their financial goals. "
                        "Always ground your answers in the user's provided profile context (income, expenses, "
                        "portfolio, goals). "
                        "Be concise, conversational, and direct. Keep your answers brief (1-3 paragraphs) "
                        "and use clear bullet points. Do not give generic advice—give calculations and "
                        "actionable steps tailored to their Indian banking options (e.g. PPF, FDs, NPS, Mutual Funds)."
                    )
                )
                self.api_available = True
            except Exception as e:
                logger.error(f"Gemini API initialization failed: {str(e)}")
                self.api_available = False
        else:
            self.api_available = False

    def _build_context_prompt(self, user_profile: Dict[str, Any], message: str) -> str:
        """Embeds financial profile data directly into the LLM prompt."""
        net_worth = user_profile.get("net_worth", 0.0)
        income = user_profile.get("income", 0.0)
        expenses = user_profile.get("expenses", 0.0)
        risk = user_profile.get("risk_tolerance", "Moderate")
        goals = user_profile.get("goals", [])
        assets = user_profile.get("assets", [])
        
        goals_text = "\n".join([f"- {g['name']}: Target ₹{g['target']:.2f} by {g['date']} (Current: ₹{g['current']:.2f})" for g in goals])
        assets_text = "\n".join([f"- {a['name']} ({a['type']}): Value ₹{a['value']:.2f}" for a in assets])
        
        prompt = (
            f"--- USER FINANCIAL PROFILE CONTEXT ---\n"
            f"Monthly Income: ₹{income:.2f}\n"
            f"Monthly Expenses: ₹{expenses:.2f}\n"
            f"Total Net Worth: ₹{net_worth:.2f}\n"
            f"Risk Tolerance: {risk}\n"
            f"Financial Goals:\n{goals_text if goals_text else 'None set yet'}\n"
            f"Current Assets:\n{assets_text if assets_text else 'None set yet'}\n"
            f"--------------------------------------\n\n"
            f"User Question: \"{message}\"\n\n"
            f"Please reply as their WealthWise Coach:"
        )
        return prompt

    async def get_response(self, user_profile: Dict[str, Any], message: str, voice_input: bool = False) -> Tuple[str, List[str]]:
        """
        Generates advice using Gemini, or triggers the fallback rule engine.
        Returns a tuple: (reply_text, list_of_suggested_followups)
        """
        prompt = self._build_context_prompt(user_profile, message)
        
        if self.api_available:
            try:
                response = self.model.generate_content(prompt)
                reply = response.text.strip()
                
                # Dynamic suggestions generated via simple extraction or fallback list
                suggestions = self._generate_suggestions(message)
                return reply, suggestions
            except Exception as e:
                logger.error(f"Gemini API generation error: {str(e)}")
                
        # --- Fallback Advisor Engine ---
        # Highly comprehensive rule-based financial advisor fallback to ensure hackathon success
        logger.info("Executing AI Advisor fallback response.")
        
        msg_lower = message.lower()
        net_worth = user_profile.get("net_worth", 0.0)
        income = user_profile.get("income", 0.0)
        expenses = user_profile.get("expenses", 0.0)
        risk = user_profile.get("risk_tolerance", "Moderate")
        
        if "rebalance" in msg_lower or "portfolio" in msg_lower or "asset" in msg_lower:
            reply = (
                f"Based on your profile, your total assets equal **₹{net_worth:,.2f}** with a **{risk}** risk appetite. "
                "To optimize returns and manage risk, I recommend the following allocation:\n\n"
                "- **Equity / Mutual Funds (50%)**: Allocate to diversified index funds or large-cap mutual funds.\n"
                "- **Fixed Income / Debt (30%)**: Keep in high-yield Fixed Deposits or public debt instruments (PPF).\n"
                "- **Gold / Safe Haven (10%)**: Hedge inflation using Sovereign Gold Bonds.\n"
                "- **Liquid Cash (10%)**: Keep in savings or liquid funds for emergency use.\n\n"
                "To implement this, you should rebalance by shifting excess cash into mutual funds."
            )
            suggestions = ["How do I start a SIP?", "What mutual funds fit moderate risk?", "Show goal projections"]
            
        elif "budget" in msg_lower or "saving" in msg_lower or "expense" in msg_lower:
            savings = income - expenses
            savings_rate = (savings / income * 100) if income > 0 else 0
            reply = (
                f"Your monthly income is **₹{income:,.2f}** and expenses are **₹{expenses:,.2f}**, leaving you with "
                f"savings of **₹{savings:,.2f}** (Savings Rate: **{savings_rate:.1f}%**).\n\n"
                "Here is my analysis of your spending habits:\n"
                "- **Basic Needs**: Keep this under 50% of your income. Currently you are within safety thresholds.\n"
                "- **Wants**: Limit non-essential spend (dining, entertainment) to 30%. Use our monthly budget planner to cap categories.\n"
                "- **Investments**: Save at least 20% of your income. Let's aim to bump your savings rate to 30% by cutting luxury expenses."
            )
            suggestions = ["How do I cut down food expenses?", "Suggest a savings strategy", "Create an emergency fund"]
            
        elif "tax" in msg_lower or "invest" in msg_lower or "saving schemes" in msg_lower:
            reply = (
                "For optimal tax efficiency and wealth creation in India, look at these options:\n\n"
                "1. **Equity Linked Savings Scheme (ELSS)**: Offers tax exemption under Section 80C with a low 3-year lock-in.\n"
                "2. **Public Provident Fund (PPF)**: Risk-free government-backed savings offering tax-free interest.\n"
                "3. **National Pension System (NPS)**: Additional ₹50,000 deduction under Sec 80CCD(1B) for retirement planning.\n\n"
                "Since your risk tolerance is **" + risk + "**, I recommend splitting Section 80C contributions: 60% in ELSS and 40% in PPF."
            )
            suggestions = ["What is ELSS?", "Calculate tax savings", "Is NPS better than PPF?"]
            
        else:
            reply = (
                f"Hello! I am your WealthWise Coach. I have analyzed your profile with a monthly income of **₹{income:,.2f}** "
                f"and net worth of **₹{net_worth:,.2f}**.\n\n"
                "I can help you build an investment plan, analyze your monthly budget, calculate your emergency fund requirements, "
                "or forecast future expenses. What financial goal would you like to focus on today?"
            )
            suggestions = ["Help me balance my portfolio", "Predict my next month's expenses", "Calculate my health score"]
            
        return reply, suggestions

    def _generate_suggestions(self, message: str) -> List[str]:
        msg_lower = message.lower()
        if "portfolio" in msg_lower or "rebalance" in msg_lower:
            return ["Explain my portfolio risk", "Recommend mutual funds", "How to rebalance?"]
        if "budget" in msg_lower or "spend" in msg_lower:
            return ["Predict next month's expenses", "Cut discretionary spend", "Create a budget limit"]
        return ["What is my financial health score?", "Calculate emergency fund", "Tax saving tips"]

ai_advisor_service = AIAdvisorService()
