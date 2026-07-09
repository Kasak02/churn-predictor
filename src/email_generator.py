# src/email_generator.py
# Direct Groq SDK — no LangChain needed for single prompt

import os
from groq import Groq
from dotenv import load_dotenv

load_dotenv()

def generate_retention_email(churn_prob, risk_level, top_features):
    """
    Generate personalised retention email using Groq API directly.

    Args:
        churn_prob   : float — churn probability 0-100
        risk_level   : str   — High / Medium / Low
        top_features : list  — list of dicts with feature + shap_value

    Returns:
        str — generated email text
    """
    # Format top SHAP features as readable text
    risk_factors = []
    for f in top_features[:3]:
        if f['shap_value'] > 0:
            risk_factors.append(
                f"{f['feature'].replace('_', ' ')} "
                f"(SHAP impact: +{f['shap_value']:.3f})"
            )
    risk_text = ', '.join(risk_factors) if risk_factors \
                else "low platform engagement"

    # Build the prompt
    prompt = f"""
You are a Customer Success Manager at a B2B SaaS company.
Write a personalised retention email for a customer showing signs of disengagement.

Customer risk profile:
- Churn probability : {round(churn_prob, 1)}%
- Risk level        : {risk_level}
- Key risk factors  : {risk_text}

STRICT RULES:
1. NEVER use the words churn, leaving, cancel, or at risk
2. Frame everything positively — you are reaching out to ADD value
3. Reference their specific situation naturally
4. Offer ONE clear next step (a call, a guide, or a feature demo)
5. Keep it under 120 words
6. Be warm, human, professional — not robotic or salesy
7. First line must be: Subject: [your subject line]
8. Then blank line, then email body

Write the email now:
"""

    # Call Groq API directly
    client   = Groq(api_key=os.getenv("GROQ_API_KEY"))
    response = client.chat.completions.create(
        model    = "llama-3.3-70b-versatile",
        messages = [{"role": "user", "content": prompt}],
        temperature = 0.7,
        max_tokens  = 300
    )

    return response.choices[0].message.content.strip()