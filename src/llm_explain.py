"""
Natural-language adverse-action reasons, grounded in SHAP.

Regulations (e.g. ECOA/Reg B in the US) require that a declined applicant be
given the specific reasons for the decision. This module turns the structured
SHAP drivers from explain.py into a plain-language explanation.

The critical design constraint is grounding: the LLM is only allowed to
describe the exact features SHAP surfaced. A validation step checks the
generated text against the permitted feature descriptions and rejects any
explanation that appears to introduce factors SHAP did not identify. This
prevents the well-known failure mode where an LLM invents plausible-sounding
but unsupported reasons for a credit decision.
"""

import os

try:
    from src.explain import FEATURE_DESCRIPTIONS
except ModuleNotFoundError:
    from explain import FEATURE_DESCRIPTIONS


SYSTEM_PROMPT = """You are a compliance assistant that writes adverse-action \
reason statements for declined loan applications.

You will be given a list of factors that a credit risk model identified as \
increasing the applicant's default risk. Write a brief, professional \
explanation (2-3 sentences) suitable for an adverse-action notice.

STRICT RULES:
- Only reference the exact factors provided. Do not introduce any factor that \
is not in the list.
- Do not speculate about the applicant's character, race, gender, or anything \
not in the provided factors.
- Do not invent numbers or thresholds.
- Be factual and neutral in tone."""


def _format_drivers_for_prompt(drivers) -> str:
    """Render the risk-increasing drivers as a bullet list for the prompt."""
    risk_drivers = [d for d in drivers if d["direction"] == "increases_risk"]
    if not risk_drivers:
        return "(No risk-increasing factors — applicant was not declined on these grounds.)"
    lines = [f"- {d['description']} (value: {d['value']})" for d in risk_drivers]
    return "\n".join(lines)


def generate_reason(drivers, client=None) -> str:
    """
    Call the LLM to produce a grounded adverse-action reason.

    `client` is an anthropic.Anthropic instance; if None, one is created from
    the ANTHROPIC_API_KEY environment variable.
    """
    if client is None:
        import anthropic
        client = anthropic.Anthropic(api_key=os.environ["ANTHROPIC_API_KEY"])

    driver_text = _format_drivers_for_prompt(drivers)
    user_message = (
        "The model identified the following risk-increasing factors:\n\n"
        f"{driver_text}\n\n"
        "Write the adverse-action reason statement."
    )

    response = client.messages.create(
        model="claude-sonnet-4-6",
        max_tokens=300,
        system=SYSTEM_PROMPT,
        messages=[{"role": "user", "content": user_message}],
    )
    return response.content[0].text.strip()


def validate_grounding(reason_text: str, drivers) -> dict:
    """
    Check that the generated reason only references permitted feature concepts.

    Each feature has a set of distinctive keywords. A feature that is NOT among
    the drivers is "forbidden"; if the reason text contains a forbidden
    feature's distinctive keyword (and none of the permitted features legitimately
    explain that keyword), grounding is flagged as violated. In a real system a
    violation would block the explanation from being sent.
    """
    # Distinctive keywords per feature — chosen to avoid substring collisions
    # (e.g. "grade" would match inside "sub-grade", so grade uses a phrase).
    FEATURE_KEYWORDS = {
        "loan_amnt": ["loan amount", "requested amount"],
        "int_rate": ["interest rate"],
        "installment": ["installment"],
        "annual_inc": ["annual income", "income level"],
        "dti": ["debt-to-income", "debt to income"],
        "delinq_2yrs": ["delinquenc"],
        "open_acc": ["open credit", "open account"],
        "pub_rec": ["public record", "derogatory"],
        "revol_bal": ["revolving balance"],
        "revol_util": ["utilization", "revolving line"],
        "total_acc": ["total credit", "total number of credit"],
        "emp_length_years": ["employment", "employed"],
        "credit_history_months": ["credit history", "history length"],
        "term": ["loan term", "-month term", "month term"],
        "grade": ["loan grade", "assigned grade"],
        "sub_grade": ["sub-grade", "subgrade"],
        "home_ownership": ["home ownership", "homeowner"],
        "verification_status": ["verification"],
        "purpose": ["purpose of the loan", "loan purpose"],
        "addr_state": ["state of residence", "residence state"],
    }

    permitted_features = {d["feature"] for d in drivers}
    text_lower = reason_text.lower()

    violations = []
    for feat, keywords in FEATURE_KEYWORDS.items():
        if feat in permitted_features:
            continue
        if any(kw in text_lower for kw in keywords):
            violations.append(FEATURE_DESCRIPTIONS.get(feat, feat))

    return {
        "grounded": len(violations) == 0,
        "violations": violations,
    }


if __name__ == "__main__":
    # Offline demo: use a mock client so this runs without an API key.
    try:
        from src.model import build_split, train_model
        from src.explain import build_explainer, top_drivers
    except ModuleNotFoundError:
        from model import build_split, train_model
        from explain import build_explainer, top_drivers

    (X_train, y_train), (X_val, y_val), (X_test, y_test) = build_split()
    model = train_model(X_train, y_train, X_val, y_val)
    explainer = build_explainer(model)

    # Find a declined-style applicant (high predicted PD)
    import numpy as np
    probs = model.predict_proba(X_test)[:, 1]
    idx = int(np.argmax(probs))
    drivers = top_drivers(explainer, X_test.iloc[[idx]])

    print(f"Applicant {idx} — predicted default probability: {probs[idx]:.1%}")
    print("\nRisk-increasing drivers (the only permitted basis for the reason):")
    print(_format_drivers_for_prompt(drivers))

    if os.environ.get("ANTHROPIC_API_KEY"):
        reason = generate_reason(drivers)
        print("\nGenerated adverse-action reason:")
        print(reason)
        check = validate_grounding(reason, drivers)
        print(f"\nGrounding check: {'PASSED' if check['grounded'] else 'FAILED'}")
        if check["violations"]:
            print(f"Ungrounded concepts detected: {check['violations']}")
    else:
        print("\n[Set ANTHROPIC_API_KEY to generate and validate a real reason.]")
