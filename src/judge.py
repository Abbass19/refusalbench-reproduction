"""LLM judge, prompt copied verbatim from the original repo.

Source: refusalbench/naturalquestions/run_models_all.py, `classify_and_evaluate_response`
(paper Appendix I.1.4). Original judge: Claude Sonnet 4 on Bedrock. Ours is a documented
substitute (Gemini). The parsing logic mirrors the original, including its fallbacks.
"""

from .schema_adapter import REFUSAL_CODES


def build_judge_prompt(query: str, model_output: str, reference_answers) -> str:
    if reference_answers:
        if isinstance(reference_answers, list):
            formatted_correct = "\n".join(str(a) for a in reference_answers if a)
        else:
            formatted_correct = str(reference_answers)
    else:
        formatted_correct = "[No reference answers provided - this may be an unanswerable instance]"

    return f"""Analyze the following model response by performing BOTH classification and quality evaluation:

STEP 1: CLASSIFICATION
Determine if this model response is:
1. An attempt to answer the query → "answer_attempt"
2. A refusal with a specific code:
   - REFUSE_AMBIGUOUS_QUERY
   - REFUSE_CONTRADICTORY_CONTEXT
   - REFUSE_INFO_MISSING_IN_CONTEXT
   - REFUSE_FALSE_PREMISE_IN_QUERY
   - REFUSE_GRANULARITY_MISMATCH
   - REFUSE_NONFACTUAL_QUERY
   - REFUSE_OTHER

Look for refusal codes even if they appear with additional text.

STEP 2: QUALITY EVALUATION
If the response is an answer attempt AND reference answers are provided, evaluate quality:
- Score 1: Not relevant to query and reference answers
- Score 2: According to reference answers but not relevant to query
- Score 3: Relevant to query and reference answers but contains mistakes
- Score 4: Relevant and has exact same metrics as reference, but not concise
- Score 5: Relevant to query and fully correct according to reference answers

If the response is a refusal OR no reference answers provided, use "N/A".

QUERY:
{query}

MODEL RESPONSE:
{model_output}

REFERENCE ANSWERS:
{formatted_correct}

Respond in this exact format:
CLASSIFICATION: [answer_attempt OR one of the REFUSE_* codes]
QUALITY_SCORE: [1-5 if answer_attempt with references, otherwise N/A]
EXPLANATION: [brief reasoning for both classification and score]"""


def parse_judge_response(response: str):
    """Returns (classification, quality_score, explanation, parse_status).

    Same behavior as the original: on failure fall back to scanning for a refusal
    code, else default to answer_attempt. We additionally report parse_status so
    fallbacks are visible and never silent.
    """
    try:
        lines = response.strip().split("\n")
        cls_line = [l for l in lines if l.startswith("CLASSIFICATION:")][0]
        classification = cls_line.split("CLASSIFICATION:")[1].strip()
        score_line = [l for l in lines if l.startswith("QUALITY_SCORE:")][0]
        score_text = score_line.split("QUALITY_SCORE:")[1].strip()
        quality = None if score_text == "N/A" else float(score_text)
        expl = [l for l in lines if l.startswith("EXPLANATION:")]
        explanation = expl[0].split("EXPLANATION:")[1].strip() if expl else ""
        return classification, quality, explanation, "ok"
    except Exception as e:
        upper = response.upper()
        for code in REFUSAL_CODES:
            if code in upper:
                return code, None, "fallback: refusal code found in text", "fallback_code"
        return "answer_attempt", None, f"fallback: could not parse ({e})", "fallback_default"
