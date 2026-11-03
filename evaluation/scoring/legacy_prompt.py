"""Prompt extracted verbatim from evaluation(1).py; no credentials.

Its short-answer routing is selectable in judge_answers.py.
"""
def legacy_judge_prompt(question: str, golden_answer: str, generated_answer: str) -> str:
    """
    IMPORTANT: LLM must *extract the correct answer-type*, not random trailing tokens.
    normalized_ans must be <= 3 words or "HALLUCINATED".
    """
    return f"""You are evaluating a QA model output.

Task:
1) Extract the intended answer to the question from the MODEL OUTPUT.
   Return it as normalized_ans (<= 3 words).
2) Decide if normalized_ans matches ANY acceptable gold variant (gold variants are comma-separated).

You MUST be type-aware:
- If question asks "In which year" -> normalized_ans must be a year/number if present.
- If question asks "Which country" -> normalized_ans must be a country name.
- If question asks "Where" -> normalized_ans must be a location/region.
- If question asks "Which season" -> normalized_ans must be a season (Winter/Spring/Summer/Autumn/Fall).
- If model output does not provide a meaningful answer of the required type, output normalized_ans="HALLUCINATED".

Do NOT output random words like "pact", "understanding", "December" unless the question asks for them.
If the model output contains a different year/country/etc than the gold, extract THAT (so we can mark incorrect properly).

Return ONLY valid JSON with EXACT keys:
- "normalized_ans": string (<=3 words OR "HALLUCINATED")
- "is_correct": true/false
- "verdict": one of ["CORRECT","INCORRECT","NO_ANSWER"]

Rules:
- If normalized_ans == "HALLUCINATED" -> verdict="NO_ANSWER" and is_correct=false.
- Otherwise, compare normalized_ans to gold variants using strict semantic matching.
  Ignore punctuation/case. Do not be lenient.

Question:
{question}

Gold answer variants (comma-separated):
{golden_answer}

MODEL OUTPUT:
{generated_answer}
"""
