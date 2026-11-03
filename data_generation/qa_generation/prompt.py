"""QA prompt preserved verbatim, including the original 1-2 word requirement."""
def build_prompt(summary_text, question_type):
    return f"""
    You are an expert at creating question-answer pairs for a benchmark. Your task is to generate a question that is both **simple and unambiguously answerable** only from the provided text.

    **RULE 1: The Question Must Be SIMPLE and FACTUAL.**
    * The structure of the question should be a direct query. Focus on "Who", "What", "Where", "When".
    * AVOID "Why" or "How" questions that require complex reasoning.

    **RULE 2: The Question Must Be UNAMBIGUOUSLY ANCHORED.**
    * To prevent ambiguity in a large dataset, the question **MUST include a second piece of context** from the text to act as an "anchor".
    * This anchor should be another entity (like a town or place), a specific action (like "proposing a pact"), or a season (NOT YEAR, BECAUSE YEAR WILL BE EASY FOR THE SYSTEM TO DO SIMPLE KEYWORD SEARCH). This ensures the question can only be answered by this specific document.

    * **Bad Example (Too Vague):**
        -   Summary: "Gabon extended an invitation to Congo, proposing the formation of a robust coalition to secure crucial supplying centers across the Mediterranean Sea."
        -   Vague Question: "Which town proposed a coalition?" (Many countries could do this).
    * **Bad Example (Year Specified, hence it is too easy, do not use year as an anchor or identifier in the question):**
        -   Question: Which season did Aberystwyth announce its intention to advance into Munich in 8065 BC?

    * **Good Example (Simple and Anchored):**
        -   Summary: "Gabon extended an invitation to Congo, proposing the formation of a robust coalition to secure crucial supplying centers across the Mediterranean Sea."
        -   Anchored Question: "Which town proposed a coalition to Congo to secure the Mediterranean Sea?"
        -   Answer: Gabon

    **RULE 3: The Answer Must Be EXTREMELY CONCISE.**
    * The answer must be a single entity (name, place, etc.) extracted directly from the text.
    * **STRICT LIMIT: One to two words maximum.**

    ---
    **YOUR TASK:**
    Generate one **simple but anchored** question-answer pair from the summary below.

    **Additional Instruction:** The question must begin with a phrase like "{question_type}".

    **Summary:** "{summary_text}"

    **Format your output strictly as follows:**
    QUESTION: [Your simple and anchored question here]
    ANSWER: [Your 1-2 word answer here]
    """
