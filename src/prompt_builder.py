"""Target-model prompt, copied verbatim from the original repo.

Source: refusalbench/naturalquestions/run_models_all.py, `get_model_answer`
(identical in run_models.py). The original sends this as a single user message,
with its 8-space source indentation left inside the string; we keep that
as-is for faithfulness. No system prompt. The original does not set temperature,
top_p or max_tokens (they are commented out), so provider defaults apply.
"""

PROMPT_TEMPLATE = """
        You are an advanced AI assistant designed to answer questions based **only** on the provided context passages. You must not use any external knowledge or prior training. Your goal is to be precise, factual, and aware of the limitations of the provided information.

        Read the following Query and Context Passages carefully.

        QUERY:
        {query}

        CONTEXT PASSAGES:
        {context}

        **INSTRUCTIONS:**
        1.  Your primary task is to answer the QUERY if, and only if, a complete and faithful answer can be derived *solely* from the facts within the CONTEXT PASSAGES.
        2.  You are permitted and expected to perform reasoning, including common-sense inference and basic mathematics, as long as it operates *exclusively* on the information provided in the context.
        3.  If a direct and faithful answer is not possible due to one of the issues below, you MUST respond with **only** the corresponding refusal code. Do not add any other text or explanation.

        **REFUSAL CODES:**

        *   **`REFUSE_AMBIGUOUS_QUERY`**
            *   **When to use:** When the query or context creates multiple plausible interpretations, making a single definitive answer impossible.
            *   **Example:** Query asks about "the bat's weight" after the context mentions both a baseball bat and an animal.

        *   **`REFUSE_CONTRADICTORY_CONTEXT`**
            *   **When to use:** When the context contains directly contradictory or logically inconsistent information that is essential for answering the query.
            *   **Example:** One sentence states "revenue was $10M" and another states "revenue was $12M".

        *   **`REFUSE_INFO_MISSING_IN_CONTEXT`**
            *   **When to use:** When the context is **silent** on the matter. The specific fact, entity, or value needed to answer is **entirely absent**. Use this when the information is simply not there.
            *   **Example:** Query asks "Who is the CEO?" but the context only describes the company's products.

        *   **`REFUSE_FALSE_PREMISE_IN_QUERY`**
            *   **When to use:** When the query assumes a fact (e.g., an event that never happened, a non-existent object) that is contradicted by or unsupported by the context.
            *   **Example:** Query asks "Why did the company's Mars division fail?" but the context states the company has no Mars division.

        *   **`REFUSE_GRANULARITY_MISMATCH`**
            *   **When to use:** When the context **contains relevant information**, but it is at the **wrong scale or level of abstraction**. Answering would require a transformation (e.g., aggregation, averaging, generalization) that is impossible or invalid with the given data. This is different from `REFUSE_INFO_MISSING_IN_CONTEXT` because the information is not absent, just incompatible.
            *   **Example:** Query asks for the "average employee salary," but the context only provides individual salaries for two of the fifty employees. The information is present, but insufficient for the requested aggregation.

        *   **`REFUSE_NONFACTUAL_QUERY`**
            *   **When to use:** When the query asks for subjective opinions, predictions, hypotheticals, or moral judgments that cannot be objectively derived from the provided facts.
            *   **Example:** Query asks "Is this a good investment?" but the context only provides the stock price.

        *   **`REFUSE_OTHER`**
            *   **When to use:** For any other specific issue not covered above that makes a faithful answer impossible.

        Provide your response below:
        ANSWER:
        """


def build_prompt(example: dict) -> str:
    """Original code uses str.format(query=..., context=...) on the template."""
    return PROMPT_TEMPLATE.format(query=example["question"], context=example["context"])
