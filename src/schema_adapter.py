"""Map a released RefusalBench-NQ row to our internal example format.

Released HF fields -> internal fields. Nothing is invented: every internal
field is either copied from the dataset or trivially derived (is_answerable).
"""

ANSWER_LABEL = "ANSWER_CORRECTLY"

REFUSAL_CODES = [
    "REFUSE_AMBIGUOUS_QUERY",
    "REFUSE_CONTRADICTORY_CONTEXT",
    "REFUSE_INFO_MISSING_IN_CONTEXT",
    "REFUSE_FALSE_PREMISE_IN_QUERY",
    "REFUSE_GRANULARITY_MISMATCH",
    "REFUSE_NONFACTUAL_QUERY",
    "REFUSE_OTHER",
]


def adapt_row(row: dict) -> dict:
    expected = row["expected_rag_behavior"]
    if expected != ANSWER_LABEL and expected not in REFUSAL_CODES:
        raise ValueError(f"Unknown expected_rag_behavior: {expected!r} (id={row['id']})")
    return {
        "id": row["id"],
        "source_id": row["source_id"],
        "question": row["perturbed_query"],
        "context": row["perturbed_context"],
        "expected_behavior": expected,
        "is_answerable": expected == ANSWER_LABEL,
        "uncertainty_type": row["perturbation_class"],
        "intensity": row["intensity"],
        # Only meaningful for answerable rows (the original script also only uses it then).
        "reference_answers": list(row["original_answers"]) if expected == ANSWER_LABEL else None,
        "generator_model": row["generator_model"],
    }
