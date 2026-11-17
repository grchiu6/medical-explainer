"""Retrieval query builder with a leakage guard.

The query is built only from the question and option texts. It never reads
``answer_letter`` or any other field, so retrieval cannot depend on which
answer is correct.
"""


def build_query(item: dict) -> str:
    """Return the question followed by option texts, one per line, in letter order.

    Option labels (A., B., ...) are omitted. Only ``question`` and ``options``
    are read; all other keys, including ``answer_letter``, are ignored.
    """
    options = item["options"]
    return item["question"] + "\n" + "\n".join(options[l] for l in sorted(options))
