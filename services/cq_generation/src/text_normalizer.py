"""Unicode normalization of LLM output and text extracted from PDFs.

## When it runs in LangChain + Pydantic

The chain inside with_structured_output(GeneratedCQList) has two steps.

`ChatOpenAI  ->  RunnableLambda (Pydantic parsing)`

`@field_validator(mode="before")` runs in the RunnableLambda step, before Pydantic's
type-validation pipeline, and receives the raw input value of each field.

So the order is `JSON parsing -> normalize() [mode="before"] -> Pydantic type
validation -> field storage`.

Example:
```python
class CQCore(BaseModel):
    @field_validator("question", "answer", ..., mode="before")
    def normalize_unicode(cls, v):
        return normalize(v)  # <- takes the raw string, returns it normalized
```

At serialization time (`model_dump()`) the field values are already normalized, so
nothing is reprocessed. Note that bypassing the validator, e.g. with
`model_construct()`, skips normalization.

## Reading Langfuse traces

Langfuse captures both spans, so what you see depends on where you look.

- ChatOpenAI span: raw LLM response, before normalization. U+2011, U+202F etc. show
  as is.
- RunnableLambda span: Pydantic parsing result, after normalization. The value the
  app actually uses.

To see which characters the LLM generated, look at the ChatOpenAI span; to see the
final value the app stores, look at the RunnableLambda span.
"""

import unicodedata

# Replace characters that survive NFKC normalization with their ASCII equivalents
CHAR_SUBSTITUTIONS: dict[str, str] = {
    # Hyphens / dashes
    "‐": "-",  # HYPHEN (intermediate NFKC result)
    "‑": "-",  # NON-BREAKING HYPHEN
    "‒": "-",  # FIGURE DASH
    "–": "-",  # EN DASH
    "—": "-",  # EM DASH
    "­": "",   # SOFT HYPHEN (removed)
    # Quotation marks
    "‘": "'",  # LEFT SINGLE QUOTATION MARK
    "’": "'",  # RIGHT SINGLE QUOTATION MARK
    "“": '"',  # LEFT DOUBLE QUOTATION MARK
    "”": '"',  # RIGHT DOUBLE QUOTATION MARK
    # Spaces (NFKC handles these, kept as an explicit safety net)
    " ": " ",  # NO-BREAK SPACE
}

_TABLE = str.maketrans(CHAR_SUBSTITUTIONS)


def normalize(text: str) -> str:
    text = unicodedata.normalize("NFKC", text)
    return text.translate(_TABLE)
