"""CQ regeneration from human review feedback.

Handles only the LLM step of the two actions rewrite/reanswer; choosing which draft to
regenerate and loading the results are done by the caller (the router).
"""

from typing import cast

from langchain_core.messages import HumanMessage, SystemMessage

from config import settings
from src.client import refiner_llm
from src.models import HumanRefineResult, ReanswerResult
from src.prompt import build_human_refine_user_prompt, load_prompt

rewrite_llm = refiner_llm.with_structured_output(
    HumanRefineResult,
    method="json_mode",  # type: ignore[arg-type]
).with_retry(stop_after_attempt=3, wait_exponential_jitter=True)


reanswer_llm = refiner_llm.with_structured_output(
    ReanswerResult,
    method="json_mode",  # type: ignore[arg-type]
).with_retry(stop_after_attempt=3, wait_exponential_jitter=True)


def needs_regeneration(action: str | None) -> bool:
    """Returns False for None (unreviewed) and accept/drop: they are not regenerated."""
    return action == "rewrite" or action == "reanswer"


def rewrite(
    question: str,
    answer: str,
    feedback: str,
    paragraph_text: str,
) -> HumanRefineResult:
    """Fixes the question per the feedback, and the answer only as far as needed.

    The answer is changed only as much as the fixed question requires.
    """
    messages = [
        SystemMessage(
            content=load_prompt(settings.cq_human_refine_rewrite_prompt_path)
        ),
        HumanMessage(
            content=build_human_refine_user_prompt(
                question=question,
                answer=answer,
                feedback=feedback,
                paragraph_text=paragraph_text,
            )
        ),
    ]
    return cast(HumanRefineResult, rewrite_llm.invoke(messages))


def reanswer(
    question: str,
    answer: str,
    feedback: str,
    paragraph_text: str,
) -> HumanRefineResult:
    """Fixes only the answer per the feedback; the original question is kept as-is."""
    messages = [
        SystemMessage(
            content=load_prompt(settings.cq_human_refine_reanswer_prompt_path)
        ),
        HumanMessage(
            content=build_human_refine_user_prompt(
                question=question,
                answer=answer,
                feedback=feedback,
                paragraph_text=paragraph_text,
            )
        ),
    ]
    result = cast(ReanswerResult, reanswer_llm.invoke(messages))
    return HumanRefineResult(question=question, answer=result.answer)
