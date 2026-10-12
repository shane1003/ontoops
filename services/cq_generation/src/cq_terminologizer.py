"""CQ Terminologizer: generates ontology terms (classes/object/data properties).

Design:
- **The graph is acyclic per item + a conditional edge.** It branches from `START` via
  `gate_composite`: a CQ with a `composite_answer` flag goes to `terminologize_split`
  (reason-seeded multi-concept decomposition), all others (including null) go to
  `terminologize_simple` (single-concept extraction). Both nodes extract terms from the
  Q/A only, assemble a `TerminologizedCQ` and finish (single pass).
- **The function (`terminologize_cq`) takes one CQ** and runs the graph once. CQs do not
  depend on each other, so there is no reason to batch them; distributing many and
  their concurrency is the caller's (orchestrator's) job.
- A failed/None single LLM call is fail-open: the original Q/A is kept in a
  `TerminologizedCQ` with empty term lists. Because empty lists are valid by contract, a
  silent failure is indistinguishable from normal output, so it must always be logged.
- Term extraction is Q/A-only (no document injected). value is derived from the answer.
  The split branch takes the flag.reason set by verify as the decomposition seed; if
  reason is empty, it decomposes from the answer alone.
"""

import logging
from typing import cast

from langchain_core.messages import HumanMessage, SystemMessage
from langgraph.graph import END, START, StateGraph
from pydantic import BaseModel

from config import settings
from src.client import terminologizer_llm
from src.models import OntologyTerms, TerminologizedCQ
from src.prompt import (
    build_section,
    build_terminologizer_split_user_prompt,
    load_prompt,
)
from src.schemas import DraftCQ

structured_terminologizer_llm = terminologizer_llm.with_structured_output(
    OntologyTerms,
    method="json_mode",  # type: ignore[arg-type]  # supported at runtime (stub missing)
).with_retry(stop_after_attempt=3, wait_exponential_jitter=True)


class TerminologizerState(BaseModel):
    """Per-item graph state: one pass of term extraction for one CQ.

    Attributes:
        input_cq: The CQ to extract from. Takes refine's output `DraftCQ` (with flag)
            and routes the branch by the composite_answer flag.
        output_cq: The terms + Q/A emitted at the end.
    """

    input_cq: DraftCQ
    output_cq: TerminologizedCQ | None = None


def _terminologized_cq(
    cq: DraftCQ, terms: OntologyTerms | None = None
) -> TerminologizedCQ:
    """Wraps the original Q/A with terms; no terms gives an empty output (fail-open)."""
    terms = terms or OntologyTerms()
    return TerminologizedCQ(
        question=cq.question,
        answer=cq.answer,
        classes=terms.classes,
        object_properties=terms.object_properties,
        data_properties=terms.data_properties,
    )


def _extract(
    cq: DraftCQ, system_prompt: str, user_content: str, run_name: str
) -> TerminologizedCQ:
    """Single structured call + fail-open assembly, shared by both branches."""
    messages = [
        SystemMessage(content=system_prompt),
        HumanMessage(content=user_content),
    ]
    try:
        terms = structured_terminologizer_llm.invoke(
            messages, config={"run_name": run_name}
        )
    except Exception as e:
        # Includes exhausted validation retries. Keep with empty terms, but log it so
        # it is observable.
        logging.warning(
            f"[{run_name}] single call failed (fail-open=empty terms) cq_id={cq.id}: "
            f"{e.__class__.__name__}: {e}"
        )
        return _terminologized_cq(cq)
    if terms is None:
        # Structured output can come back as None instead of an exception (plain-text
        # response without a tool call, truncation, etc.). with_retry retries only
        # exceptions, so None passes through -> catch it here observably.
        logging.warning(
            f"[{run_name}] no structured output (fail-open=empty terms) cq_id={cq.id}"
        )
        return _terminologized_cq(cq)
    return _terminologized_cq(cq, cast(OntologyTerms, terms))


def terminologize_simple(state: TerminologizerState) -> TerminologizerState:
    cq = state.input_cq
    state.output_cq = _extract(
        cq,
        load_prompt(settings.cq_terminologizer_system_prompt_path),
        build_section("Competency Question", f"Q: {cq.question}\nA: {cq.answer}"),
        "terminologize_simple",
    )
    return state


def terminologize_split(state: TerminologizerState) -> TerminologizerState:
    cq = state.input_cq
    reason = cq.flag.reason if cq.flag else ""
    state.output_cq = _extract(
        cq,
        load_prompt(settings.cq_terminologizer_split_system_prompt_path),
        build_terminologizer_split_user_prompt(cq, reason),
        "terminologize_split",
    )
    return state


def gate_composite(state: TerminologizerState) -> str:
    """Routes composite_answer to split, all else (incl. null) to simple (hard gate)."""
    flag = state.input_cq.flag
    if flag and flag.check_name == "composite_answer":
        return "split"
    return "simple"


_builder = StateGraph(TerminologizerState)
_builder.add_node("terminologize_simple", terminologize_simple)
_builder.add_node("terminologize_split", terminologize_split)
_builder.add_conditional_edges(
    START,
    gate_composite,
    {"split": "terminologize_split", "simple": "terminologize_simple"},
)
_builder.add_edge("terminologize_simple", END)
_builder.add_edge("terminologize_split", END)
graph = _builder.compile()


def terminologize_cq(
    cq: DraftCQ,
    callbacks: list | None = None,
) -> TerminologizedCQ:
    """Converts one CQ into an enriched TerminologizedCQ (single pass, no re-feeding).

    One CQ at a time because CQs do not depend on each other: running many at once is a
    fan-out, not the meaning of this step, so distributing them and their concurrency
    is up to the caller (orchestrator).
    """
    try:
        result = TerminologizerState.model_validate(
            # LangGraph returns a dict even for a pydantic state -> restore it to
            # TerminologizerState.
            graph.invoke(
                TerminologizerState(input_cq=cq),
                config={"callbacks": callbacks or []},
            )
        )
        made = result.output_cq or _terminologized_cq(cq)
    except Exception as e:
        # graph run failed -> keep the original (fail-open).
        logging.warning(
            f"[terminologizer] graph run failed (fail-open=empty terms) cq_id={cq.id}: "
            f"{e.__class__.__name__}: {e}"
        )
        made = _terminologized_cq(cq)

    # An empty output is valid by contract and indistinguishable from normal output, so
    # it must always be logged.
    if made.is_empty:
        logging.warning(f"[terminologizer] cq_id={cq.id} produced 0 terms.")
    return made
