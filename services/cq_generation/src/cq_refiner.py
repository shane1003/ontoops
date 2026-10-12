"""CQ verify-refine pipeline.

The graph is a pure acyclic per-item decision tree.
It takes one CQ, checks it, and if defective repairs it right away (refine_*) and ends.
The loop is handled outside the graph: split outputs come back through the work queue
and run through the graph again.
"""

import logging
from typing import Any, NamedTuple

from langchain_core.messages import HumanMessage, SystemMessage
from langgraph.graph import END, StateGraph
from pydantic import BaseModel, Field

from config import settings
from src.client import refiner_llm, verifier_llm
from src.models import (
    CompositeAnswerVerdict,
    CompositeQuestionVerdict,
    IllPosedVerdict,
    RephraseResult,
    SplitResult,
    VerifyStatus,
)
from src.prompt import (
    build_ill_posed_refine_user_prompt,
    build_section,
    load_prompt,
)
from src.schemas import CheckFlag, DraftCQ

ill_posed_llm = verifier_llm.with_structured_output(
    IllPosedVerdict, method="json_mode"  # type: ignore[arg-type]
).with_retry(stop_after_attempt=3, wait_exponential_jitter=True)
composite_question_llm = verifier_llm.with_structured_output(
    CompositeQuestionVerdict, method="json_mode"  # type: ignore[arg-type]
).with_retry(stop_after_attempt=3, wait_exponential_jitter=True)
composite_answer_llm = verifier_llm.with_structured_output(
    CompositeAnswerVerdict, method="json_mode"  # type: ignore[arg-type]
).with_retry(stop_after_attempt=3, wait_exponential_jitter=True)
split_llm = refiner_llm.with_structured_output(
    SplitResult, method="json_mode"  # type: ignore[arg-type]
).with_retry(stop_after_attempt=3, wait_exponential_jitter=True)
rephrase_llm = refiner_llm.with_structured_output(
    RephraseResult, method="json_mode"  # type: ignore[arg-type]
).with_retry(stop_after_attempt=3, wait_exponential_jitter=True)


class ItemState(BaseModel):
    """Per-item graph state: one verify-refine pass for one CQ.

    Attributes:
        input_cq: The incoming CQ, immutable raw.
        flag: This run's check result (scalar, transient). Written by check, read by
            gate/refine. Stamped onto the DraftCQ in output_cqs only at the end
            (kept/drop).
        document_text: The source document to refer to.
        output_cqs: The result CQs emitted at the end.
        status: Final classification; the function loop branches on it.
        iteration: refine budget counter.
    """

    input_cq: DraftCQ
    flag: CheckFlag | None = None
    # TODO: consider moving to config
    document_text: str = ""
    output_cqs: list[DraftCQ] = Field(default_factory=list)
    status: VerifyStatus = VerifyStatus.UNVERIFIED
    iteration: int = 0


class VerifyCallFailed(Exception):
    """The check (verify) LLM call gave no result (exception or no structured output).

    Why not fail open by returning None: the verdict models have no "cannot judge"
    channel (is_ill_posed/is_composite are required bools), so None can only mean a
    fault; if the caller read it as "no defect", a CQ that was never checked would be
    stored as VERIFIED.
    """


# Not merged with _refine despite the identical body: the meaning of failure and the
# exception type differ.
def _judge(llm: Any, system_prompt: str, user: str, run_name: str) -> Any:
    """Never returns None; failure raises VerifyCallFailed."""
    try:
        result = llm.invoke(
            [SystemMessage(content=system_prompt), HumanMessage(content=user)],
            config={"run_name": run_name},
        )
    except Exception as e:
        logging.warning(f"[{run_name}] single call failed: {e.__class__.__name__}: {e}")
        raise VerifyCallFailed(run_name) from e

    if result is None:
        logging.warning(
            f"[{run_name}] no structured output (None) - treated as parse failure"
        )
        raise VerifyCallFailed(run_name)

    return result


class RefineCallFailed(Exception):
    """The repair (refine) LLM call gave no result (exception or no structured output).

    Why not fail open by returning None: to the caller, None is indistinguishable from
    the *verdict* "cannot be fixed", and the two are handled in opposite ways. A verdict
    rightly deletes the original; a fault must not (a retry gives a different result).
    Raised as an exception, refine_until_converged catches it and returns it in the
    failed bucket, separate from kept.
    """


def _refine(llm: Any, system_prompt: str, user: str, run_name: str) -> Any:
    """Returns the structured result or raises RefineCallFailed; never returns None."""
    try:
        result = llm.invoke(
            [SystemMessage(content=system_prompt), HumanMessage(content=user)],
            config={"run_name": run_name},
        )
    except Exception as e:
        logging.warning(f"[{run_name}] single call failed: {e.__class__.__name__}: {e}")
        raise RefineCallFailed(run_name) from e

    if result is None:
        logging.warning(
            f"[{run_name}] no structured output (None) - treated as parse failure"
        )
        raise RefineCallFailed(run_name)

    return result


# ---------------------------------------------------------------------------
# Check nodes (single item)
# ---------------------------------------------------------------------------


def check_ill_posed(state: ItemState) -> ItemState:
    cq = state.input_cq
    verdict = _judge(
        ill_posed_llm,
        load_prompt(settings.cq_verify_ill_posed_prompt_path),
        build_section("Question and Answer", f"Q: {cq.question}\nA: {cq.answer}"),
        "check_ill_posed",
    )

    if verdict.is_ill_posed:
        state.flag = CheckFlag(check_name="ill_posed", reason=verdict.reason)
        state.status = VerifyStatus.NEED_REFINE

    return state


def check_composite_question(state: ItemState) -> ItemState:
    verdict = _judge(
        composite_question_llm,
        load_prompt(settings.cq_verify_composite_question_prompt_path),
        build_section("Question", state.input_cq.question),
        "check_composite_question",
    )

    if verdict.is_composite:
        state.flag = CheckFlag(check_name="composite_question", reason=verdict.reason)
        state.status = VerifyStatus.NEED_REFINE

    return state


def check_composite_answer(state: ItemState) -> ItemState:
    # composite_answer is a non-blocking note: only set the flag; finalize stamps it
    # onto the output.
    verdict = _judge(
        composite_answer_llm,
        load_prompt(settings.cq_verify_composite_answer_prompt_path),
        build_section("Answer", state.input_cq.answer),
        "check_composite_answer",
    )

    if verdict.is_composite:
        state.flag = CheckFlag(check_name="composite_answer", reason=verdict.reason)

    return state


# ---------------------------------------------------------------------------
# edge, conditional edge
# ---------------------------------------------------------------------------


def gate_ill_posed(state: ItemState) -> str:
    return (
        "refine_ill_posed"
        if state.status == VerifyStatus.NEED_REFINE
        else "check_composite_question"
    )


def gate_composite_question(state: ItemState) -> str:
    return (
        "refine_composite_question"
        if state.status == VerifyStatus.NEED_REFINE
        else "check_composite_answer"
    )


def finalize(state: ItemState) -> ItemState:
    cq = state.input_cq
    state.output_cqs = [
        DraftCQ(
            id=cq.id,
            question=cq.question,
            answer=cq.answer,
            flag=state.flag,
        )
    ]
    state.status = VerifyStatus.VERIFIED
    return state


# ---------------------------------------------------------------------------
# Repair nodes (single item) + drop
# ---------------------------------------------------------------------------


def refine_ill_posed(state: ItemState) -> ItemState:
    assert state.flag is not None  # entered only when gate saw NEED_REFINE.
    cq = state.input_cq
    result = _refine(
        rephrase_llm,
        load_prompt(settings.cq_refine_ill_posed_prompt_path),
        build_ill_posed_refine_user_prompt(
            item=cq,
            reason=state.flag.reason,
            document_text=state.document_text,
        ),
        "refine_ill_posed",
    )

    if result.question is None:
        # The LLM *judged* it cannot be rephrased (faults were filtered out by _refine
        # as exceptions) -> drop, carrying the reason (flag).
        state.output_cqs = [
            DraftCQ(
                id=cq.id,
                question=cq.question,
                answer=cq.answer,
                flag=state.flag,
            )
        ]
        state.status = VerifyStatus.UNREFINED
        return state

    # Keep the id, replace question/answer -> re-feed as a fresh CQ without a flag
    # (full re-verification).
    answer = result.answer if result.answer is not None else cq.answer
    state.output_cqs = [
        DraftCQ(
            id=cq.id,
            question=result.question,
            answer=answer,
        )
    ]
    state.status = VerifyStatus.REFINED
    return state


def refine_composite_question(state: ItemState) -> ItemState:
    cq = state.input_cq
    result = _refine(
        split_llm,
        load_prompt(settings.cq_refine_split_prompt_path),
        build_section(
            "Composite question to split", f"Q: {cq.question}\nA: {cq.answer}"
        ),
        "refine_split",
    )

    if not result.questions:
        # The LLM *judged* it cannot be split (faults were filtered out by _refine as
        # exceptions) -> drop, carrying the reason (flag).
        state.output_cqs = [
            DraftCQ(
                id=cq.id,
                question=cq.question,
                answer=cq.answer,
                flag=state.flag,
            )
        ]
        state.status = VerifyStatus.UNREFINED
        return state

    # Assign new ids (split products) -> re-feed each as a fresh CQ without a flag
    # (full re-verification).
    state.output_cqs = [
        DraftCQ(
            question=q.question,
            answer=q.answer,
        )
        for q in result.questions
    ]
    state.status = VerifyStatus.REFINED
    return state


# ---------------------------------------------------------------------------
# graph (acyclic, multiple ENDs)
# ---------------------------------------------------------------------------

_builder = StateGraph(ItemState)
_builder.add_node("check_ill_posed", check_ill_posed)
_builder.add_node("refine_ill_posed", refine_ill_posed)
_builder.add_node("check_composite_question", check_composite_question)
_builder.add_node("refine_composite_question", refine_composite_question)
_builder.add_node("check_composite_answer", check_composite_answer)
_builder.add_node("finalize", finalize)
_builder.set_entry_point("check_ill_posed")
_builder.add_conditional_edges(
    "check_ill_posed",
    gate_ill_posed,
    {
        "refine_ill_posed": "refine_ill_posed",
        "check_composite_question": "check_composite_question",
    },
)
_builder.add_edge("refine_ill_posed", END)
_builder.add_conditional_edges(
    "check_composite_question",
    gate_composite_question,
    {
        "refine_composite_question": "refine_composite_question",
        "check_composite_answer": "check_composite_answer",
    },
)
_builder.add_edge("refine_composite_question", END)
_builder.add_edge("check_composite_answer", "finalize")
_builder.add_edge("finalize", END)
graph = _builder.compile()


class RefineOutcome(NamedTuple):
    """Result of refining one CQ; only kept is stored.

    Separating failed from dropped is why this type exists: both leave kept empty, but
    they are handled in opposite ways. dropped is a verification *verdict*, so deleting
    the original is right; failed is an LLM fault, so it must not be deleted (a retry
    gives a different result). Returning both merged as an empty list would make the
    caller mistake a fault for a discard and delete the row.
    """

    kept: list[DraftCQ]
    failed: list[DraftCQ]  # LLM call failure / graph exception, not a verdict
    dropped: list[DraftCQ]  # discarded by verification


def refine_until_converged(
    cq: DraftCQ,
    document_text: str,
    callbacks: list | None = None,
    max_work_queue_iteration: int = settings.cq_refine_max_iterations,
) -> RefineOutcome:
    """Runs verify-refine on one CQ until convergence and returns three buckets.

    kept is a list although the input is one CQ because of composite decomposition: one
    CQ can be split into several.
    """
    work_queue: list[ItemState] = [ItemState(input_cq=cq, document_text=document_text)]

    kept: list[DraftCQ] = []
    failed: list[DraftCQ] = []
    dropped: list[DraftCQ] = []

    while work_queue:
        work = work_queue.pop(0)  # The queue holds only a few split products.
        try:
            # LangGraph returns a dict even for a pydantic state -> restore ItemState.
            result = ItemState.model_validate(
                graph.invoke(work, config={"callbacks": callbacks or []})
            )
        except Exception as e:
            # On a fault, keep the original in failed so the caller does not delete it.
            logging.warning(
                f"[refine] run failed (original kept) cq_id={work.input_cq.id}: "
                f"{e.__class__.__name__}: {e}"
            )
            failed.append(DraftCQ(**work.input_cq.model_dump()))
            continue

        if result.status == VerifyStatus.VERIFIED:
            kept.extend(result.output_cqs)
            continue

        # Judged unrepairable (UNREFINED) or budget exhausted -> drop.
        if (
            result.status == VerifyStatus.UNREFINED
            or result.iteration >= max_work_queue_iteration
        ):
            dropped.extend(result.output_cqs)
            continue

        # REFINED -> re-feed with iteration+1 for full re-verification.
        work_queue.extend(
            ItemState(
                input_cq=out,
                document_text=document_text,
                iteration=result.iteration + 1,
            )
            for out in result.output_cqs
        )

    # Discards and failures must not vanish silently: both leave kept empty and look the
    # same to the caller.
    if dropped or failed:
        logging.warning(
            f"[refine] cq_id={cq.id} kept={len(kept)} dropped={len(dropped)} "
            f"failed={len(failed)}"
        )
    return RefineOutcome(kept=kept, failed=failed, dropped=dropped)
