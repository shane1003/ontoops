import logging
from typing import cast

from langchain_core.messages import HumanMessage, SystemMessage
from langgraph.graph import END, StateGraph
from pydantic import BaseModel, Field

from config import settings
from src.client import generator_llm
from src.models import GeneratedCQList
from src.prompt import build_cq_user_prompt, load_prompt
from src.schemas import DraftCQ


class GenerateState(BaseModel):
    """Generation graph state: takes purpose, scope and document and generates CQs."""

    purpose: str
    scope: str
    document_text: str
    cqs: list[DraftCQ] = Field(default_factory=list)


structured_llm = generator_llm.with_structured_output(GeneratedCQList, method="json_mode")  # type: ignore[arg-type]  # supported at runtime (only the type stub is missing)


def generate_node(state: GenerateState) -> GenerateState:
    messages = [
        SystemMessage(content=load_prompt(settings.cq_generator_system_prompt_path)),
        HumanMessage(
            content=build_cq_user_prompt(
                state.purpose, state.scope, state.document_text
            )
        ),
    ]
    try:
        result = cast(GeneratedCQList, structured_llm.invoke(messages))
    except Exception as e:
        logging.error(f"{e.__class__}: {e}")
        raise
    entities = [DraftCQ(**q.model_dump()) for q in result.questions]
    return state.model_copy(update={"cqs": entities})


builder = StateGraph(GenerateState)
builder.add_node("generate", generate_node)
builder.set_entry_point("generate")
builder.add_edge("generate", END)
graph = builder.compile()


def generate_cqs(
    purpose: str,
    scope: str,
    document_text: str,
    callbacks: list | None = None,
) -> list[DraftCQ]:
    """Generate CQ from docuemnt."""
    initial_state = GenerateState(
        purpose=purpose,
        scope=scope,
        document_text=document_text,
    )
    state = graph.invoke(
        initial_state,
        config={"callbacks": callbacks or []},
    )
    return state["cqs"]
