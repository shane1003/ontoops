import logging
from pathlib import Path

from config import settings
from src.schemas import DraftCQ


def load_prompt(path: Path) -> str:
    return path.read_text(encoding="utf-8")


def build_section(title: str, content: str) -> str:
    return f"## {title}\n{content}"


def build_cq_user_prompt(
    purpose: str,
    scope: str,
    document_text: str,
    doc_char_limit: int = settings.cq_generator_document_content_char_limit,
) -> str:
    if len(document_text) > doc_char_limit:
        logging.warning(
            f"[generate] Document exceeds char limit, truncated: "
            f"{len(document_text)} > {doc_char_limit} chars "
            f"({len(document_text) - doc_char_limit} chars lost)"
        )

    sections = [
        build_section("Ontology Purpose", purpose),
        build_section("Ontology Scope", scope),
        build_section("Document", document_text[:doc_char_limit]),
    ]
    return "\n\n".join(sections)


def build_terminologizer_split_user_prompt(item: DraftCQ, reason: str = "") -> str:
    sections = [
        build_section("Competency Question", f"Q: {item.question}\nA: {item.answer}")
    ]
    # If a reason (the concepts verify listed) exists, inject it as the decomposition
    # seed. Otherwise decompose from the answer alone.
    if reason:
        sections.append(build_section("Decomposition hint", reason))
    return "\n\n".join(sections)


def build_ill_posed_refine_user_prompt(
    item: DraftCQ,
    reason: str,
    document_text: str,
    doc_char_limit: int = settings.cq_generator_document_content_char_limit,
) -> str:
    sections = [
        build_section("Document", document_text[:doc_char_limit]),
        build_section(
            "Ill-posed question to rephrase",
            f"Q: {item.question}\nA: {item.answer}\nReason: {reason}",
        ),
    ]
    return "\n\n".join(sections)


def build_human_refine_user_prompt(
    question: str,
    answer: str,
    feedback: str,
    paragraph_text: str,
    doc_char_limit: int = settings.cq_generator_document_content_char_limit,
) -> str:
    """Builds the user prompt for regeneration from human feedback.

    The source paragraph is always included: the answer must be a single fact the
    paragraph actually states, and given only the feedback without the paragraph, the
    model can only fill it with what it knows.

    The instruction (rewrite or reanswer) is not here. The system prompt, split per
    action, handles that.
    """
    sections = [
        build_section("Source paragraph", paragraph_text[:doc_char_limit]),
        build_section("Competency question", f"Q: {question}\nA: {answer}"),
        build_section("Reviewer feedback", feedback or "(none given)"),
    ]
    return "\n\n".join(sections)


def build_odl_identify_user_prompt(confirmed: str, window: str, i: int, n: int) -> str:
    """Builds the user prompt for one window.

    Prepending the confirmed part is the key: without it the window does not know
    "which division is open right now" and mistakes the largest heading inside the
    window for the top level."""
    return (
        f"CONFIRMED SO FAR\n{confirmed or '(none yet)'}\n---\nWINDOW {i}/{n}\n{window}"
    )
