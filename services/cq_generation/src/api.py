import json
import logging
from collections.abc import AsyncGenerator
from contextlib import asynccontextmanager
from pathlib import Path
from typing import NoReturn
from urllib.parse import quote

import uvicorn
from fastapi import Body, FastAPI, File, HTTPException, UploadFile
from fastapi.param_functions import Query
from fastapi.responses import Response
from langchain_core.exceptions import OutputParserException
from langfuse.langchain import CallbackHandler
from pydantic import BaseModel, ValidationError

from config import settings
from src.cq_terminologizer import terminologize_cq
from src.cq_generator import generate_cqs
from src.cq_human_refiner import needs_regeneration, reanswer, rewrite
from src.cq_refiner import refine_until_converged
from src.db import (
    MissingReference,
    create_tables,
    load_cq,
    load_draft,
    load_drafts_by_run,
    load_paragraph,
    load_promoted_ids,
    promote_drafts,
    replace_draft,
    save_drafts,
    save_paragraphs,
    update_draft_regenerated,
    update_draft_review,
    update_draft_verdict,
)
from src.mcs_chunker import chunk_tree
from src.odl_converter import ConversionFailed, ConversionTimeout, convert
from src.odl_structure import (
    Structure,
    StructureLLMError,
    build_structure,
    validate_uploaded,
)
from src.odl_tree import parse_odl_to_tree
from src.schemas import (
    CQ,
    CheckFlag,
    CQDraftRow,
    CQRef,
    CQsCreated,
    CQsRefined,
    DraftCQ,
    ParagraphsCreated,
    RegenerationVerdict,
    ReviewerAction,
)

@asynccontextmanager
async def lifespan(app: FastAPI) -> AsyncGenerator[None]:
    """FastAPI lifespan."""
    try:
        create_tables()
    except Exception as e:
        raise RuntimeError(
            f"Cannot start: failed to connect to postgres. Check POSTGRES_DSN"
            f" ({settings.postgres_dsn.rsplit('@', 1)[-1]})."
            f" Cause: {e.__class__.__name__}: {e}"
        ) from e
    yield


app = FastAPI(title="Ontology CQ Generator", lifespan=lifespan)


def read_pdf_upload(file: UploadFile) -> bytes:
    max_bytes = settings.cq_max_pdf_bytes
    if not file.size:
        raise HTTPException(400, "An empty file was uploaded.")
    if file.size > max_bytes:
        raise HTTPException(
            413, f"PDF must be at most {max_bytes // (1024 * 1024)}MB."
        )

    file.file.seek(0)
    return file.file.read()


def attachment(content: bytes, media_type: str, filename: str) -> Response:
    """Builds a download response.

    FileResponse is not used because temp file cleanup would have to be left to a
    BackgroundTask, and on the Range header parse failure and mid-transfer exception
    paths starlette returns before reaching the background, leaving the temp file
    forever."""
    # filename comes from the upload's file name. Names that quote() changes (quotes,
    # newlines, non-ASCII) would be header injection if inserted as-is, so they are
    # sent as RFC 5987 filename*.
    quoted = quote(filename)
    disposition = (
        f'attachment; filename="{filename}"'
        if quoted == filename
        else f"attachment; filename*=utf-8''{quoted}"
    )
    return Response(
        content=content,
        media_type=media_type,
        headers={"content-disposition": disposition},
    )


def json_download(payload: BaseModel | list | dict, filename: str) -> Response:
    body = (
        payload.model_dump_json()
        if isinstance(payload, BaseModel)
        else json.dumps(payload, ensure_ascii=False)
    )
    return attachment(body.encode(), "application/json", filename)


def download_name(file: UploadFile, suffix: str) -> str:
    return f"{Path(file.filename or 'output').stem}{suffix}"


def handle_llm_error(e: Exception, fallback: str) -> NoReturn:
    """Raises an LLM connection failure as 400 and anything else as 500.

    Each provider's SDK names its connection error 'APIConnectionError', so it is
    identified by class name without an import."""
    if type(e).__name__ == "APIConnectionError":
        raise HTTPException(400, "Cannot connect to the LLM server.") from e
    raise HTTPException(500, fallback) from e


@app.post("/api/v1/parser-odl")
def parser_odl(file: UploadFile = File(...)):
    """Converts a PDF with opendataloader-pdf and returns the result JSON as a download.

    Args:
        file: PDF file (at most `CQ_MAX_PDF_BYTES`).

    Returns:
        Download of the conversion result file (`<original name>.odl.json`).

    Raises:
        HTTPException: empty file 400, size exceeded 413, java conversion failure 500,
            missing conversion result file 500, conversion timeout 504.
    """
    pdf_bytes = read_pdf_upload(file)
    try:
        content = convert(pdf_bytes)
    except ConversionTimeout as e:
        raise HTTPException(504, str(e)) from e
    except ConversionFailed as e:
        raise HTTPException(500, str(e)) from e

    return attachment(content, "application/json", download_name(file, ".odl.json"))


def read_json_upload(file: UploadFile, what: str) -> dict:
    max_bytes = settings.cq_max_json_bytes
    if not file.size:
        raise HTTPException(400, "An empty file was uploaded.")
    # json.load loads everything into memory, so it must be blocked before parsing.
    if file.size > max_bytes:
        raise HTTPException(
            413, f"{what} JSON must be at most {max_bytes // (1024 * 1024)}MB."
        )
    file.file.seek(0)
    try:
        payload = json.load(file.file)
    except (json.JSONDecodeError, UnicodeDecodeError) as e:
        raise HTTPException(422, f"{what} JSON file is invalid: {e}")
    if not isinstance(payload, dict):
        raise HTTPException(422, f"{what} JSON top level must be an object.")
    return payload


@app.post("/api/v1/structure-odl")
def structure_odl(file: UploadFile = File(...)):
    """Extracts a document's top-level skeleton from opendataloader-pdf JSON as a file.

    Only the skeleton (parts, chapters, appendices) is decided by the LLM; the sections
    and clauses inside are filled by `/api/v1/chunk-odl` from coordinates and numbering.

    The LLM does not produce block text; it only picks ids of existing blocks. The
    result is validated against the block list and rejected with 422 on mismatch.

    Args:
        file: Upload the file `/api/v1/parser-odl` returned, as-is
            (at most `CQ_MAX_JSON_BYTES`).

    Returns:
        Download of `<original name>.structure.json`. Each unit holds title_id/prefix_id
        (the blocks the skeleton points to), kind, depth and title. Upload this file
        together to `/api/v1/chunk-odl`.

    Raises:
        HTTPException: empty file 400, size exceeded 413, JSON format mismatch 422, no
            text blocks or headings 422, window count over `ODL_STRUCTURE_MAX_WINDOWS`
            422, skeleton validation failure 422, LLM connection failure 400, LLM output
            failure 500.
    """
    payload = read_json_upload(file, "ODL")
    try:
        structure = build_structure(payload, callbacks=[CallbackHandler()])
    # OutputParserException is a ValueError subclass, so this order carries meaning.
    # Moved below, a server-side LLM failure would go out as "your file is wrong" (422)
    # and mix with legitimate 422s.
    except (StructureLLMError, OutputParserException) as e:
        # The raw parser text goes only to the log. Put in the response body, the prompt
        # and model response would leak as-is.
        logging.exception("[structure-odl] LLM output failed")
        handle_llm_error(
            e, "Document skeleton extraction failed. Please try again shortly."
        )
    except ValueError as e:
        raise HTTPException(422, str(e))
    except Exception as e:
        handle_llm_error(e, f"Document skeleton extraction failed: {e}")

    return json_download(structure, download_name(file, ".structure.json"))


@app.post("/api/v1/chunk-odl")
def chunk_odl(
    file: UploadFile = File(...),
    structure: UploadFile | None = File(default=None),
    budget: int = Query(default=settings.cq_chunk_char_budget, gt=0),
    document_id: str = Query(...),
    # run_id is a free column without an FK (unlike document_id/paragraph_id), so the DB
    # does not block an empty value. Unless blocked here, rows with run_id="" pile up
    # silently.
    run_id: str = Query(..., min_length=1),
) -> ParagraphsCreated:
    """Splits opendataloader-pdf JSON into MCS chunks and loads them into paragraph.

    A chunk is defined on the section tree as "the whole subtree of the shallowest
    ancestor that fits the budget". Paragraphs and inner nodes are never cut across
    chunk boundaries, so each chunk's sources always form a list of intact paragraphs.

    Args:
        file: Upload the file `/api/v1/parser-odl` returned, as-is
            (at most `CQ_MAX_JSON_BYTES`).
        structure: The file `/api/v1/structure-odl` returned (optional). If uploaded,
            the top-level skeleton is built from it and only the inside is filled from
            coordinates and numbering. It is checked to come from this document, so a
            hand-edited file or one from another document gets 422. If not uploaded,
            everything is built from coordinates and numbering; for documents with no
            headings that express hierarchy only by numbering, the skeleton does not
            help, so omitting it is better.
        budget: Target chunk size (in characters). It is a target, not a cap, so a node
            with no children to split comes out whole even over budget.
        document_id: Document id.
        run_id: Pipeline run ID.

    Returns:
        run_id and the paragraph_ids of the loaded paragraphs (in chunk order).
        A chunk's context/chars and source list (paragraphs) have no place in the table,
        so they are dropped.

    Raises:
        HTTPException: empty file 400, size exceeded 413, unknown document_id 400,
            JSON format/schema mismatch 422, no text blocks 422,
            skeleton not from this document 422.
    """
    payload = read_json_upload(file, "ODL")

    spine = None
    if structure is not None and structure.size:
        try:
            uploaded = Structure.model_validate(
                read_json_upload(structure, "structure")
            )
        except ValidationError as e:
            raise HTTPException(422, f"structure JSON file is invalid: {e}")
        errors = validate_uploaded(payload, uploaded)
        if errors:
            raise HTTPException(
                422, f"Skeleton does not match this document: {', '.join(errors)}"
            )
        spine = uploaded.units

    try:
        chunks = chunk_tree(parse_odl_to_tree(payload, spine), budget=budget)
    except ValueError as e:
        raise HTTPException(422, str(e))

    try:
        paragraph_ids = save_paragraphs(chunks, document_id, run_id)
    except MissingReference as e:
        raise HTTPException(400, str(e))

    return ParagraphsCreated(run_id=run_id, paragraph_ids=paragraph_ids)


@app.post("/api/v1/generate")
def generate(
    purpose: str = Body(...),
    scope: str = Body(...),
    paragraph_id: str = Body(...),
    run_id: str = Body(..., min_length=1),
) -> CQsCreated:
    """Generates CQs (Competency Questions) from a paragraph and loads them into cq.

    The text and owning document are read via paragraph_id.

    Args:
        purpose: Purpose of the ontology.
        scope: Scope of the ontology.
        paragraph_id: Source paragraph identifier (issued by `/api/v1/chunk-odl`).
        run_id: Pipeline run ID.

    Returns:
        run_id and the cq_ids of the loaded CQs. This stage does not create flags.

    Raises:
        HTTPException: unknown paragraph_id 404, LLM connection failure 400, other
            generation failure 500.
    """
    paragraph = load_paragraph(paragraph_id)
    if paragraph is None:
        raise HTTPException(404, f"Unknown paragraph_id: {paragraph_id}")

    try:
        cqs = generate_cqs(
            purpose=purpose,
            scope=scope,
            document_text=paragraph.text,
            callbacks=[CallbackHandler()],
        )
    except Exception as e:
        handle_llm_error(e, f"CQ generation failed: {e}")

    try:
        save_drafts(cqs, paragraph.document_id, paragraph_id, run_id)
    except MissingReference as e:
        raise HTTPException(400, str(e))

    return CQsCreated(run_id=run_id, cq_ids=[c.id for c in cqs])


@app.post("/api/v1/refine")
def refine(
    cq_id: str = Query(...),
    run_id: str = Query(..., min_length=1),
) -> CQsRefined:
    """Verifies and fixes one loaded CQ against its paragraph until it converges.

    The stored CQ is then replaced with the result.

    Args:
        cq_id: Target CQ identifier (= cqs[].cq_id of the `/api/v1/generate` response).
            If it does not converge, nothing is saved: the returned list is the table's
            content, so an option like "include discarded ones in the result" would
            load discarded ones as normal rows.
        run_id: Pipeline run ID.

    Returns:
        run_id and the list of CQs left after convergence. The response is a list
        because one CQ can split into several (composite decomposition). When split,
        several come out with new ids, and when discarded it is empty, so it may differ
        from the input cq_id.

    Raises:
        HTTPException: unknown cq_id/paragraph_id 404, LLM connection failure 400,
            verification not completed so the original is kept 503, other refine
            failure 500.
    """
    row = load_draft(cq_id)
    if row is None:
        raise HTTPException(404, f"Unknown draft_id: {cq_id}")

    # The FK guarantees it exists, but if missing, refine would run without grounds, so
    # it is blocked.
    paragraph = load_paragraph(row.paragraph_id)
    if paragraph is None:
        raise HTTPException(404, f"Unknown paragraph_id: {row.paragraph_id}")

    draft = DraftCQ(
        question=row.question,
        answer=row.answer,
        id=row.draft_id,
    )
    try:
        outcome = refine_until_converged(
            draft,
            document_text=paragraph.text,
            callbacks=[CallbackHandler()],
        )
    except Exception as e:
        handle_llm_error(e, f"CQ refine failed: {e}")

    if outcome.failed:
        raise HTTPException(
            503, f"refine could not complete; the original is kept: {cq_id}"
        )

    replace_draft(cq_id, outcome.kept, row.document_id, row.paragraph_id, run_id)

    return CQsRefined(
        run_id=run_id, cqs=[CQRef(cq_id=c.id, flag=c.flag) for c in outcome.kept]
    )


@app.post("/api/v1/terminologize")
def terminologize(cq_id: str = Query(...)) -> CQ:
    """Structures one CQ into ontology terms (classes / object and data properties).

    Args:
        cq_id: Target CQ identifier (= an item of the `/api/v1/promote-drafts`
            response).

    Returns:
        One CQ with terms attached. The source is pointed to by paragraph_id, not by an
        evidence sentence; if the text is needed, read the paragraph with that id.

    Raises:
        HTTPException: unknown cq_id 404, LLM connection failure 400, other
            terminologize failure 500.
    """
    row = load_cq(cq_id)
    if row is None:
        raise HTTPException(404, f"Unknown cq_id: {cq_id}")

    # The text comes from the promoted cq, the flag from cq_draft. The flag is not in cq
    # because it is not a persistent attribute; promotion inserts with cq_id = draft_id,
    # so the draft can be traced back by the same id.
    draft_row = load_draft(cq_id)
    flag = draft_row.refine_check_flag if draft_row else None

    draft = DraftCQ(
        question=row.question,
        answer=row.answer,
        id=row.cq_id,
        flag=CheckFlag(**flag) if flag else None,
    )
    try:
        made = terminologize_cq(draft, callbacks=[CallbackHandler()])
    except Exception as e:
        handle_llm_error(e, f"CQ terminologizer failed: {e}")

    # Identifiers are attached here: the row read above already knows which row this
    # output belongs to. Passed inward (graph/LLM), a value unrelated to extraction
    # would travel along that path.
    return CQ(
        cq_id=row.cq_id,
        paragraph_id=row.paragraph_id,
        **made.model_dump(),
    )


@app.post("/api/v1/draft-review", status_code=204)
def draft_review(
    draft_id: str = Query(..., min_length=1),
    action: ReviewerAction | None = Body(default=None),
    feedback: str | None = Body(default=None),
) -> None:
    """Records the reviewer's decision on a draft in cq_draft (gate 1).

    A review tool calls it per row. It only does an UPDATE that keeps draft_id: the
    promotion invariant stands on the row count of cq_draft, so if a row disappears,
    promote rolls back.

    The request is the complete state of both columns, and omitted values are
    overwritten with NULL: FastAPI cannot tell an explicit null from an omission in an
    embedded body, so there is no way to express a partial patch.

    Args:
        draft_id: Target draft identifier.
        action: Review decision. `rewrite`/`reanswer` become targets of
            `/api/v1/regenerate` afterwards. null resets to "unreviewed", and no answer
            means accept, so that row is promoted too.
        feedback: Reviewer note. Carried into the regeneration prompt as-is.

    Raises:
        HTTPException: unknown draft_id 404.
    """
    if not update_draft_review(draft_id, action, feedback):
        raise HTTPException(404, f"Unknown draft_id: {draft_id}")


@app.post("/api/v1/draft-verdict", status_code=204)
def draft_verdict(
    draft_id: str = Query(..., min_length=1),
    verdict: RegenerationVerdict | None = Query(default=None),
) -> None:
    """Records the reviewer's verdict on a regeneration result (gate 2).

    It is a separate endpoint from gate 1 because it writes a different column: fixing
    reviewer_action during regeneration review is not reflected, since regeneration has
    already finished, and only the record goes out of sync.

    Args:
        draft_id: Target draft identifier.
        verdict: `reject` leaves it out of promotion. No answer (NULL) means accept.

    Raises:
        HTTPException: unknown draft_id 404.
    """
    if not update_draft_verdict(draft_id, verdict):
        raise HTTPException(404, f"Unknown draft_id: {draft_id}")


@app.post("/api/v1/regenerate", status_code=204)
def regenerate_draft(draft_id: str = Query(..., min_length=1)) -> None:
    """Regenerates one draft from the reviewer feedback and the source paragraph.

    If it is not a target, it ends without calling the LLM, so a retried step makes no
    duplicate call.

    Args:
        draft_id: Target draft identifier (`reviewer_action` must already be recorded).

    Raises:
        HTTPException: unknown draft_id/paragraph_id 404, LLM connection failure 400,
            otherwise 500.
    """
    draft = load_draft(draft_id)
    if draft is None:
        raise HTTPException(404, f"Unknown draft_id: {draft_id}")

    already_regenerated = draft.regenerated_at is not None
    if already_regenerated or not needs_regeneration(draft.reviewer_action):
        return

    paragraph = load_paragraph(draft.paragraph_id)
    if paragraph is None:
        raise HTTPException(404, f"Unknown paragraph_id: {draft.paragraph_id}")

    regenerate = rewrite if draft.reviewer_action == "rewrite" else reanswer
    try:
        result = regenerate(
            question=draft.question,
            answer=draft.answer,
            feedback=draft.reviewer_feedback or "",
            paragraph_text=paragraph.text,
        )
    except Exception as e:
        handle_llm_error(e, f"human-refine regeneration failed: {draft_id}")

    update_draft_regenerated(draft_id, result.question, result.answer)


@app.get("/api/v1/drafts")
def list_drafts(run_id: str = Query(..., min_length=1)) -> list[CQDraftRow]:
    """Returns every cq_draft belonging to run_id.

    Gate review surface: the reviewer sees question/answer/refine_check_flag here and
    edits the row in a review tool.
    """
    return load_drafts_by_run(run_id)


@app.post("/api/v1/promote-drafts")
def promote_drafts_endpoint(run_id: str = Query(..., min_length=1)) -> list[str]:
    """Promotes cq_draft → cq. Returns the promoted cq_ids in paragraph order."""
    promote_drafts(run_id)
    return load_promoted_ids(run_id)


if __name__ == "__main__":
    uvicorn.run("src.api:app", host="0.0.0.0", port=8000, reload=True)
