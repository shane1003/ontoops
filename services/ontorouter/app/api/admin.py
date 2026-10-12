"""Internal admin endpoints.

Internal-only by deployment: these run on a trusted network and carry no auth in
v1. Do not expose this router publicly.
"""

from typing import Annotated

from fastapi import APIRouter, BackgroundTasks, Depends, HTTPException, Request, status

from app.clients.embed import embed_client
from app.core.config import Settings, get_settings
from app.services.repository import PostgresRepository
from app.services.semantic_index import run_backfill

router = APIRouter(prefix="/admin")


@router.post("/embed-backfill", status_code=status.HTTP_202_ACCEPTED)
def trigger_embed_backfill(
    request: Request,
    tasks: BackgroundTasks,
    settings: Annotated[Settings, Depends(get_settings)],
) -> dict[str, str]:
    """Trigger an incremental embedding backfill.

    Args:
        request: The incoming request, carrying app state.
        tasks: FastAPI background task queue.
        settings: Injected settings.

    Returns:
        An acknowledgement; the pass runs in the background.

    Raises:
        HTTPException: 503 when the semantic tier is not configured.
    """
    if not settings.semantic_enabled:
        raise HTTPException(
            status_code=status.HTTP_503_SERVICE_UNAVAILABLE,
            detail="semantic tier is disabled: set EMBED_URL and EMBEDDING_MODEL_ID",
        )
    repo = PostgresRepository(request.app.state.db, timeout_s=settings.pg_timeout_s)
    embedder = embed_client(str(settings.embed_url), settings.embed_timeout_s)
    tasks.add_task(run_backfill, repo, embedder, str(settings.embedding_model_id))
    return {"status": "accepted"}
