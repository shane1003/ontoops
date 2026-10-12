from pathlib import Path
from typing import Literal

from pydantic_settings import BaseSettings, SettingsConfigDict

_ROOT = Path(__file__).parent


class Settings(BaseSettings):
    model_config = SettingsConfigDict(
        env_file=".env", env_file_encoding="utf-8", extra="ignore"
    )

    # LLM provider: "local" | "openai" | "anthropic"
    llm_provider: Literal["local", "openai", "anthropic"] = "local"

    # Local vLLM
    vllm_base_url: str = "http://localhost:8000/v1"
    vllm_model: str = "your-model-id"

    # OpenAI API
    openai_api_key: str = ""
    openai_model: str = "gpt-4o"

    # Anthropic API
    anthropic_api_key: str = ""
    anthropic_model: str = "claude-sonnet-4-6"

    # parser configs
    cq_max_pdf_bytes: int = 20971520
    # Upload cap for ODL/structure JSON. It is separate from the PDF cap because the
    # JSON/PDF ratio swings 7x across documents, 0.09-0.62 (PDF images are not carried
    # into the JSON). So the cap uses the maximum ratio, bounding the JSON a document
    # that passed cq_max_pdf_bytes can produce: 20MB x 0.62 = 12.4MiB -> the smallest
    # MB value that holds it.
    cq_max_json_bytes: int = 13631488
    # Wall-clock cap for opendataloader-pdf's java subprocess. Without it, one hung JVM
    # holds an anyio thread of the sync endpoint forever, and once the thread pool
    # (default 40) fills, the whole app stops responding; it does not die, so restart
    # never triggers either. A size cap cannot guard this axis.
    # Conversion time scales almost linearly with size, so even a cq_max_pdf_bytes
    # (20MB) document finishes in about 15s. Slower hardware is allowed for with 8x
    # that: a value that never cuts a normal conversion but kills a hung JVM.
    odl_convert_timeout: int = 120
    # Heap cap for the same java subprocess. Without it the JVM default is '1/4 of
    # physical memory', so the bigger the host, the more one process takes. As a sync
    # endpoint, each request spawns its own JVM and up to the thread pool size (default
    # 40) can live at once, which can drag vLLM on the same node into OOM too. This
    # value fixes the worst case = this value x 40.
    # 512m was chosen because of a cliff, not for headroom: on a public manual close to
    # cq_max_pdf_bytes (java 21), 512m finishes as fast as an uncapped JVM, while one
    # step below (256m) runs only GC for minutes before OOM, and 128m fails at once.
    # Why headroom cannot be pinned as a number: with a 1g heap the GC log shows
    # occupancy climbing to nearly 1g. The parser uses as much as it gets, so "the max
    # observed at 1g" is not the real requirement. Hence only the cliff position is used
    # as evidence.
    # Note that a shortfall has two symptoms: badly short, it dies with OOM and returns
    # 500; barely short, it runs only GC until odl_convert_timeout cuts it and returns
    # 504. If only large documents start returning 504, suspect this value, not the
    # timeout.
    odl_convert_max_heap: str = "512m"

    # ODL structure extraction (LLM for the skeleton only, coordinates and markers
    # inside) - see src/odl_structure.py
    odl_structure_identify_prompt_path: Path = (
        _ROOT / "prompts" / "odl_structure_identify.txt"
    )
    odl_structure_hierarchy_prompt_path: Path = (
        _ROOT / "prompts" / "odl_structure_hierarchy.txt"
    )
    odl_structure_temperature: float = 0.0
    odl_structure_max_tokens: int = 8000
    # Cap for one LLM call (not the whole request). The base is not 'a single request
    # on an idle server' but the cold maximum under 4 concurrent requests: real
    # requests arrive concurrently and vLLM queueing pushes window latency back
    # accordingly, so a single-request value does not represent real use.
    # Base: the cold latency of one full 30000-char window under concurrency 4, without
    # retry (about 21.5s), x 1.5 = 33.
    # Coverage: up to concurrency 4 the 1.5x margin stays intact; at concurrency 8 the
    # latency reaches about 33s, a boundary with zero margin. Beyond that, healthy calls
    # get cut and their retries put 3x load on an already busy server. What that range
    # needs is a concurrency limit, not a larger timeout, so it is not a problem to
    # solve by raising this value.
    # Why odl_convert_timeout is 8x while this is 1.5x (same 'the server is slower'
    # risk, different multiplier): convert is called once per request, so its
    # multiplier is the request time, but this value is amplified by
    # (6 windows + 1 hierarchize) x 3 attempts = 21x; giving it 8x makes the request
    # worst case 38 minutes.
    # Also, convert is local CPU/JVM and deterministic, so one 8x covers hardware
    # variance, but LLM latency grows with server load, so no multiplier can cover all
    # concurrency. So instead of raising the multiplier, the concurrency to cover is
    # chosen and used as the base.
    # Caution: this value is multiplied by 3 in request time, because _MAX_RETRIES=2 in
    # client.py explicitly pins SDK retries, so one call is attempted up to 3 times
    # (= 3 x this value).
    # So the request worst case is (max_windows 6 + hierarchize 1) x 3 x 33 = 693s.
    # The odl_structure LLM calls have no with_retry, so this is the only retry layer.
    # Adding one multiplies the two layers, so this formula must be updated with it
    # (for a 3-attempt one, 693 x 3 = 2079s).
    odl_structure_request_timeout: int = 33
    odl_structure_window_chars: int = 30000
    # The LLM is called once, sequentially, per window, so this value is the cap on
    # LLM calls per request. Without it, one large input occupies serving entirely. A
    # manual of a few hundred pages needs about 4 windows, so this is 1.5x that.
    odl_structure_max_windows: int = 6

    # MCS chunking (section tree -> intact subtrees) - see src/mcs_chunker.py
    # Target chunk size. It is not a cap, so a node with no children to split goes out
    # whole even past this value: not cutting an intact unit takes priority over
    # staying within budget.
    # It is the default of the budget query parameter of /api/v1/chunk-odl, so callers
    # can override it per request.
    cq_chunk_char_budget: int = 1500

    # generator configs
    cq_generator_system_prompt_path: Path = (
        _ROOT / "prompts" / "cq_generator_system.txt"
    )
    cq_generator_temperature: float = 0.0
    cq_generator_request_timeout: int = 120
    cq_generator_document_content_char_limit: int = 20000
    cq_generator_max_tokens: int = 15000

    # verification config
    cq_verify_ill_posed_prompt_path: Path = _ROOT / "prompts" / "verify_ill_posed.txt"
    cq_verify_composite_question_prompt_path: Path = (
        _ROOT / "prompts" / "verify_composite_question.txt"
    )
    cq_verify_composite_answer_prompt_path: Path = (
        _ROOT / "prompts" / "verify_composite_answer.txt"
    )
    cq_verify_temperature: float = 0.0
    cq_verify_max_tokens: int = 128
    cq_verify_max_concurrency: int = 32
    cq_verify_request_timeout: int = 10

    # refine config
    cq_refine_split_prompt_path: Path = _ROOT / "prompts" / "refine_split.txt"
    cq_refine_ill_posed_prompt_path: Path = _ROOT / "prompts" / "refine_ill_posed.txt"
    # human-refine uses a separate prompt per action: rewrite fixes the question and
    # reanswer the answer, and their instructions to leave the other one alone are
    # opposite, so in one file the model touches both. Follows the same one-file-per-
    # operation rule as verify_*/refine_*.
    cq_human_refine_rewrite_prompt_path: Path = (
        _ROOT / "prompts" / "human_refine_rewrite.txt"
    )
    cq_human_refine_reanswer_prompt_path: Path = (
        _ROOT / "prompts" / "human_refine_reanswer.txt"
    )
    cq_refine_temperature: float = 0.0
    cq_refine_max_tokens: int = 512
    cq_refine_request_timeout: int = 40
    cq_refine_max_iterations: int = 3

    # terminologizer chain (CQ → ontology terms: classes / object / data properties)
    cq_terminologizer_system_prompt_path: Path = (
        _ROOT / "prompts" / "terminologizer_system.txt"
    )
    # split prompt for the composite_answer branch (multi-concept decomposition)
    cq_terminologizer_split_system_prompt_path: Path = (
        _ROOT / "prompts" / "terminologizer_split_system.txt"
    )
    cq_terminologizer_temperature: float = 0.0
    cq_terminologizer_max_tokens: int = 1024
    cq_terminologizer_request_timeout: int = 60

    # PostgreSQL loading - see src/db.py. The connection info is one DSN line rather
    # than split into fields because psycopg.connect() takes this string as-is (no
    # assembly code needed).
    postgres_dsn: str = "postgresql://postgres:postgres@localhost:5432/ontology"
    # The libpq default is to wait forever. When the DB is not dead but *just not
    # responding* (firewall drop, mid-failover, connection limit exceeded), the sync
    # endpoint's anyio threads (default 40) all hang here and the app stops responding
    # while alive; the process does not die, so restart never triggers. It is the same
    # failure mode odl_convert_timeout guards against.
    # Why 5s: a normal connection takes a few ms on a local network, so this is two
    # orders of magnitude of headroom, and one refine request opens a connection 3
    # times, so the worst wait is bounded at 15s and cut before the thread pool fills.
    postgres_connect_timeout: int = 5
    # The DDL that create_tables() runs. This file is the single source of truth for
    # table definitions; the *Row models in src/schemas.py mirror its columns.
    db_schema_path: Path = _ROOT / "sql" / "schema.sql"

    langfuse_secret_key: str = ""
    langfuse_public_key: str = ""
    langfuse_host: str = ""


settings = Settings()
