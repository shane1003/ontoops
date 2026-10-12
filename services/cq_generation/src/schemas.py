"""Schema contracts used between APIs.

Design:
- Notation checks are delegated to format_validator's ensure_* functions (this file
  only wires them).
"""

from datetime import datetime
from typing import ClassVar, Literal
from uuid import uuid4

from pydantic import BaseModel, Field, field_validator

from src.format_validator import (
    ensure_lower_camel,
    ensure_upper_camel,
    ensure_xsd_datatype,
)
from src.text_normalizer import normalize


class CheckFlag(BaseModel):
    """One verification check that fired on a CQ, naming the check and its reason."""

    check_name: str
    reason: str


class DraftCQ(BaseModel):
    """Schema shared across the pipeline.

    id + Q/A + optional verification flag.
    flag is None before the verify step (generate) and is filled after refine when there
    is a defect/note.
    """

    question: str = Field(
        description="A natural language question the ontology must be able to answer."
    )
    answer: str = Field(
        description="The expected answer to the competency question, as it would be represented in the ontology."
    )
    id: str = Field(default_factory=lambda: str(uuid4()))
    flag: CheckFlag | None = None

    @field_validator("question", "answer", mode="before")
    @classmethod
    def normalize_unicode(cls, v: str) -> str:
        return normalize(v) if isinstance(v, str) else v


class OntologyClass(BaseModel):
    name: str = Field(description="The ontology class name in UpperCamelCase.")

    @field_validator("name")
    @classmethod
    def check_name(cls, v: str) -> str:
        return ensure_upper_camel(v)


class Property(BaseModel):
    name: str = Field(
        description="The property name in lowerCamelCase, e.g. issuedBy or hasOrderIdentifier."
    )
    domain: str = Field(
        description="The class this property belongs to, in UpperCamelCase."
    )
    range: str = Field(
        description="The range of this property (a class name for object properties, an XSD datatype for data properties)."
    )
    value: str = Field(
        description="The concrete value of this property for the answer, extracted from the document."
    )

    @field_validator("name")
    @classmethod
    def check_name(cls, v: str) -> str:
        return ensure_lower_camel(v)

    @field_validator("domain")
    @classmethod
    def check_domain(cls, v: str) -> str:
        return ensure_upper_camel(v)


class ObjectProperty(Property):
    """Extract an object property: a relation linking the domain class to another class (the range)."""

    range: str = Field(
        description="The target class name this property points to, in UpperCamelCase."
    )

    @field_validator("range")
    @classmethod
    def check_range(cls, v: str) -> str:
        return ensure_upper_camel(v)


class DataProperty(Property):
    """Extract a data property: an attribute linking the domain class to a literal value."""

    range: str = Field(
        description="The XSD datatype of the literal value, e.g. xsd:string or xsd:date."
    )

    @field_validator("range")
    @classmethod
    def check_range(cls, v: str) -> str:
        return ensure_xsd_datatype(v)


class CQBase(BaseModel):
    """Extract a competency question with its answer and the ontology terms it implies."""

    question: str = Field(
        description="A natural language question the ontology must be able to answer."
    )
    answer: str = Field(
        description="The expected answer to the competency question, as it would be represented in the ontology."
    )
    classes: list[OntologyClass]
    object_properties: list[ObjectProperty]
    data_properties: list[DataProperty]


class CQ(CQBase):
    """A competency question enriched with the ids assigned by code (not by the LLM).

    Provenance points to paragraph_id instead of carrying a copy of the supporting
    sentence: the body is in the paragraph table, so copying it would duplicate the
    text, and above all terminologize runs on the Q/A only and never reads the document
    (= it has no supporting text to copy in the first place).
    """

    cq_id: str
    paragraph_id: str


# ---------------------------------------------------------------------------
# Request contracts
# ---------------------------------------------------------------------------


# Decisions a reviewer can pick at gate 1.
ReviewerAction = Literal["accept", "rewrite", "reanswer", "drop"]

# Verdicts that can be picked for a regeneration result at gate 2. No answer (NULL)
# means accept.
RegenerationVerdict = Literal["accept", "reject"]


# ---------------------------------------------------------------------------
# Response contracts (load result = ids of the rows created)
#
# Only ids are returned, not the output body. The body is in the table, and the next
# step picks it up by id.
# ---------------------------------------------------------------------------


class ParagraphsCreated(BaseModel):
    run_id: str
    paragraph_ids: list[str]


class CQRef(BaseModel):
    """Reference to one stored CQ; the body is in the table, so it is not included.

    flag is this run's verification result attached by refine. The cq table has no
    column for it because it is not a persistent attribute; instead it travels to
    terminologize in this response (terminologize uses the composite_answer flag to take
    the multi-concept decomposition branch).
    """

    cq_id: str
    flag: CheckFlag | None = None


class CQsCreated(BaseModel):
    """Response of generate; it carries no flag.

    The flag at this step is always None (refine does the verification), so including
    it would publish a structurally dead field.
    """

    run_id: str
    cq_ids: list[str]


class CQsRefined(BaseModel):
    """Response of refine; carries the verification flag on to terminologize."""

    run_id: str
    cqs: list[CQRef]


# ---------------------------------------------------------------------------
# Storage contracts (PostgreSQL tables = the stored form of the API outputs)
#
# Outputs are stored in tables and returned instead of exported as JSON, so the row
# models live next to the request/response contracts above. The table definitions (DDL)
# are in sql/schema.sql and these models mirror their columns. A mismatch would only
# break when inserting into a real DB. The storage behavior (connection, INSERT) is in
# src/db.py.
# ---------------------------------------------------------------------------


class Row(BaseModel):
    """Base of the row models; field names and order must match sql/schema.sql."""

    TABLE: ClassVar[str]


class ParagraphRow(Row):
    """One paragraph cut from a document (own-node text)."""

    TABLE = "paragraph"

    paragraph_id: str = Field(default_factory=lambda: str(uuid4()))
    document_id: str
    run_id: str
    section_path: str
    page_start: int
    # Equal to page_start when the paragraph does not span pages.
    page_end: int
    seq: int
    text: str


class CQRow(Row):
    """One competency question generated from a paragraph."""

    TABLE = "cq"

    cq_id: str
    # Derivable from paragraph_id, but denormalized here for query convenience.
    document_id: str
    paragraph_id: str
    run_id: str
    question: str
    answer: str


class CQDraftRow(Row):
    """Draft table that owns a CQ's whole indexing-time lifetime.

    On promotion, cq_id = draft_id.
    """

    TABLE = "cq_draft"

    draft_id: str
    document_id: str
    paragraph_id: str
    run_id: str
    question: str
    answer: str
    refine_check_flag: dict | None = None
    reviewer_action: str | None = None
    reviewer_feedback: str | None = None
    regenerated_at: datetime | None = None
    regeneration_verdict: str | None = None
    original_question: str | None = None
    original_answer: str | None = None
