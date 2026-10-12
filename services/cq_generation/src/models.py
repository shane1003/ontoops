"""Parsing and validation contract for the LLM's structured output.

Every structured-output call uses method="json_mode". The model never sees these
classes: neither class docstrings, Field(description=...) nor ConfigDict(title=...) are
passed to it.

The single source of truth for key names, enum values, empty-value rules and array
rules is the JSON shape instruction block at the end of each prompt in prompts/. A rule
written only in a docstring here reaches no one, so rules the model must know go in the
prompt.

These classes are responsible only for the parsing/validation contract: types,
validators, defaults.

Writing principles:
  - Pydantic defaults are honored as is: a field the model omits gets its default.
    So only fields the prompt allows to be omitted get a default; required keys get
    none (see the comments on IdentifiedUnits, UnitDepths). A default on a required key
    lets a wrong key name from the model pass silently.
  - If the prompt's key names and the field names diverge, parsing breaks. Change the
    prompt whenever you change a field.
"""

from enum import Enum

from pydantic import BaseModel, ConfigDict, Field, field_validator

from src.format_validator import (
    ensure_lower_camel,
    ensure_upper_camel,
    ensure_xsd_datatype,
)
from src.text_normalizer import normalize


class CQCore(BaseModel):
    """Core question/answer fields of a CQ.

    Neutral base of the LLM schemas (GeneratedCQ etc.). Kept separate because
    inheriting an LLM schema directly would propagate its LLM-facing docstring/title
    alias. The pipeline entity/contract is schemas.DraftCQ (an independent DTO).
    """

    model_config = ConfigDict(populate_by_name=True)

    question: str = Field(
        description="A natural language question the ontology must be able to answer."
    )
    answer: str = Field(
        description="The expected answer to the competency question, as it would be represented in the ontology."
    )

    @field_validator("question", "answer", mode="before")
    @classmethod
    def normalize_unicode(cls, v: str) -> str:
        return normalize(v) if isinstance(v, str) else v


class GeneratedCQ(CQCore):
    """Extract a single Competency Question from the document with its expected answer."""

    model_config = ConfigDict(title="CompetencyQuestion")


class GeneratedCQList(BaseModel):
    """Extract all Competency Questions relevant to the document."""

    model_config = ConfigDict(title="CompetencyQuestionList")

    questions: list[GeneratedCQ]


class IllPosedVerdict(BaseModel):
    """Judge whether THIS single competency question is ill-posed."""

    is_ill_posed: bool = Field(
        description=(
            "True if the question's text alone does not determine a specific answer "
            "(a dangling reference, a selection from an unspecified set, or not a "
            "genuine question)."
        )
    )
    reason: str = Field(
        default="",
        description="One-sentence reason, only when is_ill_posed is true.",
    )


class CompositeQuestionVerdict(BaseModel):
    """Judge whether THIS single question is composite (asks for two or more independent things)."""

    is_composite: bool = Field(
        description="True if the question conjoins or enumerates multiple independent aspects."
    )
    reason: str = Field(
        default="",
        description="One-sentence reason, only when is_composite is true.",
    )


class CompositeAnswerVerdict(BaseModel):
    """Judge whether THIS single answer is composite (holds two or more distinct values)."""

    is_composite: bool = Field(
        description="True if the answer holds multiple distinct values rather than a single fact."
    )
    reason: str = Field(
        default="",
        description="One-sentence reason, only when is_composite is true.",
    )


class SplitQuestion(CQCore):
    """One atomic question-answer pair produced by splitting a composite question.

    Derive both the question and the answer from the original question and answer
    only. If the answer to this atomic question is not present in the original
    answer, return an empty string for the answer instead of inventing one.
    """

    model_config = ConfigDict(title="AtomicQuestion")


class SplitResult(BaseModel):
    """Split THIS one composite question into atomic question-answer pairs."""

    questions: list[SplitQuestion] = Field(
        description="Two or more atomic questions, each asking for a single fact."
    )


class RephraseResult(BaseModel):
    """A well-posed rewrite of THIS one ill-posed question, grounded in the Document."""

    question: str | None = Field(
        description=(
            "The rephrased, well-posed question, grounded in the Document. "
            "Return null if it cannot be made well-posed from the Document."
        ),
    )
    answer: str | None = Field(
        description="The updated answer if it changes; null to keep the original answer.",
    )


class VerifyStatus(str, Enum):
    """Verification outcome for a CQ."""

    VERIFIED = "verified"
    UNVERIFIED = "unverified"
    NEED_REFINE = "need_refine"
    REFINED = "refined"
    UNREFINED = "unrefined"


# Terminologizer: CQ → ontology terms (classes / object / data properties)


class TerminologizedClass(BaseModel):
    """Extract an ontology class implied by the competency question, in UpperCamelCase."""

    model_config = ConfigDict(title="OntologyClass")

    name: str = Field(description="The ontology class name in UpperCamelCase.")

    @field_validator("name")
    @classmethod
    def check_name(cls, v: str) -> str:
        return ensure_upper_camel(v)


class TerminologizedProperty(BaseModel):
    """Shared name/domain/value base; the object/data subclasses narrow range."""

    name: str = Field(
        description="The property name in lowerCamelCase, e.g. issuedBy or hasOrderIdentifier."
    )
    domain: str = Field(
        description="The class this property belongs to, in UpperCamelCase."
    )
    value: str = Field(
        description="The concrete value of this property, derived from the CQ's answer text."
    )

    @field_validator("name")
    @classmethod
    def check_name(cls, v: str) -> str:
        return ensure_lower_camel(v)

    @field_validator("domain")
    @classmethod
    def check_domain(cls, v: str) -> str:
        return ensure_upper_camel(v)


class TerminologizedObjectProperty(TerminologizedProperty):
    """Extract an object property: a relation linking the domain class to another class (the range).

    Use an object property when the answer names another entity/class. Prefer a data
    property when the answer is a literal value (see DataProperty).
    """

    model_config = ConfigDict(title="ObjectProperty")

    range: str = Field(
        description="The target class name this property points to, in UpperCamelCase."
    )

    @field_validator("range")
    @classmethod
    def check_range(cls, v: str) -> str:
        return ensure_upper_camel(v)


class TerminologizedDataProperty(TerminologizedProperty):
    """Extract a data property: an attribute linking the domain class to a literal value.

    This is the default when the answer is a literal (number, string, date, code). Only
    use an object property when the answer clearly names another entity/class.
    """

    model_config = ConfigDict(title="DataProperty")

    range: str = Field(
        description="The XSD datatype of the literal value, e.g. xsd:string or xsd:date."
    )

    @field_validator("range")
    @classmethod
    def check_range(cls, v: str) -> str:
        return ensure_xsd_datatype(v)


class OntologyTerms(BaseModel):
    """Extract the ontology terms implied by THIS competency question and its answer.

    Emit a class for the question's subject and for every class referenced as an object
    property's domain or range (intra-CQ referential closure). Derive each value from the
    answer text only; do not use any external document. If the answer holds several values
    for the same property, emit one property entry per value.
    """

    model_config = ConfigDict(title="OntologyTerms")

    classes: list[TerminologizedClass] = Field(default_factory=list)
    object_properties: list[TerminologizedObjectProperty] = Field(default_factory=list)
    data_properties: list[TerminologizedDataProperty] = Field(default_factory=list)


class TerminologizedCQ(CQCore):
    """The original Q/A wrapped with the LLM-extracted terms (Terminologizer output)."""

    classes: list[TerminologizedClass] = Field(default_factory=list)
    object_properties: list[TerminologizedObjectProperty] = Field(default_factory=list)
    data_properties: list[TerminologizedDataProperty] = Field(default_factory=list)

    @property
    def is_empty(self) -> bool:
        """Whether there is no term at all (extraction failed or empty output).

        Used to observe fail-open.
        """
        return not (self.classes or self.object_properties or self.data_properties)


class HumanRefineResult(CQCore):
    """Revise the question and/or answer based on human reviewer feedback."""


class ReanswerResult(BaseModel):
    """Output of reanswer.

    The question is not asked of the LLM; the original is attached after the call,
    because instructions cannot guarantee that a question echoed back by the model is
    character-for-character identical.
    """

    answer: str

    @field_validator("answer", mode="before")
    @classmethod
    def normalize_unicode(cls, v: str) -> str:
        return normalize(v) if isinstance(v, str) else v


# ODL structure extraction: block list → document skeleton (division/chapter/appendix)


class StructuralUnitKind(str, Enum):
    DIVISION = "division"
    CHAPTER = "chapter"
    APPENDIX = "appendix"
    FRONT = "front"
    BACK = "back"


class IdentifiedUnit(BaseModel):
    title_id: int
    prefix_id: int | None = None
    kind: StructuralUnitKind


class IdentifiedUnits(BaseModel):
    model_config = ConfigDict(title="StructuralUnits")

    # No default. The prompt demands an explicit [] even for an empty window, so a
    # response missing the key is not an empty window but a lost one, and must surface
    # here as a parse failure.
    units: list[IdentifiedUnit]


class UnitDepth(BaseModel):
    """Assign one structural unit its nesting depth."""

    id: int = Field(description="The unit's id, exactly as given in the input.")
    depth: int = Field(description="Nesting depth. 0 is the top level.")


class UnitDepths(BaseModel):
    """Report the nesting depth of every unit given, sorted by id ascending."""

    model_config = ConfigDict(title="UnitDepths")

    # No default. The prompt demands that every unit be reported, so a response missing
    # the key is not an empty result but a lost depth judgment, and must surface here as
    # a parse failure.
    units: list[UnitDepth]
