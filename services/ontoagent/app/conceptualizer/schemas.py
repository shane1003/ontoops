"""conceptualizer boundary schemas: CQ input, TBox components, ABox individuals.

This module defines the data structures that cross the DomainSpecificClassifier
module boundary.

- The CQ input models (`CQDocument` etc.) are the classifier input.
- `AnnotatedComponent` is a TBox component carrying the domain layer and the
  TBox/ABox verdict for one proposed class/property.
- `Individual` is one ABox individual (the ABox side of the split output).
- `ExtractionOutput` is the envelope holding both outputs ({components, individuals}).
"""

from __future__ import annotations

from typing import Literal

from pydantic import (
    AliasChoices,
    BaseModel,
    ConfigDict,
    Field,
    field_validator,
)

# The language tag policy is owned by core (constants shared with the consumer
# tboxcraft). Re-exported from this module for existing consumers.
from app.core.lang import (  # noqa: F401 — re-export (backward-compatible import surface)
    MIXED_LANG,
    UNDETERMINED_LANG,
    normalize_lang,
    script_lang,
)

# =========================================================================
# 1. CQ input models (Stage-1 INPUT) — the CQ document contract
# =========================================================================


class OntologyMeta(BaseModel):
    iri: str = Field(..., description="Base IRI of the ontology")
    # prefix is for the document name and IRIs. It once doubled as the classifier
    # domain (when the comment said "consumed as the classifier domain context"), but
    # once CQ Gen started building prefix from the document title that premise silently
    # became false: the document title went into the prompt as domain knowledge. The domain is the separate field below.
    prefix: str = Field(..., description="Namespace prefix (for the document name and IRIs)")
    # Domain that term judgments are based on (short label: military, medical,
    # legal ...). None when the document does not carry it: the runner falls back to
    # the deployment default (CONCEPTUALIZER_DOMAIN(env) > settings.agents.domain) and
    # exposes that fact as domain_source in the response. **It never falls back to
    # prefix**: a deployment default someone chose beats a document name in the
    # domain slot.
    domain: str | None = Field(
        default=None,
        description="Domain for term judgments (short label). Omit if unknown",
    )
    # Note: extra meta keys such as title/purpose/scope are ignored if present in the
    # input (extra=ignore). Ontology-level meta is owned by the external ontology
    # system (this module neither consumes nor outputs it).


class ClassDef(BaseModel):
    name: str
    comment: str | None = None
    subClassOf: str | None = None
    # Names of the owl:disjointWith target classes (disjointness).
    disjointWith: list[str] = []


class ObjectPropertyDef(BaseModel):
    name: str
    domain: str | None = None
    range: str | None = None
    comment: str | None = None  # General description of the property (to TBox if set).
    value: str | None = None  # Answer extracted in this CQ (goes only to ABox individual)
    inverse: str | None = None  # Inverse property name (default builder mapping if unset)


class DataPropertyDef(BaseModel):
    name: str
    domain: str | None = None
    range: str = Field("xsd:string", description="xsd datatype")
    comment: str | None = None  # General description of the property (to TBox if set).
    value: str | None = None  # Answer extracted in this CQ (goes only to ABox individual)


class ParagraphRef(BaseModel):
    """Provenance of CQ content: the page and section path in the source text.

    Example: {"page": [1], "section_path": ["Preface"]}

    Both are lists because one CQ can span several pages/sections. Unknown keys are
    kept (extra=allow) so that richer provenance does not break this model.
    """

    model_config = ConfigDict(extra="allow")

    page: list[int] = Field(default_factory=list, description="Page numbers in the source text")
    section_path: list[str] = Field(
        default_factory=list, description="Section title path (outer → inner)"
    )


class CompetencyQuestion(BaseModel):
    # Accept aliases so that renamed input fields (id↔cq_id, answer↔expected_answer,
    # paragraph↔opord_paragraph) still parse; code accesses the canonical field names
    # (id/answer/paragraph).
    model_config = ConfigDict(populate_by_name=True)

    # Real upstream output sends `cq_id` (old fixtures use `id`). Same alias policy
    # as answer/paragraph.
    id: str = Field(validation_alias=AliasChoices("id", "cq_id"))
    question: str
    # Answer to this CQ. Attached only to ABox as owl:AnnotationProperty
    # (expectedAnswer); not included in TBox.
    answer: str | None = Field(
        default=None, validation_alias=AliasChoices("answer", "expected_answer")
    )
    classes: list[ClassDef] = []
    object_properties: list[ObjectPropertyDef] = []
    data_properties: list[DataPropertyDef] = []
    # Provenance of this CQ's content. A list, since one CQ can span several places.
    # An empty list means unassigned (title section etc.).
    #
    # Both forms are accepted:
    #   - structured: {"page": [1], "section_path": ["Preface"]} → ParagraphRef
    #   - string (legacy): the paragraph key as-is → str
    paragraph: list[ParagraphRef | str] = Field(
        default_factory=list, validation_alias=AliasChoices("paragraph", "opord_paragraph")
    )

    @field_validator("paragraph", mode="before")
    @classmethod
    def _coerce_paragraph(cls, v):
        # Old-format compatibility: accept opord_paragraph as a single string/None
        # and turn it into a list.
        if v is None:
            return []
        if isinstance(v, str):
            return [v] if v else []
        return v


class CQDocument(BaseModel):
    """Corresponds to one data/cq/*.json file (Stage-1 input)."""

    ontology: OntologyMeta
    competency_questions: list[CompetencyQuestion]
    source: str | None = None


# =========================================================================
# 2. Annotated Component — handed to the external system
# =========================================================================

# Component kind: class / object property / data property
ComponentKind = Literal["class", "object_property", "data_property"]
# Classifier verdict: TBox element / ABox label / conflict (across chunks)
TBoxVerdict = Literal["tbox", "abox", "conflict"]


class DomainLayer(BaseModel):
    """Domain layer (additive layer) attached by DomainSpecificClassifier.

    Holds the suggested correct term and in-batch duplicate/hierarchy observations.
    """

    model_config = ConfigDict(extra="allow")

    # The canonical term recommended from the domain's point of view (if any) and
    # **the language of that term**.
    #
    # Without the language the receiver fills in a default: tboxcraft has always
    # stamped labels as `@en` and comments as `@ko` unconditionally. A Korean
    # correct_term then becomes an `rdfs:label "..."@en`, **leaving a false claim in the
    # ontology** (comments from an English manual going out as `@ko` is the same defect
    # in the other direction).
    #
    # The language is decided in this order (details: script_lang, classifier
    # _seal_language):
    #   Hangul script → `ko`/`mul` (fixed) > tag given by the classifier > request
    #   `?language=` > `und`.
    # Mixed text is not folded to one side; it is `mul`.
    # No value means no language (the language of absent text is not a claim).
    correct_term: str | None = Field(default=None, description="Recommended canonical term")
    correct_term_lang: str | None = Field(
        default=None,
        description=f"Language of correct_term (BCP-47). {UNDETERMINED_LANG} if unknown",
    )
    # synonyms and duplicates claim different things and must not be mixed.
    #   synonyms  : "this concept is also called this" (surface form). E.g. spelling
    #               variants such as IPB ↔ Ipb.
    #   duplicates: "a separate component exists and is the same as this" (a claim
    #               about another name).
    # Downstream (tboxcraft), duplicates feed not only surface forms but also the gold
    # labels for threshold calibration, cluster merge edges and the approval merge
    # shortcut. Putting spelling variants there silently shrinks the calibration
    # positives, and the moment a variant happens to match a real component name, a
    # wrong merge bypasses similarity checks. Hence a separate
    # surface-form-only channel.
    synonyms: list[str] = Field(
        default_factory=list,
        description="Other spellings of the same concept (surface forms only)",
    )
    duplicates: list[str] = Field(
        default_factory=list, description="Duplicate candidates from the domain view"
    )
    hierarchy: dict[str, list[str]] = Field(
        default_factory=dict,
        description="Hierarchy observations such as broader/narrower terms",
    )
    # Rationale prose and its language (same rules as correct_term /
    # correct_term_lang).
    notes: str | None = None
    notes_lang: str | None = Field(
        default=None,
        description=f"Language of notes (BCP-47). {UNDETERMINED_LANG} if unknown",
    )

    @field_validator("correct_term_lang", "notes_lang", mode="before")
    @classmethod
    def _normalize_lang(cls, v):
        # Values not shaped like a tag (e.g. "Korean", "") fold to None, so
        # the classifier fallback (request language → und) fills the slot. Passing an
        # unknown-shaped value through breaks downstream when used as a literal tag.
        return normalize_lang(v)


class VerdictCandidate(BaseModel):
    """One competing candidate held by a disagreement flag (value + source)."""

    value: str = Field(
        ...,
        description="Competing value (verdict: tbox/abox · hierarchy: broader/narrower)",
    )
    # Source: id of the chunk that produced this value (e.g. "chunk0").
    source: str = Field(..., description="Source of this candidate (chunk id)")


class DisagreementFlag(BaseModel):
    """Flag recording that observations disagree (two kinds).

    - `verdict_conflict`: the TBox/ABox verdict for the same (kind,name) component
      differs between chunks.
      candidates = [{value:"tbox",source:"chunk0"},{value:"abox",source:"chunk1"}]
    - `hierarchy_conflict`: the same name is observed as both broader **and** narrower
      of this component (contradictory edge direction). subject is that other name,
      candidates = [{value:"broader",source:"chunk3"},{value:"narrower",source:"chunk7"}].

    Both arise in reintegration (fan-in). Neither is resolved here: the verdict is
    only marked conflict and both edges are passed on. The disposition (which edge to
    drop) belongs to the downstream that owns the global TBox, and this flag is the
    only evidence for that decision. Passing on the contradiction without evidence
    would leave downstream flipping a coin.
    """

    kind: Literal["verdict_conflict", "hierarchy_conflict"] = Field(
        default="verdict_conflict",
        description=(
            "verdict_conflict (verdict across chunks) | hierarchy_conflict "
            "(contradictory edge direction)"
        ),
    )
    # Carry both ends of the edge by name. The flag arrives nested in a component,
    # but consumers **detach** it like a review queue item, losing its owner at that
    # moment. Two components can hold the same flag with the same subject and the
    # same candidates, and detached the two contradictions are indistinguishable. With component set, (component, subject) identifies the edge.
    component: str | None = Field(
        default=None,
        description="Name of the component this flag is on (this end of the edge)",
    )
    # What the disagreement is about. For hierarchy_conflict it is the name of the
    # other component whose direction differs (without it downstream would have to
    # parse the prose note to decide which edge to drop).
    # verdict_conflict targets the component itself, so it stays empty (only
    # component is set).
    subject: str | None = Field(
        default=None,
        description="Name the disagreement is about (other component of hierarchy_conflict)",
    )
    candidates: list[VerdictCandidate] = Field(
        default_factory=list, description="Competing candidates (value + source)"
    )
    note: str | None = None


class AnnotatedComponent(BaseModel):
    """TBox component for one proposed class/property.

    Carries the component structure (name/kind/domain/range), the domain layer, the
    TBox/ABox verdict and the cross-chunk verdict conflict flags.
    """

    model_config = ConfigDict(populate_by_name=True)

    # --- component structure ---
    name: str = Field(
        ...,
        description="Component name (byte-identical to the individuals' class ref)",
    )
    kind: ComponentKind = Field(..., description="class | object_property | data_property")
    domain: str | None = Field(default=None, description="domain of the property (if applicable)")
    range: str | None = Field(default=None, description="range of the property (if applicable)")
    comment: str | None = None

    # --- additive layer ---
    domain_layer: DomainLayer = Field(default_factory=DomainLayer)

    # --- verdict and disagreement ---
    # TBox-vs-ABox-label verdict. conflict when chunks disagree.
    verdict: TBoxVerdict = Field(default="tbox", description="tbox | abox | conflict")
    disagreement_flags: list[DisagreementFlag] = Field(
        default_factory=list,
        description="Cross-chunk verdict conflict flags",
    )

    # --- origin: declared by the CQ, or reconstructed from structure ---
    # A name that appears only as a property's domain/range and not in that CQ's
    # classes is synthesized as a class component (in OWL domain/range must be
    # classes, and individuals use that name as their anchor too). Downstream must
    # treat "concepts the CQ declared" and "concepts reconstructed from structure to
    # fill a gap" differently (merge and review queue decisions differ), so the two
    # are passed on distinguished.
    synthesized: bool = Field(
        default=False,
        description="Component reconstructed from a domain/range reference, not declared by a CQ",
    )

    # --- provenance: ids of the CQs this component came from (can be several after
    # the name-based merge) ---
    source_cq_ids: list[str] = Field(
        default_factory=list, description="Ids of the CQs this component appears in"
    )

    # NOTE: there is no synonym→representative mapping field (formerly
    # merged_aliases). This module does no semantic merging, so no name is discarded,
    # and choosing the representative belongs to the downstream that owns the global
    # TBox: it records the original name directly in surface_form (ALTER), which is
    # the path individual resolution already queries, so no separate field is needed.


# =========================================================================
# 3. Individual — handed directly to Stage 2
# =========================================================================


class Individual(BaseModel):
    """One ABox individual.

    Carries the Stage-1 class reference (a name string, byte-identical to the
    component name), property values and the source question text.
    """

    model_config = ConfigDict(populate_by_name=True)

    # Stage-1 class reference: exactly the same string as the component name, so an
    # individual always resolves to its class.
    class_name: str = Field(
        ...,
        validation_alias=AliasChoices("class_name", "class_ref"),
        description="Stage-1 class name (byte-identical to the component name)",
    )
    # Unique local name of the individual (if any). Otherwise Stage 2 derives it from
    # the question/values.
    local_name: str | None = None

    # Property values, split into two maps to keep the object/data distinction.
    # property-name -> value.
    object_property_values: dict[str, str] = Field(
        default_factory=dict,
        description="object property name → value (target individual name)",
    )
    data_property_values: dict[str, str] = Field(
        default_factory=dict, description="data property name → literal value"
    )

    # Source question text (required): used by the downstream identity merge.
    source_question: str = Field(..., description="Source CQ question text (required)")
    # provenance: id of the CQ this individual came from (single, since an individual
    # comes from one CQ).
    source_cq_id: str | None = Field(default=None, description="Source CQ id")


# =========================================================================
# 4. Output Envelope
# =========================================================================


class ExtractionOutput(BaseModel):
    """Classifier output envelope: {components, individuals} + run observations.

    Holds the TBox components and the ABox individuals as two separate lists (split
    output).
    """

    # Holds every extracted component regardless of the verdict: verdict is a 'mark',
    # not a gate. Individuals reference component names through class_name and
    # property keys, so filtering here would leave individuals referencing those
    # names with nowhere to attach. Whether to mint is decided by the downstream that
    # owns the global TBox after reading the verdict.
    components: list[AnnotatedComponent] = Field(default_factory=list)
    individuals: list[Individual] = Field(default_factory=list)

    # --- observations (diagnostic) -----------------------------------------
    # Running chunks concurrently interleaves the per-chunk logs, which made
    # observations judged by counting log lines unreliable. So the same numbers are
    # also carried in the response (so they can be judged from the response instead
    # of the logs).
    #
    # Both are **per-chunk sums**: a name appearing in several chunks is counted once
    # per chunk, so they must not be compared directly with the length of the
    # reintegrated `components`. They play no part in pipeline decisions (pure
    # observation): whatever the value, components/individuals go out unchanged.
    n_unmatched: int = Field(
        default=0,
        description=(
            "Number of components left with the default verdict and no domain layer "
            "attached (sum over chunks). Exactly equal to the sum of 'unmatched N' in "
            "the chunk logs. Components of degraded chunks are included too, so tell "
            "the causes apart with n_degraded_batches."
        ),
    )
    n_synthesized_components: int = Field(
        default=0,
        description=(
            "Number of components reconstructed from domain/range references rather "
            "than declared by a CQ (= number of components with synthesized=true). "
            "A large value signals that CQ authoring systematically omits classes "
            "declarations."
        ),
    )
    n_orphan_individuals: int = Field(
        default=0,
        description=(
            "Number of individuals whose class_name matches no class component "
            "(individuals without an anchor). If nonzero, this output contradicts "
            "itself and shows up as a miss downstream."
        ),
    )
    n_degraded_batches: int = Field(
        default=0,
        description=(
            "Number of chunks whose LLM call/parse broke so the whole domain layer "
            "came out empty. If n_unmatched is large and this is nonzero, the cause "
            "is those chunks, not name matching (large chunks get truncated "
            "responses)."
        ),
    )
    n_promoted_conflict: int = Field(
        default=0,
        description=(
            "Number of components that are a property's domain/range but were judged "
            "abox and promoted to conflict (sum over chunks). The evidence stays in "
            "each component's disagreement_flags."
        ),
    )

    # --- domain this batch was actually judged with ------------------------------
    # /target's default_domain is the default; what this batch was judged with is
    # carried only here. A document without domain silently falls back, and without
    # an echo "the document said so" and "not carried, so default" are
    # indistinguishable from outside (same reason as tboxcraft's base_iri echo).
    domain: str | None = Field(
        default=None, description="Domain this batch was judged with (short label)"
    )
    domain_source: str | None = Field(
        default=None, description="Origin of the domain: document | env | settings"
    )
