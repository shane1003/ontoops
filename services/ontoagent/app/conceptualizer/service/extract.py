"""Stage-1 extraction: CQList → annotated components (structure) + ABox individuals.

This module does two things.

1. Loads a CQ JSON file into the `CQDocument` input model (a self-contained loader;
   check the file exists → parse json → pydantic validation).
2. From a CQ batch
   (a) builds an `AnnotatedComponent` for each proposed class/object-property/
       data-property (structure only; the language/domain layers are filled later by
       the expert), merging equal names (domain/range union).
   (b) builds `Individual`s: each CQ's **items holding a value** become individuals.
       A value is inherently an ABox fact, so it is not gated on the component
       verdict (usually tbox = schema). The verdict stays on the component and is
       passed on; it never blocks an individual from existing.

Batch IDs, reintegration and disk writes are runner.py's responsibility; this module
does pure extraction only.
"""

from __future__ import annotations

import json
from pathlib import Path

from app.conceptualizer.schemas import (
    AnnotatedComponent,
    CQDocument,
    Individual,
)
from app.conceptualizer.service.merge_util import merge_dedup

# =========================================================================
# 1. CQ JSON loading
# =========================================================================


def load_cq_document(path: str | Path) -> CQDocument:
    """Loads and validates one CQ JSON file into the CQDocument input model.

    Checks the file exists → parses json → validates with pydantic, using CQDocument from
    app.conceptualizer.schemas.

    Raises:
        FileNotFoundError: If the file does not exist.
    """
    path = Path(path)
    if not path.exists():
        raise FileNotFoundError(f"CQ file not found: {path}")
    data = json.loads(path.read_text(encoding="utf-8"))
    return CQDocument(**data)


# =========================================================================
# 2. Component extraction (structure only, merged by name)
# =========================================================================


def _join(values: list[str]) -> str | None:
    """Expresses a domain/range union as the component's single slot string.

    When a property of one name has different domain/range values across CQs, the
    union is kept. AnnotatedComponent.domain/range is a single string slot, so the
    union is joined with ', '.
    """
    merged = merge_dedup(values)
    if not merged:
        return None
    return ", ".join(merged)


def extract_components(cqs: list) -> list[AnnotatedComponent]:
    """Builds a structure-only AnnotatedComponent list from the CQs (merged by name).

    Components with the same name are merged into one, and property domain/range
    values are collected as a union. The language/domain layers and the verdict are not filled here
    (the expert fills them later).

    A name a property points to as domain/range but that is not declared in that CQ's
    `classes` is **synthesized** as a class component marked `synthesized=True`. In
    OWL a property's domain/range must be a class (the same fact turns an abox
    verdict back into conflict), and individuals use that name as their anchor too:
    without synthesis, one line the author declared as range but left out of classes
    costs the individual its anchor. "What in the existing ontology is this the same as" is a
    downstream judgment, but "did a CQ declare the names it references" is a fact
    within this batch, so it is restored here. The declared/synthesized distinction
    travels as a flag, because downstream merge decisions must treat the two
    differently.

    Returns:
        The components in order of first appearance (if a class appears before a
        property, that order is kept).
    """
    # name -> component (keyed by (kind, name) so namespaces stay separate per kind).
    order: list[tuple[str, str]] = []
    by_key: dict[tuple[str, str], AnnotatedComponent] = {}
    # Temporary accumulators collecting the property domain/range unions.
    domains: dict[tuple[str, str], list[str]] = {}
    ranges: dict[tuple[str, str], list[str]] = {}

    def ensure(kind: str, name: str, comment: str | None, cq_id: str) -> AnnotatedComponent:
        key = (kind, name)
        comp = by_key.get(key)
        if comp is None:
            comp = AnnotatedComponent(name=name, kind=kind, comment=comment)
            by_key[key] = comp
            order.append(key)
            domains[key] = []
            ranges[key] = []
        else:
            # Fill in the first non-empty comment (refine rule).
            comp.comment = comp.comment or comment
        # provenance: accumulate the ids of the CQs this component came from (deduplicated).
        if cq_id and cq_id not in comp.source_cq_ids:
            comp.source_cq_ids.append(cq_id)
        return comp

    def ensure_class_ref(name: str | None, cq_id: str) -> None:
        """Ensures a class name a property points to exists as a component (synthesized if absent).

        If it already exists it is left alone: the provenance of a component created by
        a declaration is not inflated with CQs that 'merely referenced' it.
        """
        # Names with a namespace prefix (xsd:string etc.) are not classes of this batch.
        if not name or ":" in name:
            return
        if ("class", name) in by_key:
            return
        ensure("class", name, None, cq_id).synthesized = True

    for cq in cqs:
        for c in cq.classes:
            # A declaration clears the synthesized mark (a reference can precede the
            # declaration).
            ensure("class", c.name, c.comment, cq.id).synthesized = False
        for op in cq.object_properties:
            key = ("object_property", op.name)
            ensure("object_property", op.name, op.comment, cq.id)
            if op.domain:
                domains[key].append(op.domain)
            if op.range:
                ranges[key].append(op.range)
            # Both domain and range of an object property are classes.
            ensure_class_ref(op.domain, cq.id)
            ensure_class_ref(op.range, cq.id)
        for dp in cq.data_properties:
            key = ("data_property", dp.name)
            ensure("data_property", dp.name, dp.comment, cq.id)
            if dp.domain:
                domains[key].append(dp.domain)
            if dp.range:
                ranges[key].append(dp.range)
            # A data property range is a literal type (xsd:*), not a class: domain only.
            ensure_class_ref(dp.domain, cq.id)

    # Write the unions into the domain/range slots.
    for key in order:
        comp = by_key[key]
        if domains[key]:
            comp.domain = _join(domains[key])
        if ranges[key]:
            comp.range = _join(ranges[key])

    return [by_key[k] for k in order]


# =========================================================================
# 3. Individual extraction (TBox/ABox decided by the domain verdict)
# =========================================================================


def extract_individuals(cqs: list) -> list[Individual]:
    """Extracts Individuals from the CQs.

    Individual extraction is decided by **the presence of a value**: a value is
    inherently an instance-level ABox fact, so even when the component is (rightly)
    judged `tbox`, its value becomes an individual/literal. The verdict does not gate
    whether an individual exists: it is only classification information that stays on
    the component and is passed on (otherwise every class and property would be tbox
    and there would be zero individuals).

    Items holding a value in one CQ:
      - object property with a value: an individual of the target (range) class
        (value = target name) + the property value recorded on the subject (domain)
        individual.
      - data property with a value: an individual of the subject (domain) class + the
        literal value recorded.

    Each individual carries class_name (byte-identical to the component name), the property values
    and source_question (= the CQ question text), so every individual resolves to its class. Naming
    individuals and the identity merge are handled by downstream modules (outside this module).
    """
    return [ind for cq in cqs for ind in _individuals_for_cq(cq)]


def _individuals_for_cq(cq) -> list[Individual]:
    """Builds individuals from the value-holding items of one CQ.

    Within one CQ, individuals of the same class are gathered into one, carrying
    several property values on that one individual.
    """
    individuals: list[Individual] = []
    # class_name -> Individual (the individuals within this CQ).
    subjects: dict[str, Individual] = {}

    def subject_for(class_name: str) -> Individual:
        ind = subjects.get(class_name)
        if ind is None:
            # provenance: record this CQ on its individuals as source_cq_id.
            ind = Individual(
                class_name=class_name, source_question=cq.question, source_cq_id=cq.id
            )
            subjects[class_name] = ind
            individuals.append(ind)
        return ind

    # object property with value → individual of the target (range) class (value =
    # target name) + property value recorded on the subject individual. Independent of
    # the verdict (a value is an ABox fact).
    for op in cq.object_properties:
        if not op.value:
            continue
        target_class = op.range or op.domain
        if target_class:
            # The target individual the value points to. Stamping op.value into
            # local_name lets Stage 2 wire value → target individual (local_name=None
            # breaks the wiring).
            target_ind = subject_for(target_class)
            if target_ind.local_name is None:
                target_ind.local_name = op.value
        if op.domain:
            # Put the property value (target name) on the subject (domain) individual.
            subject_for(op.domain).object_property_values[op.name] = op.value

    # data property with value → subject (domain) class individual + literal. Independent
    # of the verdict.
    for dp in cq.data_properties:
        if not dp.value or not dp.domain:
            continue
        subject_for(dp.domain).data_property_values[dp.name] = dp.value

    return individuals
