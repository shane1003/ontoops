"""Load ``components.json`` into :class:`~tboxcraft.models.Component` objects."""

from __future__ import annotations

import json
from dataclasses import replace
from pathlib import Path

from app.tboxcraft.models import KIND_CLASS, Component, Hierarchy

# NOTE: the loader does NOT filter on ``verdict``. It used to keep only "tbox", written when
# every component in the batch carried that value, so the drop was invisible. It is not the
# loader's decision to make: dropping a component here destroys the anchor for every individual
# that names it, and it happens before ``route_by_verdict`` — which then tallied an
# already-filtered list and reported n_abox=0, so the counter meant to expose this said the
# opposite. Parse everything; the pipeline routes on the verdict (see route_by_verdict).


def fold_case_variants(
    components: list[Component],
) -> tuple[list[Component], list[dict]]:
    """Fold same-kind components whose names differ only in case into one, returning
    ``(components, [{kind, kept, folded, ...}])``.

    ``IPB`` and ``Ipb`` in one kind are one term written two ways. That is an identity fact, not a
    similarity question: their glosses embedded far apart, so no threshold this pipeline could
    choose would ever have merged them, and both minted an IRI and wrote the same surface string —
    leaving the string ambiguous and every individual naming it dropped.

    Which name survives follows the rule :func:`~tboxcraft.represent.canonical_member` already uses
    for cluster representatives — a curated ``correct_term`` wins, then the lexicographically
    smallest name — so the codebase has one convention for "which spelling names the concept"
    rather than two. It is a convention, not a judgment about which spelling is better, which is
    exactly why every fold is reported.

    Nothing the loser carried is dropped. Its name, ``correct_term`` and synonyms become synonyms
    of the survivor (surface-only, never ``duplicates``: a spelling variant among the calibration
    positives is what dragged tau_high down and collapsed the class partition). Its hierarchy
    edges, curated duplicates and CQ ids are unioned in — such a pair can disagree on all three,
    and losing the CQ ids alone would strip provenance from a term that had it.
    """
    from app.tboxcraft.represent import canonical_member

    groups: dict[tuple[str, str], list[Component]] = {}
    for component in components:
        groups.setdefault((component.kind, component.name.casefold()), []).append(
            component
        )

    position = {id(c): i for i, c in enumerate(components)}
    out: list[tuple[int, Component]] = []
    folded: list[dict] = []
    for (kind, _), members in groups.items():
        if len(members) == 1:
            out.append((position[id(members[0])], members[0]))
            continue
        winner = canonical_member(members)
        losers = [m for m in members if m is not winner]
        surfaces = list(winner.dom_synonyms)
        broader, narrower = (
            list(winner.dom_hierarchy.broader),
            list(winner.dom_hierarchy.narrower),
        )
        duplicates, cq_ids = list(winner.dom_duplicates), list(winner.source_cq_ids)
        for loser in losers:
            for value in (loser.name, loser.correct_term, *loser.dom_synonyms):
                if value and value not in surfaces and value != winner.name:
                    surfaces.append(value)
            broader += [b for b in loser.dom_hierarchy.broader if b not in broader]
            narrower += [n for n in loser.dom_hierarchy.narrower if n not in narrower]
            duplicates += [d for d in loser.dom_duplicates if d not in duplicates]
            cq_ids += [c for c in loser.source_cq_ids if c not in cq_ids]
            folded.append(
                {
                    "kind": kind,
                    "kept": winner.name,
                    "folded": loser.name,
                    "carried_synonyms": [
                        s for s in (loser.name, loser.correct_term) if s
                    ],
                    "carried_broader": [
                        b
                        for b in loser.dom_hierarchy.broader
                        if b not in winner.dom_hierarchy.broader
                    ],
                    "carried_cq_ids": len(
                        [
                            c
                            for c in loser.source_cq_ids
                            if c not in winner.source_cq_ids
                        ]
                    ),
                    # The survivor keeps its own; a disagreement here is the reviewer's to see.
                    "verdict_kept": winner.verdict,
                    "verdict_folded": loser.verdict,
                }
            )
        out.append(
            (
                position[id(winner)],
                replace(
                    winner,
                    dom_synonyms=tuple(surfaces),
                    dom_duplicates=tuple(duplicates),
                    dom_hierarchy=Hierarchy(
                        broader=tuple(broader), narrower=tuple(narrower)
                    ),
                    source_cq_ids=tuple(cq_ids),
                ),
            )
        )
    # Input order is preserved so a fold does not reshuffle the batch around it.
    out.sort(key=lambda pair: pair[0])
    folded.sort(key=lambda row: (row["kind"], row["kept"], row["folded"]))
    survivors = [component for _, component in out]

    # ── Reference carry-over ───────────────────────────────────────────────────
    # A fold is a fact about identity, and when an identity changes, the references that
    # pointed at that name (domain, range, hierarchy) must move with it. While only what
    # the loser "held" was moved and what "pointed at it" was left, resolution, being an
    # exact IRI match, orphaned every such reference (the losing spellings were often
    # referenced more than the surviving ones).
    # Under the winner rule (correct_term → lexical order) the producer cannot predict
    # which spelling survives, so the **only** place references can be moved is here,
    # where the fold happens. If the fold table left this place, some users would apply
    # it and some would not, which is exactly the shape this defect had.
    rename = {
        row["folded"].casefold(): row["kept"]
        for row in folded
        if row["kind"] == KIND_CLASS
    }
    if rename:

        def _carried(names: list[str]) -> list[str]:
            # A list that referenced both folded spellings ("IPB, Ipb") gets a duplicate
            # when folded to one name, so dedupe preserving order.
            return list(dict.fromkeys(rename.get(n.casefold(), n) for n in names))

        for component in survivors:
            if component.domain:
                component.domain = ", ".join(_carried(component.domain_values()))
            if component.range_raw:
                component.range_raw = ", ".join(_carried(component.range_values()))
            hier = component.dom_hierarchy
            broader = tuple(_carried(list(hier.broader)))
            narrower = tuple(_carried(list(hier.narrower)))
            if broader != hier.broader or narrower != hier.narrower:
                component.dom_hierarchy = Hierarchy(
                    broader=broader, narrower=narrower
                )

    return survivors, folded


def scan_kind_conflicts(components: list[Component]) -> list[dict]:
    """One name judged as two different kinds — ``[{name, kinds, sides, ...}]``, deterministic.

    ``usesWeapon`` arriving as both an object property (range ``Weapon``) and a data property
    (range ``xsd:string``) is a contradiction about what the term *is*. Dual-kind names mint
    separate IRIs on purpose, and the two never co-cluster because clustering is
    kind-partitioned, so both survive to mint and sit side by side with nothing raised: the IRI
    collision detector only fires on an exact IRI match, and the kind segment differs.

    Read from the components rather than from the minted concepts, for two reasons. A concept's
    local name follows its cluster representative, so a conflicting member whose cluster is named
    after somebody else stops being visible by name (``hasPurpose`` does this). And a single-word
    name equals its own label, so it is written only as a PREF row and never reaches
    :func:`~tboxcraft.mapping.scan_surface_collisions` at all (``describes`` does this) — that scan
    reports what the *resolver* can see, which is a different question from what was minted.

    Which side is which is not decided here. The object side usually carries a real class range and
    the data side ``xsd:string``, so collapsing to data would turn "uses a weapon" into "a weapon is
    a string" — but ``hasLegitimacyStatus`` may genuinely be data. That is a human call, which is
    why both sides' ranges travel with the row.
    """
    conflicts: list[dict] = []
    for _folded, by_kind in sorted(_by_name_kind(components).items()):
        contesting = _contesting_kinds({k: c.name for k, c in by_kind.items()})
        if len(contesting) < 2:
            continue
        by_kind = {k: by_kind[k] for k in contesting}
        sides = [
            {
                "kind": kind,
                "uid": c.uid,
                "domain": c.domain,
                "range": c.range_raw,
                # Provenance travels with the row: the reviewer decides by looking at the questions
                # the two readings came from, and nothing downstream can recover them later.
                "source_cq_ids": list(c.source_cq_ids),
            }
            for kind, c in sorted(by_kind.items())
        ]
        names = sorted({c.name for c in by_kind.values()})
        # `name` is one real spelling (the smallest) rather than the folded key, so a reader
        # searching for it finds a term. `names` carries the rest: when the two readings are also
        # spelled differently that is part of what the reviewer is looking at, not noise.
        conflicts.append(
            {"name": names[0], "names": names, "kinds": sorted(by_kind), "sides": sides}
        )
    return conflicts


def scan_cross_batch_kind_conflicts(components: list[Component], stored) -> list[dict]:
    """Kind conflicts between this batch and terms an earlier batch already minted.

    ``stored`` is ``{name: [{kind, iri, range}]}`` from the store (see
    :meth:`Repository.stored_terms_by_local_name`). A conflict is a batch component whose name is
    already minted under a *different* kind: neither the in-batch scan (which only sees the batch)
    nor the IRI collision detector (kinds mint distinct IRIs) catches it. The individuals do not go
    missing — each kind has its own anchor — but the contradiction is never put to a human, and the
    in-batch case is queued loudly while this one is silent. This closes that asymmetry.

    Names already conflicting *within* the batch are skipped: :func:`scan_kind_conflicts` has them,
    and queuing both would file one contest twice.
    """
    from app.tboxcraft.represent import local_of

    in_batch = {c["name"].casefold() for c in scan_kind_conflicts(components)}
    # Folded onto the same key the batch side uses: the store may hold a spelling no component in
    # this batch wrote, and an exact key would call that a different name.
    stored_folded: dict[str, list[dict]] = {}
    for stored_name, rows in stored.items():
        stored_folded.setdefault(stored_name.casefold(), []).extend(rows)

    conflicts: list[dict] = []
    for folded, by_kind in sorted(_by_name_kind(components).items()):
        if folded in in_batch or folded not in stored_folded:
            continue
        # One name per kind for the convention test; the stored spelling is its IRI local part.
        names_by_kind = {k: c.name for k, c in by_kind.items()}
        for row in stored_folded[folded]:
            names_by_kind.setdefault(row["kind"], local_of(row["iri"]))
        contesting = set(_contesting_kinds(names_by_kind))
        if len(contesting) < 2:
            continue

        stored_kinds = {s["kind"] for s in stored_folded[folded]}
        batch_sides = [
            {
                "kind": kind,
                "uid": c.uid,
                "domain": c.domain,
                "range": c.range_raw,
                "source_cq_ids": list(c.source_cq_ids),
                "in_batch": True,
            }
            for kind, c in sorted(by_kind.items())
            if kind not in stored_kinds and kind in contesting
        ]
        if not batch_sides:
            continue  # every batch kind already stored — a re-ingest, not a new conflict
        stored_sides = [
            {"kind": s["kind"], "iri": s["iri"], "range": s["range"], "in_batch": False}
            for s in sorted(stored_folded[folded], key=lambda s: s["kind"])
            if s["kind"] in contesting
        ]
        sides = batch_sides + stored_sides
        names = sorted(
            {by_kind[k].name for k in by_kind if k in contesting}
            | {local_of(s["iri"]) for s in stored_sides}
        )
        conflicts.append(
            {
                "name": names[0],
                "names": names,
                "kinds": sorted({s["kind"] for s in sides}),
                "sides": sides,
            }
        )
    return conflicts


def _contesting_kinds(names_by_kind: dict[str, str]) -> list[str]:
    """Which kinds are actually in contest over one folded name.

    All of them, except for one carve-out: a class whose spelling differs from every property
    side. ``ActionType`` (class) beside ``actionType`` (data property) is the PascalCase-class /
    camelCase-property convention doing its job, not a claim that one term is two things — the
    two names are different names, and the earlier decision to leave them alone was right about
    that shape. It is only wrong about property-vs-property, where both sides are camelCase and a
    case difference is a spelling inconsistency rather than a convention. The case-variant pairs
    seen in practice are that kind: all properties.

    An identical spelling always contests, class or not — ``Weapon`` as a class and as an object
    property is the same name twice whatever the convention says.
    """
    klass = names_by_kind.get("class")
    kinds = sorted(names_by_kind)
    if klass is not None and all(
        n != klass for k, n in names_by_kind.items() if k != "class"
    ):
        return [k for k in kinds if k != "class"]
    return kinds


def _by_name_kind(components: list[Component]) -> dict[str, dict[str, Component]]:
    """``{casefold(name): {kind: component}}`` — the grouping both kind-conflict scans read.

    Keyed casefolded because the name a component arrives with is not the name it keeps.
    :func:`fold_case_variants` runs first and merges same-kind case variants onto whichever
    spelling ``canonical_member`` picks, so a pair that was written identically can be two
    spellings by the time this sees it. Exactly that happened to
    ``coordinatesRistaIntegration``: the data property folded onto the ``RISTA`` spelling of a
    sibling and stopped matching the object property it genuinely contradicted, and the conflict
    neither queued nor resolved — it disappeared, leaving both terms ACTIVE side by side.

    Same-kind case variants cannot normally reach here (the fold merged them), so the tie-break
    below only matters to a direct caller; smallest name, to stay deterministic.
    """
    by_name: dict[str, dict[str, Component]] = {}
    for component in components:
        slot = by_name.setdefault(component.name.casefold(), {})
        held = slot.get(component.kind)
        if held is None or component.name < held.name:
            slot[component.kind] = component
    return by_name


def parse_components(raw: dict) -> list[Component]:
    """Parse an already-loaded components document (``{components:[...]}``).

    Every component is returned whatever its verdict. Only the TBox ``components`` key is read;
    other top-level keys (e.g. the ABox ``individuals`` block) are ignored by design. Used for
    service-to-service calls that POST the document inline; :func:`load_components` is the
    file-backed wrapper.
    """
    return [Component.from_dict(item) for item in raw["components"]]


def load_components(path: str | Path) -> list[Component]:
    """Return every component in the document at ``path``.

    Raises ``KeyError``/``TypeError`` loudly on unexpected shapes rather than silently
    dropping fields — the finalized schema is still pending.
    """
    raw = json.loads(Path(path).read_text(encoding="utf-8"))
    return parse_components(raw)


def scan_non_english_labels(components: list[Component]) -> list[dict]:
    """Curated terms that could not be the representative label — ``[{uid, kind, name, ...}]``.

    The representative label is one per term and English (:data:`~tboxcraft.models.PREF_LANG`).
    A ``correct_term`` that cannot be English does not become it; it stays on the term as an
    alternative surface and the label falls back to the de-split name. Nothing is discarded and
    nothing is translated — see :func:`~tboxcraft.represent.pref_label_of`.

    Reported rather than done quietly, because "the curator's chosen term is not this term's
    displayed name" is a fact somebody has to be able to see. It is not an error and does not
    stop anything: a producer sending Korean official terms is doing what it is for. It is a
    count and a list, so the day it appears where nobody expected it — a batch that used to be
    all-English suddenly demoting half its labels — the number says so.

    ``label_is_english`` distinguishes the two shapes that land here. False means the fallback is
    not English either, so the term has no English anywhere in its input and its label is tagged
    ``und``: rarer, worse, and the one worth looking at first.
    """
    from app.tboxcraft.represent import can_be_english, pref_label_of

    rows: list[dict] = []
    for component in sorted(components, key=lambda c: c.uid):
        if not component.correct_term or can_be_english(component.correct_term):
            continue
        # A component carrying `correct_term_en` is not this case: the curator supplied both
        # languages and the label is still one of their choices. What this reports is the label
        # having been *derived* — nobody chose `battle Equipment Status_equipment Status`.
        if component.correct_term_en and can_be_english(component.correct_term_en):
            continue
        lang, label = pref_label_of(component)
        rows.append(
            {
                "uid": component.uid,
                "kind": component.kind,
                "name": component.name,
                "demoted_term": component.correct_term,
                "demoted_lang": component.surface_lang,
                "label": label,
                "label_lang": lang,
                "label_is_english": can_be_english(label),
            }
        )
    return rows


def scan_overridden_lang_declarations(components: list[Component]) -> list[dict]:
    """Surfaces whose ``en`` declaration was refused — ``[{uid, kind, name, surface, source}]``.

    See :meth:`~tboxcraft.models.Component.declared_lang_for`: a producer declaring ``en`` over a
    string with no Latin letter is stating something that cannot be true, so the row is written
    ``und`` instead. This is the count of that happening.

    It exists because the override is the kind of fix that hides the problem it fixes. With it and
    no number, a store full of correct tags looks identical whether the producers started sending
    the right thing or whether this pipeline has been quietly covering for them every batch since.
    A non-zero count is a message for the producer, not for us.

    Zero today: no producer declares a language at all, so nothing reaches the ``en`` branch.
    ontoagent cannot produce this shape either (its character rule outvotes the model's tag), which
    leaves a relational-database producer as the only way for it to become non-zero.
    """
    rows: list[dict] = []
    for component in sorted(components, key=lambda c: c.uid):
        # Read off `surface_channels`, which carries both the claim and what was written, rather
        # than re-deriving the rule here. A counter with its own copy of the rule drifts from the
        # writer, and the drift reads as the producer having fixed something.
        for surface, declared, written, source in component.surface_channels():
            if declared != written:
                rows.append(
                    {
                        "uid": component.uid,
                        "kind": component.kind,
                        "name": component.name,
                        "surface": surface,
                        "source": source,
                        "declared": declared,
                        "written": written,
                    }
                )
    return rows


def scan_slug_collisions(components: list[Component]) -> list[dict]:
    """Distinct names this batch would mint onto one IRI — ``[{name, names, kind, slug, sides}]``.

    ``slugify`` rewrites every character an IRI local part cannot carry to ``_``, which is not
    injective outside ASCII: any two four-syllable Hangul names both become ``____``. Two concepts then share an identity, and sharing an identity is not a thing a review
    can undo later — the second ``upsert_concept`` merges onto the first and whichever name lost
    is gone with no row saying so.

    Detected rather than prevented by refusing non-ASCII, which is the fix that suggests itself and
    is the wrong one. RFC 3987 IRIs and Turtle's ``PN_LOCAL`` both admit Hangul, so a character ban
    forbids what the standards allow and has to be walked back the moment a producer sends what it
    is entitled to send. Transliteration is worse: several romanisation schemes disagree, and the
    choice is not recoverable from the result. What actually breaks is the slug being
    many-to-one, so that is what is tested; names whose slugs still differ are left alone.

    Keyed by kind because the kind is part of the IRI under a slash namespace and of the naming
    convention under a hash one; two kinds sharing a slug is the kind-conflict question, which
    :func:`scan_kind_conflicts` already asks and answers differently.

    Zero rows in practice and zero in the fixed test data. That is not a reason to leave it out —
    it is currently zero because "IRIs are ASCII, labels are multilingual" holds by convention,
    and nothing enforces that convention on the producers. This is the tripwire for the day it
    stops holding.
    """
    from app.tboxcraft.represent import slugify

    by_slug: dict[tuple[str, str], dict[str, Component]] = {}
    for component in components:
        slot = by_slug.setdefault((component.kind, slugify(component.name)), {})
        held = slot.get(component.name)
        if held is None or component.uid < held.uid:
            slot[component.name] = component

    collisions: list[dict] = []
    for (kind, slug), by_name in sorted(by_slug.items()):
        if len(by_name) < 2:
            continue
        names = sorted(by_name)
        collisions.append(
            {
                "name": names[0],
                "names": names,
                "kind": kind,
                "kinds": [kind],
                "slug": slug,
                "sides": [
                    {
                        "kind": kind,
                        "uid": c.uid,
                        "name": c.name,
                        "domain": c.domain,
                        "range": c.range_raw,
                        "source_cq_ids": list(c.source_cq_ids),
                        "in_batch": True,
                    }
                    for _n, c in sorted(by_name.items())
                ],
            }
        )
    return collisions


def scan_cross_batch_slug_collisions(components: list[Component], stored) -> list[dict]:
    """Batch names whose IRI is already minted and cannot be shown to be the same name.

    ``stored`` is ``{name: [{kind, iri, range}]}`` from
    :meth:`Repository.stored_terms_by_local_name`, which finds a term by rebuilding its IRI from
    ``(base, kind, slug)``. That reconstruction is exact only while the slug determines the name.
    When it does not — when ``slugify(name) != name`` — a hit means "some earlier name slugged to
    this", and nothing in the IRI says which. A Hangul name finds the term another Hangul name of
    the same length minted and reads it as its own earlier ingest.

    So the test is not whether the two names match (they cannot be compared: the store keeps the
    slug, not the name) but whether the question is answerable at all. A faithfully-slugged name
    is its own local part and a hit on it is a genuine re-ingest; a lossy one is withheld.

    Uses the ``stored`` dict the cross-batch kind scan already fetched — no second query, and no
    new argument to thread through the pipeline.
    """
    from app.tboxcraft.represent import slugify

    stored_slugs = {name.casefold() for name in stored}
    collisions: list[dict] = []
    for component in sorted(components, key=lambda c: c.uid):
        slug = slugify(component.name)
        if slug == component.name or slug.casefold() not in stored_slugs:
            continue
        rows = [
            r
            for name, rs in stored.items()
            if name.casefold() == slug.casefold()
            for r in rs
            if r["kind"] == component.kind
        ]
        if not rows:
            continue
        collisions.append(
            {
                "name": component.name,
                "names": [component.name],
                "kind": component.kind,
                "kinds": [component.kind],
                "slug": slug,
                "sides": [
                    {
                        "kind": component.kind,
                        "uid": component.uid,
                        "name": component.name,
                        "domain": component.domain,
                        "range": component.range_raw,
                        "source_cq_ids": list(component.source_cq_ids),
                        "in_batch": True,
                    },
                    *(
                        {
                            "kind": r["kind"],
                            "iri": r["iri"],
                            "range": r["range"],
                            "in_batch": False,
                        }
                        for r in sorted(rows, key=lambda r: r["iri"])
                    ),
                ],
            }
        )
    return collisions


def withheld_slug_collision_uids(collisions: list[dict]) -> set[str]:
    """Component uids that must NOT be minted because their IRI would not identify them.

    Every side, for the same reason :func:`withheld_kind_conflict_uids` withholds every side: the
    batch carries no evidence for which name the shared IRI ought to mean, so minting one freezes
    an arbitrary choice into an identity — and an identity is the one thing here that a later
    review cannot take back.

    Unlike the kind-conflict case this applies under both namespace forms. A slash namespace keeps
    the kind segment, which separates kinds, but the local part is the slug in either form and a
    non-injective slug collides with itself whatever precedes it.
    """
    return {s["uid"] for c in collisions for s in c["sides"] if s.get("uid")}


def withheld_kind_conflict_uids(conflicts: list[dict]) -> set[str]:
    """Component uids that must NOT be minted because their name is contested across kinds.

    Under a hash namespace the kind segment is gone from the IRI and the naming convention only
    separates a class from a property — an object property and a data property of one name are
    both lowerCamel and collapse onto a single IRI. OWL 2 DL requires the two to be disjoint, so
    that IRI is not a term the reasoner will accept; HermiT rejects the ontology rather than
    picking a side.

    So neither side is minted. Not "one of them": the batch offers no evidence for which reading
    is right — in practice the data-property side was always ``xsd:string``, which is
    what an unresolved range degrades to, but that does not make the object-property side correct
    either. Minting one would freeze an arbitrary choice into an identity.

    Both sides are still queued (the caller does that), and the ABox loader resolves ACTIVE terms
    only — so an individual naming a withheld term fails resolution and lands in the downstream
    ``miss`` count, where the consuming gate stops. Withholding does not lose the stop, it moves
    it to a place that reports the whole batch instead of aborting mid-ingest.
    """
    return {s["uid"] for c in conflicts for s in c["sides"] if s.get("uid")}


def demotable_kind_conflict_iris(conflicts: list[dict]) -> list[dict]:
    """Already-minted terms on the stored side of a cross-batch conflict — ``[{iri, kind, name}]``.

    A cross-batch conflict has one side already in the store from an earlier batch, so "mint
    neither" is not reachable for it: that term exists. Refusing only the new side would be exactly
    the arbitrary choice above, decided by arrival order rather than evidence.

    The stored side is therefore demoted to DRAFT — minted but not asserted. Every consumer filters
    on ``status = 'ACTIVE'`` (the views, the ABox resolver), so a DRAFT term stops being resolvable
    while its row, its FKs and its provenance stay put, and review can restore it. DEPRECATED would
    be wrong: that status means "superseded", a judgement someone made, and here nobody has judged
    anything yet.
    """
    return [
        {"iri": s["iri"], "kind": s["kind"], "name": c["name"]}
        for c in conflicts
        for s in c["sides"]
        if s.get("in_batch") is False and s.get("iri")
    ]


def kind_conflict_reason(conflict: dict, minted_iris: list) -> str:
    """The reason in the reviewer payload, **worded by the actual minting state**.

    While "both are minted" was a literal, the sentence went out in the withheld branch
    too (hash ns: nothing minted), contradicting ``minted_iris=[null, null]`` in the same
    payload, for every KIND_CONFLICT. The two states
    call for opposite work from a reviewer: if both exist, pick one and delete it; if
    neither exists, decide the kind and mint it. Steering the latter as the former leaves
    the term missing for good (worse, since what is absent nobody sees). So the sentence is
    computed from minted_iris: the structured field and the prose cannot diverge, and
    cold/warm use the same sentence (two duplicated f-strings folded into this function).

    Both spellings ride in the names when they differ: the reviewer would otherwise read
    one name and find terms that do not share it; but now it says "go find them" only in
    the branch where those terms actually exist.
    """
    names = " / ".join(conflict["names"])
    kinds = " and ".join(conflict["kinds"])
    if all(m is None for m in minted_iris):
        return (
            f"{names} was judged as {kinds}; the hash namespace cannot hold both, "
            "so neither was minted — pick a kind and mint it, or the term stays absent"
        )
    if any(m is None for m in minted_iris):
        existing = " / ".join(m for m in minted_iris if m)
        return (
            f"{names} was judged as {kinds}; only {existing} exists — the other side "
            "was withheld and needs a decision to be minted"
        )
    return (
        f"{names} was judged as {kinds}; both are minted, so the "
        "ontology asserts both readings at once"
    )
