"""Representative selection and stable IRI minting.

Each cluster becomes one concept with a stable, kind-qualified IRI decoupled from its
(mutable) representative label. Exactly one ``prefLabel``, in English (:data:`PREF_LANG`) —
picked from a string that can be English rather than produced by translating or identifying
one. Alternative surfaces carry whatever language their producer declared; only the
representative label is fixed.
"""

from __future__ import annotations

import re
import unicodedata
from collections import deque
from dataclasses import dataclass

from app.tboxcraft.gloss import desplit_camel
from app.tboxcraft.models import PREF_LANG, UNDETERMINED_LANG, Component

# IRI kind segment — kind-qualified so dual-kind names mint distinct IRIs. Slash namespaces
# only: a hash namespace has no path to put this in, and :func:`mint` says what replaces it there.
KIND_SEGMENT = {
    "class": "class",
    "object_property": "objectProperty",
    "data_property": "dataProperty",
}
_SLUG_RE = re.compile(r"[^A-Za-z0-9_]")


def slugify(name: str) -> str:
    """Sanitize a component name into a valid IRI local part."""
    return _SLUG_RE.sub("_", name)


def is_hash_namespace(base_iri: str) -> bool:
    """Whether ``base_iri`` separates its local names with ``#`` rather than a path segment."""
    return base_iri.endswith("#")


def _kind_cased(local: str, kind: str) -> str:
    """Apply the hash-namespace naming convention: classes UpperCamel, properties lowerCamel.

    Only the first character is forced. The convention exists to keep a class and a property of the
    same name apart once the kind segment is gone, and that job is done by the first character;
    rewriting the rest would change names the curator chose.
    """
    if not local:
        return local
    head = local[0].upper() if kind == "class" else local[0].lower()
    return head + local[1:]


def mint(base_iri: str, kind: str, name: str) -> str:
    """The IRI for ``name`` as ``kind`` under ``base_iri`` — the single minting rule.

    Two namespace forms, decided by the base itself rather than by a flag, because the base IRI is
    what the upstream document declares and it already carries the answer::

        base ends in '#'  ->  {base}{Local}          hash:  http://ex/IPB#Division
        otherwise         ->  {base}/{kind}/{local}  slash: http://ex/demo/class/Division

    The slash form keeps KIND_SEGMENT and therefore one IRI per kind: a name arriving as two kinds
    mints two IRIs. The hash form has nowhere to put the segment, so the naming convention separates
    a class from a property instead — and, because that convention cannot separate an object
    property from a data property (both lowerCamel), those are refused rather than minted. See the
    kind-conflict handling in the update pipeline; this function never resolves a contested name on
    its own.

    Concatenating a hash base with the slash form is what makes this worth centralising: it yields
    ``http://ex/IPB#/class/Division``, a path inside a fragment that no prefix can abbreviate, and
    an IRI is an identity — once minted it is expensive to take back.
    """
    local = slugify(name)
    if is_hash_namespace(base_iri):
        return f"{base_iri}{_kind_cased(local, kind)}"
    if base_iri.endswith("/"):
        # The separator is already there — adding another would mint '…//class/…', an
        # empty path segment inside a permanent identifier (aboxmapper's mint_iri agrees).
        return f"{base_iri}{KIND_SEGMENT[kind]}/{local}"
    return f"{base_iri}/{KIND_SEGMENT[kind]}/{local}"


def namespace_prefix(base_iri: str) -> str:
    """The string every IRI in ``base_iri``'s namespace starts with — and no other namespace does.

    The prefix has to end on the separator, or a sibling is caught: ``http://ex/IPB`` is a prefix
    of every ``http://ex/IPB2/...`` term. A base already ending in '#' or '/' is its own boundary;
    a bare one gets the '/' that :func:`mint` puts there.

    This is what makes "belongs to this document" answerable in SQL. Membership is not stored as a
    column — the IRI *is* the namespace record — so every query that must not see another
    document's terms filters on ``starts_with(id, namespace_prefix(base))``.
    """
    return base_iri if base_iri.endswith(("#", "/")) else base_iri + "/"


def local_of(iri: str) -> str:
    """The local name of ``iri`` — the inverse of :func:`mint`.

    Deliberately does not take the base IRI. ``slugify`` leaves no '/' or '#' in a local name, so
    the last separator in the IRI is always the namespace boundary whichever form minted it, and a
    base-free rule stays correct for an IRI belonging to some *other* namespace — which the store
    hands out (a range pointing at another document's class) and which a base-checked version would
    have to either reject or silently mis-split.

    Every caller used ``iri.rsplit("/", 1)[-1]`` inline before. Under a hash namespace that returns
    ``IPB#hasName`` instead of ``hasName``: no exception, just a name that matches nothing.
    """
    _, sep, tail = iri.rpartition("#")
    return tail if sep else iri.rsplit("/", 1)[-1]


@dataclass
class Concept:
    iri: str
    kind: str
    canonical_name: str
    # ``{lang: label}`` holding exactly one entry — the representative label and its tag, which is
    # :data:`PREF_LANG` except for a term with no English in its input at all.
    #
    # A dict rather than two fields because that is the shape the store takes
    # (``Repository.upsert_concept`` / ``set_pref_labels``, and the schema's one-PREF-per-language
    # key), so the pipeline hands over what it holds instead of re-wrapping at the boundary. The
    # schema would allow several; the pipeline writes one, deliberately — a term with two
    # representative labels has two names, and every consumer that shows one has to pick.
    #
    # Nothing may index this with a literal ``"en"``: the one entry is not always ``en``.
    pref_labels: dict[str, str]
    members: list[Component]

    @property
    def sole_label(self) -> str:
        """The representative label's text, without its tag — for reports and lookups that need
        the concept's name rather than the language it is in. Empty only for a concept built from
        a nameless component, which nothing produces."""
        return next(
            iter(label for _lang, label in sorted(self.pref_labels.items())), ""
        )


def canonical_member(cluster: list[Component]) -> Component:
    """Deterministic representative: a member with a curated ``correct_term`` wins, else the
    lexicographically smallest name."""
    with_term = sorted((c for c in cluster if c.correct_term), key=lambda c: c.name)
    if with_term:
        return with_term[0]
    return min(cluster, key=lambda c: c.name)


def can_be_english(text: str) -> bool:
    """Whether ``text`` could be English — it has letters and every letter is Latin.

    **This is not language identification, and the distinction is the whole point.** "Which language
    is this string" has many possible answers, needs a guess, and is what the language handling
    refuses on principle. "Could this string be English" has one answer and needs no guess in the
    direction that matters: a string with no Latin letter is not English, without exception.

    So the test is one-directional and safe in the direction it is used. It **cannot** reject
    correct English — ``Café Order`` keeps its accented letter (Latin), ``Order — Fragmentary``
    its em dash (not a letter, not examined), ``MDMP`` its all-caps. It **does** let things
    through that are not English: romanised Korean, or a wrong translation spelled in Latin. That
    is deliberate. Those need a person, and a checker that pretended to catch them would be
    making exactly the claim about unread text that put Korean labels in the store as ``@en``.

    Letters only, via Unicode names rather than ``isascii``: punctuation and digits say nothing
    about language, and an ASCII test would reject ``Café``.
    """
    letters = [c for c in text if c.isalpha()]
    return bool(letters) and all(
        unicodedata.name(c, "").startswith("LATIN") for c in letters
    )


def pref_lang_for(text: str) -> str:
    """The tag to put on ``text`` as a representative label — :data:`PREF_LANG`, or ``und``.

    The representative label is English by policy (see :data:`~tboxcraft.models.PREF_LANG`), and
    :func:`pref_label_of` picks a string that can be. When even the component's ``name`` cannot —
    a term with no English anywhere in its input — there is nothing to translate from and nothing
    to infer, so the tag says ``und`` instead of completing the policy by asserting the thing the
    policy exists to stop.

    Rare in practice: a non-Latin name also slugs to an all-underscore IRI local part
    (a four-syllable Hangul name -> ``____``). It is decided here anyway because a rule with a
    hole does not stay unanswered — whatever code runs first answers it, and the code that used
    to run first answered ``en``.
    """
    return PREF_LANG if can_be_english(text) else UNDETERMINED_LANG


def pref_label_of(member: Component) -> tuple[str, str]:
    """The one representative label for ``member`` — ``(lang, text)``.

    Three rungs, in order of how much the string is somebody's deliberate choice:

    1. ``correct_term_en`` — the curator's official term already in English. Nothing to check
       beyond the usual one-directional test; the field's name is its declaration.
    2. ``correct_term`` — the curator's official term, when it can carry this language.
    3. the de-split ``name``.

    A ``correct_term`` that cannot be English is not discarded and not translated: it falls to an
    alternative surface (``surface_channels`` already collects it, so no writer needs to change)
    tagged with the component's declared language.

    That fallback needs no invention because ``name`` is already English by the convention that
    makes it the IRI's local part — the same convention the pipeline relies on every time it
    mints. ``RelationType`` -> ``Relation Type``. It does not beautify: an abbreviation stays an
    abbreviation, so a producer sending ``rel_typ_cd`` gets that as its label. Where that reads
    worse than the term it replaced, the answer is to ask the producer for an English official
    term, not to guess one here.

    :func:`~tboxcraft.io.components.scan_non_english_labels` reports every component this
    demotes, so a label that moved is visible rather than inferred from a diff.
    """
    for curated in (member.correct_term_en, member.correct_term):
        if curated and can_be_english(curated):
            return PREF_LANG, curated
    fallback = desplit_camel(member.name)
    return pref_lang_for(fallback), fallback


def representative_label(member: Component) -> str:
    """Just the text of :func:`pref_label_of`, for the callers that do not need the tag."""
    return pref_label_of(member)[1]


def build_concept(cluster: list[Component], base_iri: str) -> Concept:
    canon = canonical_member(cluster)
    iri = mint(base_iri, canon.kind, canon.name)
    return Concept(
        iri=iri,
        kind=canon.kind,
        canonical_name=canon.name,
        pref_labels=dict([pref_label_of(canon)]),
        members=sorted(cluster, key=lambda c: c.name),
    )


def pick_pref_owner(claims: list[dict], folded: str) -> tuple[dict, str]:
    """Which of the claims on one representative label keeps it — ``({name, iri, ...}, rule)``.

    Takes plain dicts rather than concepts so the warm path can put a stored term on the same
    footing as a batch candidate: a stored side has no Concept, only an IRI whose local part is
    the slug of its name, and the rule has to be the same one either way or the two paths would
    answer the same question differently.

    Tiered, and each tier is applied to the whole set — the first that leaves exactly one claimant
    decides. The point of the tiers is that the answer has a reason: a name that spells out the
    label is the concept the label is *of*, and the abbreviation is the one that borrowed it
    (``Location`` over ``Loc``, ``AreaOfInterest`` over ``AOI``).

    1. the name de-splits to the label. Settles nearly every pair in practice.
    2. the name contains the label — the same test with equality loosened to containment, for
       ``MdmpProcess`` vs ``MilitaryDecisionMakingProcess`` over ``MDMP``, where neither name
       de-splits to it and only one carries it at all.
    3. IRI order, casefolded. Rarely needed; it is here so the rule has no hole
       for iteration order to leak through. Casefolded because the IRIs are CamelCase and a raw
       byte compare would hang the answer on capitals sorting first.
    """
    for rule, matches in (
        (
            "name de-splits to the label",
            lambda c: desplit_camel(c["name"]).casefold() == folded,
        ),
        ("name contains the label", lambda c: folded in c["name"].casefold()),
    ):
        hits = [c for c in claims if matches(c)]
        if len(hits) == 1:
            return hits[0], rule
    return min(claims, key=lambda c: c["iri"].casefold()), "IRI order"


def assign_pref_labels(
    claims: list[dict], *, stored_prefs=None
) -> tuple[dict[str, str], list[dict]]:
    """Give every ``(kind, casefold(label))`` one owner — ``({iri: label}, rows)``.

    ``claims`` are ``{iri, kind, name, label}``. ``stored_prefs(kind, label)`` returns the live
    terms already holding that string as PREF (``[{iri}]``), or None on a path with no store.

    One label per IRI, and ownership keyed without a language, because both are the same fact
    stated twice: a term has one representative label (:data:`~tboxcraft.models.PREF_LANG`), and
    what a contest is about is a *string* — ``Repository.resolve_surface`` matches
    ``lower(surface_text)`` and filters no language, so two terms holding ``Location`` still
    resolve to an arbitrary one of them however their rows are tagged. Keying either axis by
    language would let that pair through while changing nothing about the lookup it breaks.

    Each claim has a ladder of names it can answer to: the label it asked for, then its own
    de-split name. Losing a contest moves it one rung down and it *claims again* — a fallback is a
    claim like any other, not a concession that ends the matter. Giving up after one step is what
    left this in a store:

        class/MDMP                          PREF 'Military Decision Making Process'
        class/MdmpProcess                   PREF 'MDMP'
        class/MilitaryDecisionMakingProcess PREF 'MDMP'

    ``MilitaryDecisionMakingProcess`` loses ``MDMP`` to ``MdmpProcess`` (tier ②, ``mdmp`` is inside
    ``mdmpprocess``), falls back to its own de-split name — and ``MDMP`` is sitting on it, so the
    old rule stopped there and left two terms called ``MDMP``. Re-claiming settles it in three
    steps: the fallback beats ``MDMP`` on tier ① (a name that de-splits to exactly the label), which
    pushes ``MDMP`` onto ``MDMP``, which beats ``MdmpProcess`` on tier ① in turn, which lands on
    ``Mdmp Process``. Every concept ends up named by its own name — the answer a person would give.

    It terminates because a claim only ever moves down its own ladder, which has at most two rungs.
    A claim that loses the last one has nowhere left and keeps the contested label: still a
    collision, reported with ``fallback`` None, and now genuinely irreducible rather than an
    artefact of stopping early.

    The store, when there is one, is immovable — it wins every rung it holds whatever the tier
    says, because taking a live term's label renames a concept that already exists and IRI renewal
    is human-gated (``review.py``). Those rows carry ``tier_favoured_batch`` so "the rule said
    otherwise and we deferred" reaches a person instead of quietly fixing a worse label in place.
    Probing the store at *every* rung is also why a fallback can no longer step on a stored label —
    the old check compared the fallback against this batch only.
    """
    by_iri = {c["iri"]: c for c in claims}
    ladders: dict[str, list[str]] = {}
    for c in claims:
        rungs = [c["label"]]
        own = desplit_camel(c["name"])
        if own.casefold() != c["label"].casefold():
            rungs.append(own)
        ladders[c["iri"]] = rungs

    at = {c["iri"]: 0 for c in claims}
    labels = {c["iri"]: c["label"] for c in claims}
    holder: dict[tuple[str, str], str] = {}
    rows: list[dict] = []
    pending = deque(sorted(claims, key=lambda c: c["iri"]))

    def give_way(loser_iri, kind, contested, kept_iri, rule, favoured_batch):
        at[loser_iri] += 1
        rungs = ladders[loser_iri]
        nxt = rungs[at[loser_iri]] if at[loser_iri] < len(rungs) else None
        rows.append(
            {
                "kind": kind,
                "label": contested,
                "kept": kept_iri,
                "relabelled": loser_iri,
                "fallback": nxt,
                "rule": rule,
                "tier_favoured_batch": favoured_batch,
            }
        )
        if nxt is not None:
            pending.append(by_iri[loser_iri])

    while pending:
        claim = pending.popleft()
        iri, kind = claim["iri"], claim["kind"]
        rung = ladders[iri][at[iri]]
        folded = rung.casefold()
        if not folded:
            continue
        stored = (
            []
            if stored_prefs is None
            else [s for s in stored_prefs(kind, rung) if s["iri"] != iri]
        )
        if stored:
            contenders = [{"iri": iri, "name": claim["name"]}] + [
                # The stored term's name as far as the IRI reveals it: the local part is
                # slugify(name), which only rewrites characters an IRI cannot carry, so for the
                # CamelCase names this rule turns on it is the name.
                {"iri": s["iri"], "name": local_of(s["iri"])}
                for s in stored
            ]
            winner, rule = pick_pref_owner(contenders, folded)
            favoured = winner["iri"] == iri
            give_way(
                iri,
                kind,
                rung,
                stored[0]["iri"],
                f"{rule} favoured the batch; store keeps its label (renaming is human-gated)"
                if favoured
                else rule,
                favoured,
            )
            continue
        incumbent = holder.get((kind, folded))
        if incumbent is None:
            holder[(kind, folded)] = iri
            labels[iri] = rung
            continue
        rival = by_iri[incumbent]
        winner, rule = pick_pref_owner(
            [
                {"iri": iri, "name": claim["name"]},
                {"iri": incumbent, "name": rival["name"]},
            ],
            folded,
        )
        if winner["iri"] == iri:
            holder[(kind, folded)] = iri
            labels[iri] = rung
            give_way(incumbent, kind, rung, iri, rule, False)
        else:
            give_way(iri, kind, rung, incumbent, rule, False)
    return labels, rows


def dedupe_pref_labels(concepts: list[Concept]) -> list[dict]:
    """Give every ``(kind, casefold(label))`` in the batch one owner, in place.

    A representative label is written to ``skos:prefLabel`` and to ``surface_form`` as PREF, and
    resolution does not filter ``form_type`` — so two concepts holding one label is one string
    naming two terms, and the ABox loader drops every individual that uses it. It arises without
    any clustering mistake: ``correct_term`` becomes the label, so ``Loc`` (correct_term
    ``Location``) and ``Location`` cluster apart on their names, correctly, and then both claim
    ``Location`` at the moment the label is applied.

    The loser falls back to its own de-split name rather than being merged. Merging is the better
    end state — claiming one representative label is a strong signal of one concept — but it is
    destructive and not always right (``ElectronicIntelligenceWarfare`` may be *inside*
    ``ElectronicWarfare``, not equal to it), so it belongs to a human. The fallback is cheap and
    reversible and removes the collision now; the merge is queued.

    Runs before the graph is built, not at the DB write: the label goes into the TTL that the gate
    and HermiT see, and fixing only the stored side would leave the two disagreeing about what the
    concept is called.

    Returns one row per contested label — ``{kind, label, kept, relabelled, fallback, rule}`` —
    with ``fallback`` None when the concept had no name left to fall back to, which leaves its
    label alone rather than moving the collision somewhere else. Reported so it is not silent.

    The assignment itself is :func:`assign_pref_labels`, shared with the warm path so a cold and a
    warm batch resolve the same contest identically. There is no store here, so no claim is
    immovable and ``tier_favoured_batch`` is False on every row.
    """
    labels, rows = assign_pref_labels(
        [
            {
                "iri": c.iri,
                "kind": c.kind,
                "name": c.canonical_name,
                "label": c.sole_label,
            }
            for c in concepts
        ]
    )
    for concept in concepts:
        settled = labels[concept.iri]
        if settled != concept.sole_label:
            # Re-tagged, not re-keyed: a fallback is the concept's own de-split name, which can
            # be English where the label it lost was not (and the other way round for a term
            # whose name is not Latin). Keeping the old key would carry a tag that describes a
            # string no longer there.
            concept.pref_labels = {pref_lang_for(settled): settled}
    return rows


def build_concepts(clusters: list[list[Component]], base_iri: str) -> list[Concept]:
    """Mint one concept per cluster; a repeated IRI is a hard error (one IRI per cluster)."""
    concepts: list[Concept] = []
    seen: set[str] = set()
    for cluster in clusters:
        concept = build_concept(cluster, base_iri)
        if concept.iri in seen:
            raise ValueError(f"IRI collision while minting: {concept.iri}")
        seen.add(concept.iri)
        concepts.append(concept)
    return concepts
