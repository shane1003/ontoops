"""Domain models for TBox components and competency questions.

These mirror the current ``components.json`` batch shape (the finalized schema is still
pending). The loaders in :mod:`tboxcraft.io` build these from raw
JSON/CSV and are the only place that touches the on-disk format.
"""

from __future__ import annotations

from dataclasses import dataclass, field

from app.core.lang import UNDETERMINED_LANG

# Component kinds, matching the ``kind`` field in components.json.
KIND_CLASS = "class"
KIND_OBJECT_PROPERTY = "object_property"
KIND_DATA_PROPERTY = "data_property"
PROPERTY_KINDS = frozenset({KIND_OBJECT_PROPERTY, KIND_DATA_PROPERTY})

# RFC 5646's "undetermined". The language tag for a string whose producer did not say what
# language it is in — which is every string today, because no producer sends one yet.
#
# Not "en", which is what the pipeline used to assert. That assertion was free of consequence
# only while every producer happened to send English: the RDB path does not, and its Korean
# labels ended up stored as ``@en``, a false claim inside the ontology rather than a missing
# one. Not inferred from character range either — a label can be one string in two scripts,
# and guessing is the thing this pipeline refuses everywhere else.
#
# It is a single constant because it is a placeholder with an end date: once the producers
# carry the language (the relational-database producer, and ontoagent for names and notes), real
# values arrive through ``Component.lang`` / ``notes_lang`` and this stops being reached.
# The value itself is the wire contract shared with the producer — imported from core.lang
# (top of file) so producer and consumer cannot drift apart.

# The representative label is one per term and it is English. Not a guess about the text — a
# policy, and one the pipeline can keep without translating or identifying anything, because
# ``name`` is already English by the convention that makes it the IRI's local part.
#
# A term may hold as many alternative surfaces in as many languages as it likes; those are
# ``surface_form`` rows tagged with what their producer declared, and the views expose them per
# language (``AlterTerms`` is ``jsonb_object_agg(lang, ...)``). Only the representative
# one is fixed, which is what keeps ``Classes."PrefTerm_en"`` meaning something.
#
# The one case that cannot satisfy this is a term with no English anywhere in its input; see
# :func:`~tboxcraft.represent.pref_lang_for`, which answers ``und`` there rather than lying.
PREF_LANG = "en"


def _clean(value: str | None) -> str | None:
    """Normalize empty/whitespace strings to ``None`` (``correct_term`` is often ``""``)."""
    if value is None:
        return None
    stripped = value.strip()
    return stripped or None


@dataclass(frozen=True)
class Hierarchy:
    broader: tuple[str, ...] = ()
    narrower: tuple[str, ...] = ()

    @classmethod
    def from_dict(cls, raw: dict | None) -> Hierarchy:
        raw = raw or {}
        return cls(
            broader=tuple(raw.get("broader", []) or []),
            narrower=tuple(raw.get("narrower", []) or []),
        )


@dataclass
class Component:
    """One adjudicated TBox candidate term."""

    name: str
    kind: str
    domain: str | None = None
    range_raw: str | None = None
    comment: str | None = None

    # NOTE: there is no linguistic_layer. It was a retired input channel — no current producer
    # sends one, so the four fields that read it were always empty and the code around them read
    # as live. Removed to say that plainly, not because synonyms are unwanted: they arrive as
    # ``domain_layer.synonyms`` below instead. DomainLayer is extra="allow" upstream, so that
    # value survives the trip without a schema change on the producer's side.

    # domain_layer
    correct_term: str | None = None
    # The curator's official term rendered in English, when the producer has one. A field of its
    # own rather than "put the English in correct_term": the representative label is English
    # (:data:`PREF_LANG`) but the curated term often is not, and a component carrying both has
    # nowhere to put the second one. Folding the English into ``correct_term`` pushes the Korean
    # down into ``synonyms``, where it inherits :attr:`surface_lang` — now ``en`` — and gets
    # refused to ``und`` by :meth:`declared_lang_for`. The guard is right and the outcome is still
    # wrong: the producer knew it was ``ko`` and the store ends up saying nobody knew.
    #
    # Keeping them apart lets ``surface_lang`` go on meaning "the curated channel's language",
    # which is the coarse-but-honest model this class is built on. English needs no tag of its
    # own — the field name is the declaration — and is checked like any other declaration.
    correct_term_en: str | None = None
    # Two different claims, deliberately kept apart:
    #   duplicates — "a SEPARATE component exists and is the same concept". Resolved against
    #     other components' names, so it also drives threshold calibration (the only field that
    #     links distinct components), cluster merge edges, and approval's authoritative-merge
    #     shortcut. A spelling variant put here is either ignored or, if it happens to collide
    #     with a real component name, merges two concepts while bypassing similarity checks.
    #   synonyms — "this concept is ALSO CALLED this". Needs no component of its own and feeds
    #     surface forms only, which is what resolves an individual to its class downstream.
    dom_duplicates: tuple[str, ...] = ()
    dom_synonyms: tuple[str, ...] = ()
    dom_hierarchy: Hierarchy = field(default_factory=Hierarchy)
    agreement: str | None = None
    dom_notes: str | None = None
    # The language of the strings this component carries, as its producer declared it — None
    # when it declared nothing, which is what :attr:`surface_lang` / :attr:`comment_lang` turn into
    # ``und``. Two fields rather than one because they arrive from different places and on
    # different schedules: the surfaces (``name``, ``correct_term``, synonyms, duplicates) are
    # what the label is made of, while ``dom_notes`` is domain-expert rationale written
    # separately and often in another language than the term it annotates.
    lang: str | None = None
    notes_lang: str | None = None

    verdict: str = "tbox"
    disagreement_flags: tuple[str, ...] = ()
    # Provenance, in two forms that are unioned rather than ranked (see tboxcraft.provenance).
    # source_cq_ids is the document-only original; source_refs is the source-neutral successor an
    # RDB adapter uses, because a term read out of a table row has no CQ to cite. Both are kept:
    # the document adapter can keep sending source_cq_ids unchanged.
    source_cq_ids: tuple[str, ...] = ()
    source_refs: tuple[str, ...] = ()

    @property
    def uid(self) -> str:
        """Kind-qualified identity. ``name`` alone collides across kinds (e.g. ``hasTask``
        exists as both an object and a data property), so all keying is by ``uid``."""
        return f"{self.kind}:{self.name}"

    @property
    def surface_lang(self) -> str:
        """Language tag for this component's ALTER/HIDDEN surfaces. ``und`` when it declared none.

        Not the representative label's tag — that one is :data:`PREF_LANG` by policy and is
        decided by :func:`~tboxcraft.represent.pref_lang_for`, not here. This is where a curated
        term the representative label could not use ends up: a Hangul term becomes an alternative
        surface, and the exposed views bucket alternatives by language, so getting this right is
        the difference between ``AlterTerms.ko`` and ``AlterTerms.en`` holding Korean text.

        One tag for all of a component's surfaces, which is the shape the request asks for
        ("the language the component carried", singular) and the only one both producers can
        fill. It is coarse where a component mixes scripts — an English ``name`` beside a Korean
        ``correct_term`` tags both the same — and going finer needs a tag per surface from the
        producers. The mislabelling it has to prevent is on the *representative* label, and that
        no longer reads this at all.
        """
        return self.lang or UNDETERMINED_LANG

    def declared_lang_for(self, surface: str) -> str:
        """The curated channel's declared language, refused where it cannot be true of ``surface``.
        See :meth:`refuse_impossible_lang`."""
        return self.refuse_impossible_lang(surface, self.surface_lang)

    def refuse_impossible_lang(self, surface: str, declared: str) -> str:
        """``declared``, unless it is ``en`` and provably wrong about ``surface``.

        A producer declaring ``en`` over a string with no Latin letter is stating something that
        cannot be true, and writing it down would make this pipeline the laundering step that put
        Korean labels in the store as ``@en`` — the same false claim, arriving by a different
        route. So that one declaration is refused and the row falls back to ``und``.

        **Only ``en``**, and only through the one-directional test
        :func:`~tboxcraft.represent.can_be_english`. Every other declaration is written as given:

        * ``ko`` over Latin text is romanised Korean, which is a real thing a producer may mean.
        * ``mul`` is ontoagent's tag for a genuinely mixed string (Hangul and Latin). It is a
          registered RFC 5646 code and carries *more* than ``und`` — "two languages are in here"
          is knowledge, not its absence — so folding it in would destroy information.
        * Checking any of these needs "which language is this", the guess refused everywhere else.

        Refusing is not the same as inventing: ``und`` withdraws a claim rather than making one.
        It is also never silent — :func:`~tboxcraft.io.components.scan_overridden_lang_declarations`
        counts every row this touches, because "the producer fixed it" and "we are still covering
        for the producer" are the same picture without a number.
        """
        if declared.split("-")[0].casefold() != PREF_LANG:
            return declared
        from app.tboxcraft.represent import can_be_english

        return declared if can_be_english(surface) else UNDETERMINED_LANG

    @property
    def comment_lang(self) -> str:
        """Language tag for ``dom_notes``. ``und`` until a producer sends one (ontoagent).

        Separate from :attr:`surface_lang` because the old hardcoding was wrong in the opposite
        direction: labels were asserted English and notes Korean, so an English corpus (``ipb``)
        got ``rdfs:comment ...@ko`` on rationale that is in English. Falling back to the
        label's language would just swap one guess for another — the two are independently
        unknown, so they are independently ``und``.
        """
        return self.notes_lang or UNDETERMINED_LANG

    @property
    def is_property(self) -> bool:
        return self.kind in PROPERTY_KINDS

    @property
    def is_class(self) -> bool:
        return self.kind == KIND_CLASS

    def range_values(self) -> list[str]:
        """Split a possibly comma-separated range into individual class/datatype names.

        Object-property ranges are sometimes ``"Road, Airspace, GeographicArea"`` (a
        multi-value string) which must become an ``owl:unionOf`` downstream.
        """
        if not self.range_raw:
            return []
        return [part.strip() for part in self.range_raw.split(",") if part.strip()]

    def domain_values(self) -> list[str]:
        """Split a possibly comma-separated domain into individual class names.

        The producer joins chunk-union domains with ``", "`` exactly like ranges — the two
        fields ride the same shape. Only ranges were split for a while, so a multi-value
        domain compared as one long string, matched nothing, and the property lost its
        ``rdfs:domain`` silently.
        """
        if not self.domain:
            return []
        return [part.strip() for part in self.domain.split(",") if part.strip()]

    def surface_candidates(self) -> list[tuple[str, str, str]]:
        """``(surface, lang, source)`` rows for every way this component can be written.

        The single collector for all four channels. It used to be open-coded in each consumer
        (the mapping table, the surface_form writer, the review-queue payload), and every one of
        them had drifted: the mapping table never collected ``duplicates`` at all. Adding a
        channel is one line here rather than three edits that must agree.

        The language is per row rather than per component, for one reason that is not a
        refinement: ``name`` is not in the same language as the rest. It is the IRI's local part,
        English by the convention the whole pipeline mints on, while ``correct_term`` and the
        synonyms are free text in whatever the producer writes. A component declaring ``ko``
        described the curated fields; tagging its English ``name`` ``ko`` too would file
        ``RelationType`` under ``AlterTerms.ko`` — the defect this work exists to remove, moved
        one row down.

        So ``name`` takes :data:`PREF_LANG` when it can be English, checked the same
        one-directional way :func:`~tboxcraft.represent.can_be_english` is used for the label, and
        falls back to the declared language when it cannot. Every other channel takes the declared
        language unexamined: ``Café`` is Latin and French, and deciding otherwise would be the
        language identification this pipeline refuses.

        Ordered by how much the surface is trusted to be the concept's own name, and deduped on
        first mention so a term repeated across channels keeps its strongest source.
        """
        return [
            (surface, written, source)
            for surface, _declared, written, source in self.surface_channels()
        ]

    def surface_channels(self) -> list[tuple[str, str, str, str]]:
        """``(surface, declared, written, source)`` — :meth:`surface_candidates` with its working.

        ``declared`` is the language the input claims for that row; ``written`` is what actually
        goes to ``surface_form`` after :meth:`refuse_impossible_lang`. They differ only where a
        producer said ``en`` about a string that cannot be English.

        Both are exposed because the refusal has to be countable
        (:func:`~tboxcraft.io.components.scan_overridden_lang_declarations`) and a counter that
        re-derives the rule is a second copy of it — the two would drift, and the drift would look
        like the producer having fixed something.

        The declared language differs by channel, which is not a refinement:

        ``name``            the IRI's local part, English by the convention the pipeline mints on.
                            A component declaring ``ko`` was describing its curated fields; tagging
                            ``RelationType`` ``ko`` would file it under ``AlterTerms.ko``, which is
                            this defect one row down. Falls back to the declared language when the
                            name cannot be English, and is never refused — it is derived here, not
                            claimed by anyone.
        ``correct_term_en`` English by the field's own name, and checked like any declaration.
        everything else     :attr:`surface_lang`, the curated channel's one tag.
        """
        from app.tboxcraft.represent import can_be_english

        declared = self.surface_lang
        name_lang = PREF_LANG if can_be_english(self.name) else declared
        rows = [(self.name, name_lang, name_lang, "name")]
        if self.correct_term_en:
            rows.append(
                (
                    self.correct_term_en,
                    PREF_LANG,
                    self.refuse_impossible_lang(self.correct_term_en, PREF_LANG),
                    "correct_term_en",
                )
            )
        if self.correct_term:
            rows.append(
                (
                    self.correct_term,
                    declared,
                    self.declared_lang_for(self.correct_term),
                    "correct_term",
                )
            )
        rows += [
            (d, declared, self.declared_lang_for(d), "duplicate")
            for d in self.dom_duplicates
        ]
        rows += [
            (s, declared, self.declared_lang_for(s), "synonym")
            for s in self.dom_synonyms
        ]
        seen: set[str] = set()
        out: list[tuple[str, str, str, str]] = []
        for surface, decl, written, source in rows:
            if surface and surface not in seen:
                seen.add(surface)
                out.append((surface, decl, written, source))
        return out

    def authoritative_hierarchy(self) -> Hierarchy:
        """``domain_layer.hierarchy`` — the only source of broader/narrower.

        This used to fall back to a linguistic layer when empty; that layer is retired, so an
        empty domain hierarchy now simply means the component declares none.
        """
        return self.dom_hierarchy

    @classmethod
    def from_dict(cls, raw: dict) -> Component:
        dom = raw.get("domain_layer") or {}
        return cls(
            name=raw["name"],
            kind=raw["kind"],
            domain=_clean(raw.get("domain")),
            range_raw=_clean(raw.get("range")),
            comment=_clean(raw.get("comment")),
            correct_term=_clean(dom.get("correct_term")),
            dom_duplicates=tuple(dom.get("duplicates", []) or []),
            dom_synonyms=tuple(dom.get("synonyms", []) or []),
            dom_hierarchy=Hierarchy.from_dict(dom.get("hierarchy")),
            agreement=_clean(dom.get("agreement")),
            dom_notes=_clean(dom.get("notes")),
            # The single read point for the language tags — both producers write the domain layer,
            # and a top-level fallback because the component envelope is the other place a producer
            # could reasonably put a property of the whole component. Unrecognised spelling means
            # None means ``und``: an unknown field name degrades to "we were not told", never to a
            # guess. Renaming these to match the producers' final field names is a change here and
            # nowhere else.
            #
            # ``correct_term_lang`` is what the producers actually send, and it is read first. The
            # names were guessed here before either producer had published a contract, and this one
            # guessed wrong: ontoagent tags per field, so every Korean curated term arrived and was
            # tagged ``und`` — the whole point of the change, silently not happening. Nothing
            # raised, because a missing optional field is indistinguishable from an absent one.
            #
            # The bare ``lang`` stays as a fallback: it is the "one tag for the whole curated
            # channel" shape this class is built around (see :attr:`surface_lang`), and
            # ``correct_term``'s language is that channel's best proxy — not a second meaning.
            correct_term_en=_clean(dom.get("correct_term_en")),
            lang=_clean(
                dom.get("correct_term_lang") or dom.get("lang") or raw.get("lang")
            ),
            notes_lang=_clean(dom.get("notes_lang") or raw.get("notes_lang")),
            verdict=raw.get("verdict", "tbox"),
            disagreement_flags=tuple(raw.get("disagreement_flags", []) or []),
            source_cq_ids=tuple(raw.get("source_cq_ids", []) or []),
            source_refs=tuple(raw.get("source_refs", []) or []),
        )


@dataclass(frozen=True)
class CompetencyQuestion:
    id: str
    question: str
    answer: str
