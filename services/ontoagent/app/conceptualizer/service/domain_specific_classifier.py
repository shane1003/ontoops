"""DomainSpecificClassifier: domain layer (additive) + TBox/ABox verdict.

Takes the components extracted from a CQ batch and attaches to each a domain layer
(correct_term, spelling variants, duplicate and hierarchy observations) and a verdict
(tbox|abox). The master ontology is not read; only the components passed in are seen.

The JSON returned by the LLM (fast tier) is matched by name and parsed into
`DomainLayer` + verdict; on malformed output it degrades to empty annotations (no
crash).

Instances hold no state between calls: the runner fans chunks out concurrently, so
`run()` on one instance must be safe when it overlaps across threads. The batch label
for logs is likewise a `run(label=...)` argument, not an instance attribute (as an
attribute, chunks would overwrite each other's labels).
"""

from __future__ import annotations

import json
import logging
import threading
from dataclasses import dataclass, field

from app.core.config import Settings
from app.core.llm.client import LLMClient
from app.conceptualizer.schemas import (
    UNDETERMINED_LANG,
    AnnotatedComponent,
    DisagreementFlag,
    DomainLayer,
    VerdictCandidate,
    normalize_lang,
    script_lang,
)
from app.core.llm.util import complete_json, load_prompt as _load_prompt
from app.conceptualizer.service.merge_util import split_slot

log = logging.getLogger("conceptualizer.classifier")

_PROMPT_FILE = "classifier.txt"


# Cap on the reverse-reference lists put in the prompt (keeps the prompt small even
# for large chunks).
_MAX_REFS = 5


def structural_refs(
    components: list[AnnotatedComponent],
) -> dict[str, dict[str, list[str]]]:
    """Builds the reverse-reference map name → {"domain_of": [props...], "range_of": [props...]}.

    A component only holds its own domain/range, so "who references me as
    domain/range" exists nowhere. Without it the classifier never even receives the
    fact that some class is a property's domain, and may judge something that must
    structurally be a class as abox.
    """
    refs: dict[str, dict[str, list[str]]] = {}

    def add(name: str, slot: str, prop: str) -> None:
        entry = refs.setdefault(name, {"domain_of": [], "range_of": []})
        if prop not in entry[slot]:
            entry[slot].append(prop)

    for c in components:
        if c.kind == "class":
            continue
        for n in split_slot(c.domain):
            add(n, "domain_of", c.name)
        for n in split_slot(c.range):
            add(n, "range_of", c.name)
    return refs


def _components_as_input(components: list[AnnotatedComponent]) -> str:
    """Serializes component structure + reverse references as LLM input (for the classifier).

    The reverse references (domain_of/range_of) give "this name is the domain/range of
    some property" as evidence for the verdict: in OWL a property's domain/range must
    be a class, so this is a strong TBox signal.
    """
    refs = structural_refs(components)
    payload = []
    for c in components:
        item = {
            "name": c.name,
            "kind": c.kind,
            "domain": c.domain,
            "range": c.range,
            "comment": c.comment,
        }
        # Add reverse references only where they exist (no empty keys bloating the prompt).
        ref = refs.get(c.name)
        if ref:
            for slot in ("domain_of", "range_of"):
                names = ref[slot]
                if not names:
                    continue
                item[slot] = names[:_MAX_REFS]
                if len(names) > _MAX_REFS:
                    item[f"{slot}_total"] = len(names)
        payload.append(item)
    return json.dumps(payload, ensure_ascii=False, indent=2)


@dataclass
class ClassifierStats:
    """Classifier observations filled jointly by the chunks (thread-safe).

    The same numbers also appear in the per-chunk logs, but with chunks running
    concurrently the lines interleave and are hard to count. When the caller creates
    one and passes it to every chunk, the totals can go into the response.

    Values are **per-chunk sums**: a name appearing in several chunks is added that
    many times.
    """

    n_unmatched: int = 0
    n_promoted_conflict: int = 0
    n_degraded_batches: int = 0
    # Chunks add concurrently, so lock the += (read-modify-write).
    _lock: threading.Lock = field(
        default_factory=threading.Lock, repr=False, compare=False
    )

    def add(
        self,
        *,
        unmatched: int = 0,
        promoted_conflict: int = 0,
        degraded_batches: int = 0,
    ) -> None:
        with self._lock:
            self.n_unmatched += unmatched
            self.n_promoted_conflict += promoted_conflict
            self.n_degraded_batches += degraded_batches


class DomainSpecificClassifier:
    """Domain expert classifier that attaches the domain layer + TBox/ABox verdict."""

    def __init__(
        self,
        settings: Settings,
        domain: str | None = None,
        *,
        language: str | None = None,
    ) -> None:
        self.settings = settings
        # The domain is the CQ document's ontology.domain (injected by the runner). If
        # the document does not carry it or nothing is injected, fall back to the
        # setting (agents.domain).
        self._domain = domain or settings.agents.domain
        # Fallback language of the domain-layer text (the request's `?language=`). Used
        # only when the LLM gave no language; without this too, it is und.
        self._language = normalize_lang(language)
        if language and self._language is None:
            log.warning(
                "[classifier] could not read language=%r as a language tag → ignored "
                "(domain-layer language fallback is %s).",
                language,
                UNDETERMINED_LANG,
            )
        # Defer creating the SDK client until the call, so keyless (test, mock)
        # environments still work.
        self._client: LLMClient | None = None
        # In the chunk fan-out several threads make the first call at once, so lock the
        # lazy creation (unlocked, one SDK client per chunk is created and the
        # connection pool is split).
        self._client_lock = threading.Lock()

    @property
    def client(self) -> LLMClient:
        with self._client_lock:
            if self._client is None:
                self._client = LLMClient(self.settings)
            return self._client

    def run(
        self,
        components: list[AnnotatedComponent],
        *,
        label: str = "",
        stats: ClassifierStats | None = None,
    ) -> list[AnnotatedComponent]:
        """Fills domain_layer + verdict into each of the components and returns them.

        On LLM failure/malformed output it degrades to an empty domain layer + default
        verdict (no crash).

        Args:
            components: Components of this batch (chunk). Annotations are filled in
                place.
            label: Batch label printed in the logs (e.g. "chunk 3/10"). Chunks run
                concurrently and their logs interleave, so the caller passes it to let
                each line name its chunk.
            stats: If given, unmatched and conflict-promotion counts accumulate here
                (observations to put in the response).

        Returns:
            The same components, annotated.
        """
        if not components:
            return components

        tag = label or "-"
        log.info(
            "[classifier] %s: starting domain layer/verdict review of %d components (domain=%s)",
            tag,
            len(components),
            self._domain,
        )
        by_key, by_name, degraded = self._annotate(components, tag)
        # Match by (kind, name) first: a class and a property with the same name can
        # coexist, so matching by name alone attaches one's annotation to the other
        # kind. Fall back to name-only only when the LLM did not echo kind. On no match
        # keep the empty domain layer/default verdict.
        matched = 0
        for comp in components:
            parsed = by_key.get((comp.kind, comp.name)) or by_name.get(comp.name)
            if parsed is None:
                continue
            comp.domain_layer, comp.verdict = parsed
            matched += 1
        unmatched = len(components) - matched
        if stats is not None:
            stats.add(unmatched=unmatched, degraded_batches=1 if degraded else 0)
        # The attachment result line is printed on **every** path. It used to be
        # skipped when matched==0, leaving only the WARNING, so to an observer counting
        # these lines the failed batch did not exist at all: the log sum and the
        # response's n_unmatched disagreed (one line short of the chunk count, with the
        # missing chunk accounting for all of n_unmatched).
        log.info(
            "[classifier] %s: domain layer attached %d/%d (unmatched %d)",
            tag,
            matched,
            len(components),
            unmatched,
        )
        if matched == 0:
            # Every component is left with an empty domain layer + default verdict (a
            # silent quality collapse). At INFO, the API (uvicorn) path leaves no trace
            # in the container because root is WARNING. The two causes are logged
            # separately because the remedies differ: degrade means the call/parse
            # broke (large chunks truncate the response); zero matches means a
            # response came but no name matched (prompt/output format).
            log.warning(
                "[classifier] %s: all %d components have an empty domain layer, default verdict — "
                "%s",
                tag,
                len(components),
                (
                    "LLM call/parse failed (degrade). Check whether the response was "
                    "truncated because the chunk is large."
                    if degraded
                    else "LLM response matched no component. Check the prompt/model "
                    "output format."
                ),
            )
        promoted = self._flag_structural_contradictions(components, tag)
        if stats is not None:
            stats.add(promoted_conflict=promoted)
        return components

    def _flag_structural_contradictions(
        self, components: list[AnnotatedComponent], tag: str
    ) -> int:
        """Promotes the abox verdict of components used as a property's domain/range to conflict.

        In OWL a property's domain/range must be a class, and extract_individuals also
        relies on that structure to build individuals (class_name = that name). An abox
        verdict for such a component therefore contradicts the structure the pipeline
        already assumes; it is a defect rather than a judgment.

        Passed through silently, it would surface only later as a downstream individual
        resolution failure (miss), so it is promoted to conflict here with the
        competing candidates (classifier abox vs structure tbox) kept as evidence. The
        value is not settled as tbox because the disposition belongs to the TBox
        owner: the distinction travels as the verdict, the evidence as
        disagreement_flags.

        Returns:
            Number of components promoted to conflict.
        """
        refs = structural_refs(components)
        flagged: list[str] = []
        for c in components:
            if c.verdict != "abox":
                continue
            ref = refs.get(c.name)
            if not ref:
                continue
            users = ref["domain_of"] + ref["range_of"]
            if not users:
                continue
            c.verdict = "conflict"
            c.disagreement_flags.append(
                DisagreementFlag(
                    kind="verdict_conflict",
                    component=c.name,
                    candidates=[
                        VerdictCandidate(value="abox", source="classifier"),
                        VerdictCandidate(value="tbox", source="structure"),
                    ],
                    note=(
                        f"'{c.name}' 은(는) 속성 "
                        f"{', '.join(users[:_MAX_REFS])} 의 domain/range 로 쓰여 구조상 "
                        "클래스여야 하는데 분류기가 abox 로 판정했다."
                    ),
                )
            )
            flagged.append(c.name)

        if flagged:
            # WARNING so this does not become a silent defect: it leaves a trace on the
            # API (uvicorn) path too.
            log.warning(
                "[classifier] %s: %d promoted to conflict by structural contradiction — "
                "property domain/range judged abox: %s",
                tag,
                len(flagged),
                ", ".join(flagged[:_MAX_REFS]),
            )
        return len(flagged)

    def _seal_language(self, layer: DomainLayer) -> None:
        """Settles the language of each domain-layer text: a value always has a language.

        Priority: **Hangul script (fixed) > tag given by the LLM > request
        `?language=` > `und`**.

        Why Hangul wins by string inspection: Hangul is a script used only for Korean,
        so its presence is an observation, not a guess, and cannot be more wrong than
        the model's tag. Even if the model tags a Hangul term as `en` (exactly the direction
        of the defect where labels were always `@en`), the string says otherwise, so
        the string wins. The reverse direction (Latin script → English) does not hold
        and is not judged: romanized Korean and Spanish are all Latin.

        Mixed text is `mul`. Folding it to `und` would mean "unknown" and discard the
        known fact that it is mixed; folding it to one side would make half of it
        false.

        The language of a field without a value is cleared: absent text has no
        language to claim, and if left, downstream tries to tag an empty literal.
        """
        for text_field, lang_field in (
            ("correct_term", "correct_term_lang"),
            ("notes", "notes_lang"),
        ):
            text = getattr(layer, text_field)
            if not text:
                setattr(layer, lang_field, None)
                continue
            setattr(
                layer,
                lang_field,
                script_lang(text)
                or getattr(layer, lang_field)
                or self._language
                or UNDETERMINED_LANG,
            )

    def _annotate(
        self, components: list[AnnotatedComponent], tag: str = "-"
    ) -> tuple[
        dict[tuple[str, str], tuple[DomainLayer, str]],
        dict[str, tuple[DomainLayer, str]],
        bool,
    ]:
        """Calls the LLM to build (kind,name)→(DomainLayer, verdict) and name-only maps.

        If the LLM echoes kind, entries go under the (kind,name) key, otherwise into
        the name-only fallback.

        Returns:
            The (kind,name) map, the name-only fallback map, and whether it
            **degraded**: whether the call/parse broke and nothing was received. Empty
            maps alone cannot be told apart from "a response came but no name
            matched", and the two have different remedies (chunk size vs
            prompt/output format).
        """
        domain = self._domain
        user = "다음 컴포넌트(이번 CQ 배치)를 검토하라.\n\n" + _components_as_input(
            components
        )
        try:
            # Loading the prompt is inside the try too: a missing/unreadable file must
            # also be an empty-annotation degrade + warning, not a crash (the documented
            # "degrade on failure" contract).
            system = _load_prompt(self.settings, _PROMPT_FILE, domain=domain)
            data = complete_json(
                self.client, system, user, label="classifier", fast=True
            )
        except Exception as exc:  # noqa: BLE001 — prompt/LLM/parse errors all degrade
            log.warning(
                "[classifier] %s: LLM/prompt handling failed → degrade to empty domain layer: %s",
                tag,
                exc,
            )
            return {}, {}, True

        # If the response is not a dict (e.g. a top-level JSON array), degrade to empty
        # annotations instead of crashing.
        if not isinstance(data, dict):
            log.warning(
                "[classifier] %s: LLM response not a JSON object → degrade to empty domain layer",
                tag,
            )
            return {}, {}, True

        by_key: dict[tuple[str, str], tuple[DomainLayer, str]] = {}
        by_name: dict[str, tuple[DomainLayer, str]] = {}
        for item in data.get("components", []) or []:
            if not isinstance(item, dict):
                continue
            name = item.get("name")
            if not name:
                continue
            kind = item.get("kind")
            # verdict/kind are not domain-layer fields, so parse them separately.
            verdict = item.get("verdict", "tbox")
            if verdict not in ("tbox", "abox", "conflict"):
                verdict = "tbox"
            layer_fields = {
                k: v for k, v in item.items() if k not in ("name", "kind", "verdict")
            }
            try:
                layer = DomainLayer.model_validate(layer_fields)
                self._seal_language(layer)
            except Exception as exc:  # noqa: BLE001 — skip individual malformed items
                log.warning(
                    "DomainSpecificClassifier: annotation of component '%s' failed to parse, "
                    "skipped: %s",
                    name,
                    exc,
                )
                continue
            if kind in ("class", "object_property", "data_property"):
                by_key[(kind, name)] = (layer, verdict)
            else:
                by_name[name] = (layer, verdict)
        return by_key, by_name, False
