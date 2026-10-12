# Metrics

Every number in the paper is a metric defined here, computed by
`evaluation score` from a run's exported files and aggregated by
`evaluation aggregate`. The code is the definition: each section names the
module under `evaluation/scoring/` that computes it.

## Gold standard and firewall

The gold standard of every manual is the manual itself: its glossary (terms
and abbreviations), its index and its body paragraphs. The pipeline never
sees them. Before upload, `evaluation prep` cuts each PDF down to its body pages,
so the glossary, index and table of contents reach only the scorer. Their
extracted forms are tracked under `corpus/<manual>/`.

The glossary and index of IPB were checked by hand; the other manuals' were
extracted by the same code, which reproduces the hand-checked one.

## String matching

Every string comparison goes through one normalization
(`normalize.py`), strict on purpose so that every coverage figure is a lower
bound:

1. Unicode NFKC, private-use glyphs removed, typographic quotes and dashes
   made plain.
2. A leading proponent marker `*` removed.
3. CamelCase split, for labels that come from IRI local names.
4. Lower case, punctuation to spaces, spaces collapsed.
5. A leading article (a, an, the) removed.

There is no stemming, plural folding or synonym table.

A term of the ontology has surface forms: its preferred label (`Pref`) and
its alternative and hidden labels. Metrics with a `.Pref` / `.AllForms`
suffix are computed twice, against preferred labels only and against all
surface forms.

## Experiment 1: stability (`stability.py`)

IPB runs three times at temperature 0. Every pair of repetitions yields one
row per metric.

| Metric | Definition |
|---|---|
| `Stab.<Set>.Jaccard` | Jaccard index of the two runs' sets, elements being normalized preferred labels. Sets: `Class`, `ObjProp`, `DataProp`, `SubClass` (pairs of labels), `Domain` and `Range` (property and class label pairs), `CQ` (normalized questions) |
| `Stab.<Set>.Match.Pref`, `.AllForms` | Share of both runs' elements that have a counterpart in the other run, where two terms correspond when they share a surface form. Added after the main run was seen |

Elements are labels, not IRIs: each run writes under its own namespace,
`http://example.org/doctrine/<Title>_<run stamp>_r<i>#`, so IRIs never agree
and repetitions never merge in the store.

## Experiments 1 and 2: coverage (`coverage.py`)

Per manual and run. A manual without a glossary or an index has no row for
that metric, never a zero.

| Metric | Definition |
|---|---|
| `Cov.Term.Recall.Pref`, `.AllForms` | Share of glossary terms whose normalized form equals a surface form of the ontology |
| `Cov.Abbr.Recall.Pref`, `.AllForms` | Share of glossary abbreviations matched by the abbreviation or its expansion |
| `Cov.Index.Recall.Pref`, `.AllForms` | Share of index headings matched by any of their variants |
| `Cov.Para.CQ` | Share of body paragraphs with at least one competency question |
| `Cov.Para.Onto` | Share of body paragraphs with at least one competency question that contributed a term to the ontology |

## Experiments 1 and 2: accounting (`accounting.py`)

What each run kept, dropped and produced. The identities check that every
item a step dropped was reported; each is 1 when it holds.

| Metric | Identity or definition |
|---|---|
| `Acct.Identity.Paragraphs` | paragraphs = paragraphs with drafts + paragraphs whose generation failed |
| `Acct.Identity.Drafts` | drafts = drafts refined + drafts whose refinement failed |
| `Acct.Identity.Promoted` | promoted CQs = terminologized + terminologization failures |
| `Acct.Identity.Components` | components extracted = TBox candidates + terms withheld + names folded |
| `Acct.Identity.Candidates` | TBox candidates = new + merged + relabelled + gray |
| `Acct.Identity.Individuals` | individuals sent = upserted + deduplicated + dropped + skipped |
| `Acct.Identity.Assertions` | assertions built = inserted + deduplicated + dropped |
| `Acct.NoSilentLoss` | 1 when all seven identities hold |
| `Acct.Generate.FailedParagraphs`, `.FailedText` | Share of paragraphs, and of body characters, whose CQ generation failed |
| `Acct.CQ.PerParagraph` | CQs per body paragraph |
| `Acct.CQ.Flagged` | Share of refined CQs carrying a verification flag |
| `Acct.CQ.KeptOriginal` | Share of drafts the refiner kept unchanged |
| `Acct.CQ.NoTerms`, `.TermsPerCQ` | Share of CQs with no term, and terms per CQ, after terminologization |
| `Acct.TBox.Withheld` | Share of extracted components withheld from the TBox |
| `Acct.TBox.Unresolved` | Count of property domains and ranges left unresolved |
| `Acct.ABox.Skipped` | Share of individuals the ABox load skipped |
| `Acct.Size.*` | Paragraphs, body characters, CQs, classes, object and data properties, subclass edges, individuals |
| `Acct.Hier.WithParent` | Share of classes with at least one parent class |
| `Acct.Hier.MaxDepth` | Longest chain of subclass edges below a root class (a root has depth 0) |
| `Acct.Hier.MultiParent` | Count of classes with more than one parent |
| `Acct.LLM.<server>.Truncated` | Requests the LLM server ended at its output limit during the run, from its request counters |
| `Acct.Latency.<step>`, `.Cell` | Wall-clock seconds of each pipeline step and of the whole run |

## Experiment 3: lookup invariants (`router.py`)

The lookup service is probed against the ontologies of every manual. It reads
the whole store, so each answer is judged only against the namespaces of the
probed runs.

| Metric | Invariant |
|---|---|
| `Router.NS.Purity` | Per manual: share of graph classes and terms that sit under the manual's namespace |
| `Router.Overlap.Recall` | Share of labels shared by several manuals that are answered with every owner's IRI |
| `Router.Glossary.Recall` | Per manual: share of glossary entries answered with an IRI of that manual |
| `Router.Glossary.Agreement` | How often that answer agrees with the coverage matcher over all surface forms |
| `Router.Typo.Recall` | Per manual: share of one-letter typos of an exactly found term whose first suggestion is that term's IRI |
| `Router.Typo.TieRecall` | The same, counting the term's IRI anywhere among suggestions tied for first. Added after the probe was seen, because the service orders a tie by graph name |
| `Router.OOD.Leak` | Share of a frozen list of out-of-domain queries answered inside the probed namespaces |
| `Router.Prov.Chain` | Per manual: share of terms that reach a body paragraph through a CQ |

The probe runs twice, with 5 and with 50 suggestions per query; the second
is reported separately.

## Experiment 4: faithfulness (`faithfulness.py`)

Added after the first three experiments were reported. The CQ answers of one
IPB run are judged with RAGAS Faithfulness by an external model (Claude
Sonnet 5.5); every statement, verdict and reason is stored and scored like
any other run file. The judge accepts no temperature, so the stored verdicts
are the record.

| Metric | Definition |
|---|---|
| `Faith.CQ.Mean` | Mean over CQs of the share of the answer's statements supported by the source paragraph |
| `Faith.CQ.Full`, `.Low` | Share of CQs scoring 1, and below 0.5 |
| `Faith.CQ.Mean.Ctx100Plus`, `.CtxShort` | The mean split at a paragraph length of 100 characters; shorter paragraphs are mostly titles and captions |
| `Faith.Statement.Mean` | Share of all statements supported |
| `Faith.Judged`, `Faith.Size.*` | CQs scored, statements, judge errors |

Shares are over the CQs that received a score.

## Aggregation (`evaluation/results.py`)

Rows are grouped by experiment, run type, manual and metric. Each group
reports:

| Field | Meaning |
|---|---|
| `n`, `n_failed`, `n_undefined` | Runs in the group, runs that failed, runs that succeeded without a value |
| `mean_with_zero` | Mean with failed runs counted as 0 |
| `mean_ok`, `std_ok` | Mean and standard deviation over successful runs |
| `ci_low`, `ci_high` | Percentile bootstrap 95% interval of the mean over successful runs (1000 resamples, seed 0) |

A failed run is never left out: it either counts as 0 or is visible in
`n_failed`.
