# ontorouter

A lexical routing service. Give it a term; it answers which FalkorDB graph holds
that term and at which IRI.

It reads the ontology that ontoagent (the sibling service in this repository)
produces and never writes to it. Graph traversal is the caller's job — this
service returns pointers, not neighbourhoods.

## The three tiers

`/lookup` answers from up to three tiers, and the response says which of them
actually ran.

| Tier | Source | Goes into | Fresh? |
| :--- | :--- | :--- | :--- |
| **exact** | Live Postgres, every request | `results` | Always |
| **lexical** | In-memory fuzzy index over every normalized key | `suggestions.lexical` | Reloaded on an interval |
| **semantic** | pgvector in the `ontorouter` schema | `suggestions.semantic` | Off unless `EMBED_URL` is set |

Only the exact tier produces `results`, and an approximate candidate can never
enter it. The semantic tier is **disabled by default**: in-memory fuzzy
matching replaced it for typo correction because it recovers injected typos
without admitting out-of-domain queries, where the embedding tier could not
separate the two. It is kept for the jobs fuzzy matching cannot do — a
multilingual query matching an English ontology, for one.

## Response entries

Every entry — in `results` and in both suggestion channels — carries the same
pointer fields, plus `score` on suggestions only.

| Field | Meaning |
| :--- | :--- |
| `graph_id` | The FalkorDB graph holding the term, derived from the IRI base. |
| `iri` | The term's full IRI. This is also `tbox_term.id` upstream. |
| `term_type` | `CLASS`, `OBJECT_PROP` or `DATA_PROP`. |
| `label_form` | Which surface form matched: `PREF`, `ALTER` or `HIDDEN`. |
| `pref_label` | The term's *preferred* name — usually a different string from the one that matched, since many stored surface forms are `ALTER` or `HIDDEN`. Null only when upstream holds no preferred form for the term. |
| `score` | Suggestion channels only. `[0, 1]`, higher is better. The two channels use different scales and are never merged. |

`pref_label` exists because the IRI is not a substitute for the name: some IRI
local parts differ from the preferred label outright (`#AO` for
"Area of Operations", `#C2Asset` for "Command and Control Asset"), and
recovering the rest would need a CamelCase split that mangles acronyms.

## Environment

Every setting is read from the environment — nothing is hardcoded.

| Key | Default | Notes |
| :--- | :--- | :--- |
| `POSTGRES_HOST` | `localhost` | Host of the live Postgres instance. |
| `POSTGRES_PORT` | `5432` | Port of the live Postgres instance. |
| `POSTGRES_DBNAME` | `postgres` | Database holding the exact/lexical data and the `ontorouter` schema. |
| `POSTGRES_USER` | `postgres` | Postgres user. |
| `POSTGRES_PASSWORD` | *(empty)* | Postgres password. |
| `EMBED_URL` | *(unset)* | TEI embedding endpoint. Required to enable the semantic tier. |
| `EMBEDDING_MODEL_ID` | *(unset)* | Embedding model id. Required to enable the semantic tier. |
| `EMBED_TIMEOUT_S` | `5.0` | Timeout for calls to the embedding endpoint. |
| `EMBED_BACKFILL_INTERVAL_S` | `300` | Interval between periodic semantic-index backfill passes. |
| `PG_TIMEOUT_S` | `2.0` | Statement timeout for the exact/lexical tier queries. |
| `LEXICAL_MIN_SCORE` | `0.75` | Minimum score for a lexical suggestion. Chosen so injected typos still match while out-of-domain queries do not. |
| `SEMANTIC_MIN_SCORE` | `0.5` | Minimum score for a semantic suggestion. |
| `EXACT_VARIANT_MAX_WORDS` | `6` | Word-count cap on the exact tier's separator-variant expansion. |
| `LEXICAL_CANDIDATE_KEYS` | `20` | How many keys the lexical tier takes from the in-memory fuzzy index before fetching their rows. |
| `KEY_INDEX_REFRESH_INTERVAL_S` | `300` | How often the fuzzy key index reloads. Suggestions are that stale; `results` are always live. |

The semantic tier is inactive whenever `EMBED_URL` or `EMBEDDING_MODEL_ID` is
unset — this is a normal, degraded-but-working state, not an error.

## Running the dev server

```bash
uv sync
uv run uvicorn app.main:app --reload
```

## API parameters

Both `/lookup` endpoints take the same two query parameters.

| Parameter | Type | Default | Meaning |
| :--- | :--- | :--- | :--- |
| `k` | integer, 1-50 | `5` | Cap on entries returned *per suggestion channel*. It does not bound `results`: every confirmed pointer is always returned. |
| `tiers` | `exact` \| `lexical` \| `semantic` | `semantic` | How deep to run. `exact` returns pointers only and skips both suggestion channels; `lexical` adds the lexical channel; `semantic` adds both. Any other value is rejected with 422. |

`GET /lookup` takes the term as `?term=`; `POST /lookup` takes a JSON body
`{"terms": [...]}` of 1-100 terms and answers each with the single-term logic,
in submission order.

### Suggestions never repeat a result

Both suggestion channels run over the same normalized key the exact tier
matched, so a confirmed pointer would otherwise come back at the top of each
channel with a perfect score. Every `(graph_id, iri)` already present in
`results` is excluded from both channels before ranking, so `k` buys the
consumer alternatives rather than a copy of the answer it already has. An empty
`lexical` list next to a full `results` list is the normal, correct shape for a
term with no near-misses.

### Degradation and status codes

| Situation | Behaviour |
| :--- | :--- |
| Exact tier exceeds `PG_TIMEOUT_S` | **503**. There is no answer to give, so nothing is served. |
| Lexical or semantic tier times out, or the embedding endpoint is unreachable, slow, or returns a bad response | **200**. That channel is returned empty and its name is left out of `meta.tiers_executed`; `results` and the surviving channel are unaffected. The failure is logged at WARNING. |
| `EMBED_URL` or `EMBEDDING_MODEL_ID` unset | **200**, permanently, with `semantic` absent from `meta.tiers_executed`. A normal degraded state, not an error. |
| Blank term, term over 256 characters, `k` outside 1-50, unknown `tiers`, batch outside 1-100 | **422**. |

`meta.tiers_executed` is the contract for "did this channel actually run": an
empty channel that *is* listed there ran and found nothing, and an empty
channel that is *not* listed was dropped.

## API examples

The two examples below are illustrative: the graph ids name manuals from this
repository's corpus, and the JSON has the exact shape the service returns, but
the entries are not a capture of a particular store. They show the service's
central property: a term legitimately routes to *multiple* graphs, and
`results` returns all of them, not just one.

### `GET /lookup`

```
GET /lookup?term=Avenue%20of%20Approach
```

The lexical suggestions are near neighbours the caller does *not* already hold
— confirmed results are excluded from both channels, so `k` buys alternatives
rather than echoing the answer back.

```json
{
  "term": "Avenue of Approach",
  "normalized_key": "avenue of approach",
  "results": [
    {
      "graph_id": "military/Intelligence_Preparation_of_the_Battlefield",
      "iri": "http://example.org/military/Intelligence_Preparation_of_the_Battlefield#AvenueOfApproach",
      "term_type": "CLASS",
      "label_form": "PREF",
      "pref_label": "Avenue of Approach"
    },
    {
      "graph_id": "military/Offense_and_Defense",
      "iri": "http://example.org/military/Offense_and_Defense#AvenueOfApproach",
      "term_type": "CLASS",
      "label_form": "PREF",
      "pref_label": "Avenue of Approach"
    },
    {
      "graph_id": "military/Joint_Intelligence_Preparation_of_the_Operational_Environment",
      "iri": "http://example.org/military/Joint_Intelligence_Preparation_of_the_Operational_Environment#AvenueOfApproach",
      "term_type": "CLASS",
      "label_form": "PREF",
      "pref_label": "Avenue of Approach"
    }
  ],
  "suggestions": {
    "lexical": [
      {
        "graph_id": "military/Intelligence_Preparation_of_the_Battlefield",
        "iri": "http://example.org/military/Intelligence_Preparation_of_the_Battlefield#AirAvenueOfApproach",
        "term_type": "CLASS",
        "label_form": "PREF",
        "pref_label": "Air Avenue of Approach",
        "score": 0.9
      },
      {
        "graph_id": "military/Joint_Intelligence_Preparation_of_the_Operational_Environment",
        "iri": "http://example.org/military/Joint_Intelligence_Preparation_of_the_Operational_Environment#AirAvenueOfApproach",
        "term_type": "CLASS",
        "label_form": "PREF",
        "pref_label": "Air Avenue of Approach",
        "score": 0.9
      }
    ],
    "semantic": []
  },
  "meta": {
    "tiers_executed": [
      "exact",
      "lexical"
    ]
  }
}
```

The batch example below leads with a misspelling. Candidate generation scans
an in-memory index of every normalized key with rapidfuzz, so a typo anywhere
in the term still finds its target — the prefix prefetch this replaced could
only match strings the query was a prefix of.

### `POST /lookup`

```
POST /lookup
Content-Type: application/json

{"terms": ["Fire Suport", "Avenue of Approach"]}
```

```json
{
  "items": [
    {
      "term": "Fire Suport",
      "normalized_key": "fire suport",
      "results": [],
      "suggestions": {
        "lexical": [
          {
            "graph_id": "military/Offense_and_Defense",
            "iri": "http://example.org/military/Offense_and_Defense#FireSupport",
            "term_type": "CLASS",
            "label_form": "PREF",
            "pref_label": "Fire Support",
            "score": 0.9565217391304348
          },
          {
            "graph_id": "military/Joint_Intelligence_Preparation_of_the_Operational_Environment",
            "iri": "http://example.org/military/Joint_Intelligence_Preparation_of_the_Operational_Environment#FireSupport",
            "term_type": "CLASS",
            "label_form": "PREF",
            "pref_label": "Fire Support",
            "score": 0.9565217391304348
          }
        ],
        "semantic": []
      },
      "meta": {
        "tiers_executed": [
          "exact",
          "lexical"
        ]
      }
    },
    {
      "term": "Avenue of Approach",
      "normalized_key": "avenue of approach",
      "results": [
        {
          "graph_id": "military/Intelligence_Preparation_of_the_Battlefield",
          "iri": "http://example.org/military/Intelligence_Preparation_of_the_Battlefield#AvenueOfApproach",
          "term_type": "CLASS",
          "label_form": "PREF",
          "pref_label": "Avenue of Approach"
        },
        {
          "graph_id": "military/Offense_and_Defense",
          "iri": "http://example.org/military/Offense_and_Defense#AvenueOfApproach",
          "term_type": "CLASS",
          "label_form": "PREF",
          "pref_label": "Avenue of Approach"
        },
        {
          "graph_id": "military/Joint_Intelligence_Preparation_of_the_Operational_Environment",
          "iri": "http://example.org/military/Joint_Intelligence_Preparation_of_the_Operational_Environment#AvenueOfApproach",
          "term_type": "CLASS",
          "label_form": "PREF",
          "pref_label": "Avenue of Approach"
        }
      ],
      "suggestions": {
        "lexical": [
          {
            "graph_id": "military/Intelligence_Preparation_of_the_Battlefield",
            "iri": "http://example.org/military/Intelligence_Preparation_of_the_Battlefield#AirAvenueOfApproach",
            "term_type": "CLASS",
            "label_form": "PREF",
            "pref_label": "Air Avenue of Approach",
            "score": 0.9
          },
          {
            "graph_id": "military/Joint_Intelligence_Preparation_of_the_Operational_Environment",
            "iri": "http://example.org/military/Joint_Intelligence_Preparation_of_the_Operational_Environment#AirAvenueOfApproach",
            "term_type": "CLASS",
            "label_form": "PREF",
            "pref_label": "Air Avenue of Approach",
            "score": 0.9
          }
        ],
        "semantic": []
      },
      "meta": {
        "tiers_executed": [
          "exact",
          "lexical"
        ]
      }
    }
  ]
}
```

## `POST /admin/embed-backfill`

> **Internal only. Do not expose this router publicly.** It carries no
> authentication in v1 and is meant to run on a trusted network behind whatever
> the deployment already uses to reach the service's private surface.

Triggers one incremental embedding backfill pass and returns `202 Accepted`
immediately; the pass runs in the background. Returns `503` when the semantic
tier is not configured.

```bash
curl -X POST http://127.0.0.1:8000/admin/embed-backfill
```

The same pass also runs on its own every `EMBED_BACKFILL_INTERVAL_S` seconds
while the semantic tier is configured, so this endpoint is for forcing one
early — after a bulk upstream load, say — not for routine operation. A pass
embeds every active surface form that has no current vector in the configured
model space, in committed batches, under a Postgres advisory lock so a manual
trigger and the periodic pass cannot overlap. "Current" includes a check that
the stored `md5` of the surface text still matches: if upstream recreates its
tables and reassigns `surface_form.id`, the affected vectors are re-embedded
rather than left silently attached to the wrong terms. Each pass logs its row
count at INFO, and a failed pass logs the exception.

## Applying the DDL

`db/ddl/001_ontorouter_schema.sql` creates the `ontorouter` schema that the
semantic tier writes to. **The service applies it itself at start-up**, so a
normal deployment needs no manual step. Every statement is `IF NOT EXISTS`, so
applying it repeatedly does nothing.

That requires the service's database role to hold `CREATE` on the database. If
it does not, start-up logs the failure and carries on: the exact and lexical
tiers never read the `ontorouter` schema, and a semantic lookup drops its
channel and still returns 200. Apply the file by hand in that case.

`PGPASSWORD` rather than a password in the URL: a connection string on the
command line lands in shell history and is visible in `ps` to every user on
the box.

```bash
PGPASSWORD="$POSTGRES_PASSWORD" psql \
  -h "$POSTGRES_HOST" -p "$POSTGRES_PORT" \
  -U "$POSTGRES_USER" -d "$POSTGRES_DBNAME" \
  -f db/ddl/001_ontorouter_schema.sql
```

## Running in Docker

The image reads its listening port from `PORT` at container start, so one
image can serve any port:

```bash
docker build -t ontorouter:dev .
docker run --rm -e PORT=9001 \
  -e POSTGRES_HOST -e POSTGRES_PORT -e POSTGRES_DBNAME \
  -e POSTGRES_USER -e POSTGRES_PASSWORD \
  -p 9001:9001 ontorouter:dev
```
