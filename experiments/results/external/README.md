# External evaluation: RAGAS faithfulness

`ragas_faithfulness.csv` was produced by a separate evaluation, not by this
evaluation code, and is reported in the paper as supplementary evidence in a section
of its own. It is never mixed into `results.jsonl` or `aggregates.jsonl`.

| | |
|---|---|
| Made by | another team of the project, 2026-09-29 |
| Judge | Claude Sonnet 5.5 (`claude-sonnet-5-5`), RAGAS 0.4.3 |
| Data | the production store's load of each manual, not the runs of this repository |
| Scope | CQs from body paragraphs: IPB all 2,509. The rows kept here are those the paper cites |

Rows:

- `A:paragraph->cq`: RAGAS Faithfulness as published. Context is the paragraph,
  response is the CQ's answer; the answer is split into statements and each is
  judged against the paragraph.
- `B:cq->term`: a custom adaptation, not RAGAS Faithfulness proper. Context is
  the CQ's question and answer, and one fixed statement per linked term, "The
  CQ refers to the concept '<preferred label>'.", is judged. The wording fits
  classes better than properties, so the property rows are underestimates.
- `B:cq->term (statement)`: the same verdicts averaged per statement, by term
  type.

Differences from the runs of this repository: the production load received the
whole PDF and non-body paragraphs were filtered afterwards; the namespaces, the
purpose and scope statements and the CQ counts differ (IPB 2,509 body CQs there,
2,194 to 2,200 here).

Stage A by paragraph length, from the same evaluation's report (mean, n):

| Manual | Paragraph of 100 characters or more | Shorter (table and figure captions, titles) |
|---|---|---|
| IPB | 0.975 (2,468) | 0.368 (41) |
