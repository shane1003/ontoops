"""ABox loader for the tboxcraft ontology store.

Reads individuals (ABox) from ``component.json``, resolves their class and
data-property tokens against ``surface_form`` joined to ``tbox_term``, and writes rows into
``abox_individual`` and ``abox_data_assertion``. The loader is a *consumer* of the TBox:
representative-term selection and dedup live in the TBox pipeline, not here.
"""
