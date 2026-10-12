"""conceptualizer module: CQList → annotated TBox components + ABox individuals.

No TTL is generated. router.py owns the endpoint (POST /extract), schemas.py the
boundary schemas, and service/ the extraction, classification and reintegration
logic. The master ontology is not read; later steps (ABox attachment, type
validation) are the responsibility of the external ontology system.
"""
