import re

UPPER_CAMEL = re.compile(r"^[A-Z][A-Za-z0-9]*$")
LOWER_CAMEL = re.compile(r"^[a-z][A-Za-z0-9]*$")
XSD_DATATYPE = re.compile(r"^xsd:[A-Za-z][A-Za-z0-9]*$")


def ensure_upper_camel(value: str) -> str:
    if not UPPER_CAMEL.match(value):
        raise ValueError(f"MUST be UpperCamelCase, got: {value!r}")
    return value


def ensure_lower_camel(value: str) -> str:
    if not LOWER_CAMEL.match(value):
        raise ValueError(f"MUST be lowerCamelCase, got: {value!r}")
    return value


def ensure_xsd_datatype(value: str) -> str:
    if not XSD_DATATYPE.match(value):
        raise ValueError(f"MUST be an xsd:* datatype, got: {value!r}")
    return value
