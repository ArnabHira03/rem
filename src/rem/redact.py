"""Mask secrets in conversation text before rem stores it or sends it to an LLM."""

import re

_PATTERNS: list[tuple[str, re.Pattern[str]]] = [
    ("private_key", re.compile(
        r"-----BEGIN [A-Z ]*PRIVATE KEY-----.*?-----END [A-Z ]*PRIVATE KEY-----", re.DOTALL)),
    ("anthropic_key", re.compile(r"sk-ant-[A-Za-z0-9_\-]{20,}")),
    ("openai_key", re.compile(r"sk-(?:proj-)?[A-Za-z0-9_\-]{20,}")),
    ("github_token", re.compile(r"gh[pousr]_[A-Za-z0-9]{30,}")),
    ("aws_access_key", re.compile(r"AKIA[0-9A-Z]{16}")),
    ("jwt", re.compile(r"eyJ[A-Za-z0-9_\-]+\.[A-Za-z0-9_\-]+\.[A-Za-z0-9_\-]+")),
]
# NAME=value lines whose name looks secret: keep the name, mask the value.
_ASSIGNMENT = re.compile(
    r"(?im)^([ \t]*(?:export[ \t]+)?[A-Z0-9_]*(?:KEY|TOKEN|SECRET|PASSWORD|PASSWD)[A-Z0-9_]*[ \t]*[=:][ \t]*)\S+"
)


def redact(text: str) -> str:
    for kind, pattern in _PATTERNS:
        text = pattern.sub(f"[REDACTED:{kind}]", text)
    return _ASSIGNMENT.sub(r"\1[REDACTED:secret_assignment]", text)
