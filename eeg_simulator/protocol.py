"""Small human-readable protocol used by mock and future serial devices."""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any


@dataclass(frozen=True)
class ProtocolMessage:
    command: str
    parameters: dict[str, str] = field(default_factory=dict)


def encode_message(command: str, **parameters: Any) -> str:
    name = command.strip().upper()
    if not name or any(character.isspace() for character in name):
        raise ValueError("Protocol command must be one non-empty token")
    fields = [name]
    for key, value in parameters.items():
        protocol_key = key.strip().upper()
        if not protocol_key or any(character.isspace() for character in protocol_key):
            raise ValueError(f"Invalid protocol parameter name: {key!r}")
        text_value = str(value).replace(" ", "_")
        fields.append(f"{protocol_key}={text_value}")
    return " ".join(fields)


def decode_message(text: str) -> ProtocolMessage:
    tokens = text.strip().split()
    if not tokens:
        raise ValueError("Cannot decode an empty protocol message")
    parameters: dict[str, str] = {}
    for token in tokens[1:]:
        if "=" not in token:
            parameters[token.upper()] = ""
            continue
        key, value = token.split("=", 1)
        parameters[key.upper()] = value.replace("_", " ")
    return ProtocolMessage(tokens[0].upper(), parameters)
