"""Replayable codebook concepts, bound to exact dataset members in an archive.

Descriptions license object-level correspondence only, never value equivalence,
unit conversion or paper visibility. Metadata provenance is an explicit input.
"""
from __future__ import annotations

import hashlib
import re
import zipfile
from .common import contained, file_digest, seal, seal_errors
from .dataset_catalog import fields
from .fidelity import correspondence

VERSION = "archive-bound-name-declarations/v2"


def derive(spec, catalog, root):
    if spec["catalog_sha256"] != catalog["catalog_sha256"]:
        raise ValueError("mapping_catalog_mismatch")
    path = contained(root, spec["archive_path"])
    if file_digest(path) != spec["archive_file_bytes_sha256"]:
        raise ValueError("mapping_archive_changed")
    if not spec.get("source_url", "").startswith("https://"):
        raise ValueError("metadata_origin_required")
    entries, scopes = [], set()
    objects = fields(catalog)
    with zipfile.ZipFile(path) as archive:
        raw = archive.read(spec["codebook_member"])
        text = raw.decode("utf-8-sig")  # retain original offsets and line endings
        declarations = []
        for match in re.finditer(r"(?m)^[ \t]*[-+]\s*([A-Za-z][A-Za-z0-9_]*)\s*:\s*([^\r\n]+)", text):
            name, description = match.group(1), match.group(2).strip()
            # Only noun-like first clauses. Sentences, conditions and lists are
            # retained in metadata but cannot silently become name aliases.
            phrase = re.split(r"[.(]", description, maxsplit=1)[0].strip()
            if not re.fullmatch(r"[A-Za-z]+(?:[ -][A-Za-z]+){0,7}", phrase):
                continue
            if re.search(r"\b(?:if|is|are|or|and|not|including|extracted)\b", phrase, re.I):
                continue
            variants = [(phrase, "description_head")]
            concept = re.sub(r"^Normalized\s+", "", phrase, flags=re.I)
            concept = re.sub(r"\s+in\s+(?:Celsius|Fahrenheit|Kelvin)$", "", concept, flags=re.I)
            if concept != phrase:
                variants.append((concept, "strip_representation_qualifier_object_identity_only"))
            declarations.append((name, variants, match.start(), match.end(), match.group(0)))
        for binding in spec["scope_members"]:
            scope = binding["scope_id"]
            if scope in scopes:
                raise ValueError("duplicate_mapping_scope")
            scopes.add(scope)
            table = next(t for t in catalog["tables"] if t["scope_id"] == scope)
            member_hash = hashlib.sha256(archive.read(binding["dataset_member"])).hexdigest()
            if member_hash not in {s["sha256"] for s in table["dataset_sources"] if s["role"] == "dataset"}:
                raise ValueError("codebook_dataset_member_identity_mismatch")
            for name, variants, start, end, quote in declarations:
                for obj in objects:
                    if obj["scope_id"] != scope or obj["name"] != name:
                        continue
                    for alias, rule in variants:
                        entries.append({"object_id": obj["object_id"], "scope_id": scope, "name": name, "alias": alias,
                            "basis": "bound_metadata_description_concept", "transformation": rule,
                            "archive_file_bytes_sha256": spec["archive_file_bytes_sha256"],
                            "dataset_member": binding["dataset_member"], "dataset_file_bytes_sha256": member_hash,
                            "source_url": spec["source_url"], "codebook_member": spec["codebook_member"],
                            "codebook_file_bytes_sha256": hashlib.sha256(raw).hexdigest(),
                            "char_start": start, "char_end": end, "quote": quote,
                            "value_equivalence": "not_asserted"})
    return seal({"schema_version": VERSION, "catalog_sha256": catalog["catalog_sha256"], "spec": spec,
                 "entries": entries, "authority": "explicit_metadata_concepts_not_paper_gold"}, "mapping_sha256")


def verify(document, catalog, root):
    if seal_errors(document, "mapping_sha256") or document != derive(document["spec"], catalog, root):
        raise ValueError("mapping_derivation_mismatch")
    return document["entries"]


def match(label, catalog, entries=(), scope_id=None):
    result = correspondence(label, catalog, aliases=entries, scope_id=scope_id)
    if result["reason"] == "bound_metadata_alias_declaration":
        result["reason"] = "bound_metadata_description_concept"
    if result["candidates"]:
        ids = {c["object_id"] for c in result["candidates"]}
        from .fidelity import lexical
        result["mapping_evidence"] = [a for a in entries if a["object_id"] in ids and lexical(a["alias"]) == lexical(label)]
    return result
