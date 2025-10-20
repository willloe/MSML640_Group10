from __future__ import annotations

import json
from pathlib import Path
from typing import Tuple, List, Optional

from jsonschema import Draft202012Validator


def _load_json(path: str | Path) -> dict:
    p = Path(path)
    with p.open("r", encoding="utf-8-sig") as f:
        return json.load(f)


def _validator(schema_path: str | Path) -> Draft202012Validator:
    schema = _load_json(schema_path)
    return Draft202012Validator(schema)


def validate_with_schema(instance: dict, schema_path: str | Path) -> Tuple[bool, List[str]]:
    v = _validator(schema_path)
    errs = sorted(v.iter_errors(instance), key=lambda e: e.path)
    if not errs:
        return True, []
    msgs = []
    for e in errs:
        loc = ".".join([str(x) for x in e.path]) or "<root>"
        msgs.append(f"{loc}: {e.message}")
    return False, msgs


def _find_repo_root(start: Optional[Path] = None) -> Path:
    here = start or Path(__file__).resolve()
    for p in [here] + list(here.parents):
        schemas_dir = p / "schemas"
        if (schemas_dir / "layout.schema.json").exists() and (schemas_dir / "palette.schema.json").exists():
            return p
    return Path(__file__).resolve().parents[3]


def validate_layout(layout: dict, root: str | Path = None) -> Tuple[bool, List[str]]:
    base = Path(root) if root else _find_repo_root()
    schema_path = base / "schemas" / "layout.schema.json"
    return validate_with_schema(layout, schema_path)


def validate_palette(palette: dict, root: str | Path = None) -> Tuple[bool, List[str]]:
    base = Path(root) if root else _find_repo_root()
    schema_path = base / "schemas" / "palette.schema.json"
    return validate_with_schema(palette, schema_path)
