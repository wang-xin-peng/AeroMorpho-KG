import json
import os
from pathlib import Path
from typing import Dict, Iterable, List


def ensure_parent(path: str) -> None:
    Path(path).parent.mkdir(parents=True, exist_ok=True)


def list_pdf_files(pdf_dir: str) -> List[Path]:
    return sorted(Path(pdf_dir).glob("*.pdf"))


def load_jsonl(path: str) -> List[Dict]:
    items: List[Dict] = []
    if not Path(path).exists():
        return items
    with open(path, "r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            items.append(json.loads(line))
    return items


def dump_jsonl(path: str, rows: Iterable[Dict]) -> None:
    ensure_parent(path)
    with open(path, "w", encoding="utf-8") as f:
        for row in rows:
            f.write(json.dumps(row, ensure_ascii=False) + "\n")


def safe_env(name: str, default: str = "") -> str:
    value = os.getenv(name, default)
    return value.strip() if isinstance(value, str) else default
