import argparse
import re
from pathlib import Path
from typing import Dict, List, Set, Tuple

from common import dump_jsonl


REL_PATTERNS = [
    (r"(.+?)可分为(.+?)、(.+?)[。；\n]", "可分为"),
    (r"(.+?)包括(.+?)、(.+?)[。；\n]", "包括"),
    (r"(.+?)采用(.+?)[。；\n]", "采用"),
    (r"(.+?)用于(.+?)[。；\n]", "用于"),
    (r"(.+?)影响(.+?)[。；\n]", "影响"),
    (r"(.+?)提升(.+?)[。；\n]", "提升"),
    (r"(.+?)降低(.+?)[。；\n]", "降低"),
    (r"(.+?)具有(.+?)[。；\n]", "具有"),
    (r"(.+?)是(.+?)[。；\n]", "是"),
    (r"(.+?)属于(.+?)[。；\n]", "属于"),
    (r"(.+?)由(.+?)组成[。；\n]", "由...组成"),
    (r"(.+?)由(.+?)构成[。；\n]", "由...构成"),
]


def clean_text(x: str) -> str:
    return re.sub(r"\s+", " ", x).strip(" ，,。；;:：()（）[]【】")


def split_candidates(obj: str) -> List[str]:
    chunks = re.split(r"[、，,和及与/]", obj)
    return [clean_text(c) for c in chunks if clean_text(c)]


def extract_by_rules(text: str, source_doc: str) -> List[Dict]:
    triples: List[Dict] = []
    seen: Set[Tuple[str, str, str]] = set()

    for pat, rel in REL_PATTERNS:
        for m in re.finditer(pat, text):
            head = clean_text(m.group(1))
            tail_raw = clean_text(m.group(2))
            if len(m.groups()) >= 3:
                tail_raw = f"{tail_raw}、{clean_text(m.group(3))}"
            if not head or not tail_raw:
                continue
            for tail in split_candidates(tail_raw):
                if len(head) < 2 or len(tail) < 2:
                    continue
                key = (head, rel, tail)
                if key in seen:
                    continue
                seen.add(key)
                triples.append(
                    {
                        "head": head,
                        "relation": rel,
                        "tail": tail,
                        "source": source_doc,
                        "method": "rule_based",
                    }
                )
    return triples


def extract_terms(sentence: str) -> List[str]:
    cands = re.findall(r"[\u4e00-\u9fa5A-Za-z0-9]{2,20}", sentence)
    stop = {"我们", "研究", "结果", "方法", "进行", "以及", "可以", "通过", "本文", "作者"}
    terms = []
    for c in cands:
        x = clean_text(c)
        if len(x) < 2 or x in stop:
            continue
        if x.isdigit():
            continue
        terms.append(x)
    # 保序去重
    uniq = []
    seen = set()
    for t in terms:
        if t in seen:
            continue
        seen.add(t)
        uniq.append(t)
    return uniq[:8]


def extract_cooccurrence(text: str, source_doc: str) -> List[Dict]:
    triples: List[Dict] = []
    seen: Set[Tuple[str, str, str]] = set()
    sentences = re.split(r"[。；\n]", text)
    for s in sentences:
        s = s.strip()
        if len(s) < 10:
            continue
        terms = extract_terms(s)
        if len(terms) < 2:
            continue
        head = terms[0]
        for tail in terms[1:]:
            if head == tail:
                continue
            key = (head, "相关于", tail)
            if key in seen:
                continue
            seen.add(key)
            triples.append(
                {
                    "head": head,
                    "relation": "相关于",
                    "tail": tail,
                    "source": source_doc,
                    "method": "cooccurrence",
                }
            )
    return triples


def run_extract(parsed_dir: str, out_jsonl: str) -> int:
    parsed_path = Path(parsed_dir)
    if not parsed_path.exists():
        raise FileNotFoundError(f"解析目录不存在: {parsed_dir}")

    all_triples: List[Dict] = []
    for md_file in sorted(parsed_path.glob("*.md")):
        content = md_file.read_text(encoding="utf-8", errors="ignore")
        doc_triples = extract_by_rules(content, md_file.name)
        doc_triples.extend(extract_cooccurrence(content, md_file.name))
        all_triples.extend(doc_triples)
        print(f"[EXTRACT] {md_file.name}: {len(doc_triples)} triples")

    dump_jsonl(out_jsonl, all_triples)
    print(f"[DONE] raw triples = {len(all_triples)}, saved to {out_jsonl}")
    return len(all_triples)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--parsed-dir", required=True, help="Markdown 文档目录")
    parser.add_argument("--out-jsonl", required=True, help="原始三元组输出 jsonl")
    args = parser.parse_args()
    run_extract(args.parsed_dir, args.out_jsonl)


if __name__ == "__main__":
    main()
