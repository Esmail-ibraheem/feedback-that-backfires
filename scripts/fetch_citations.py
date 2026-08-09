"""Verify every candidate reference against the arXiv API and emit BibTeX.

The rule for this project is that a citation goes into the manuscript only if a
machine-readable record for it was retrieved here. This script takes a list of
arXiv ids, fetches the canonical metadata, and writes both a JSON audit trail
(`paper/references_audit.json`) and a `.bib` file. Ids that fail to resolve are
reported and *not* written, so a typo becomes a visible gap rather than a
plausible-looking fabrication.
"""

from __future__ import annotations

import json
import os
import re
import sys
import time
import urllib.parse
import urllib.request
import xml.etree.ElementTree as ET
from typing import Dict, List, Optional

ATOM = "{http://www.w3.org/2005/Atom}"
API = "http://export.arxiv.org/api/query"

# arXiv id -> the citation key we will use in the manuscript.
CANDIDATES: Dict[str, str] = {
    # Agent loops, feedback, self-correction
    "2210.03629": "yao2023react",
    "2303.11366": "shinn2023reflexion",
    "2303.17651": "madaan2023selfrefine",
    "2310.01798": "huang2024selfcorrect",
    "2304.05128": "chen2024selfdebug",
    "2310.08118": "valmeekam2023critiquing",
    "2309.10691": "wang2024mint",
    "2607.01641": "hou2026infiniteloops",
    "2605.20072": "zenkri2026fidelity",
    "2605.30785": "adacom2026context",
    "2605.08563": "yang2026contamination",
    "2508.11027": "wang2025hellorhighwater",
    # Repetition, degeneration, copying
    "2206.02369": "xu2022breaktheloop",
    "1904.09751": "holtzman2020degeneration",
    "1908.04319": "welleck2020unlikelihood",
    "1909.05858": "keskar2019ctrl",
    "2310.04625": "mcdougall2024copysuppression",
    "2507.07810": "repetitionneurons2025",
    "2209.11895": "olsson2022induction",
    # Negation / "pink elephant"
    "2402.07896": "castricato2024pinkelephants",
    "2404.15154": "nopinkelephant2024",
    "2503.22395": "negation2025pinkelephant",
    # Structured output / constrained decoding / tools
    "2307.09702": "willard2023outlines",
    "2408.02442": "tam2024speakfreely",
    "2305.15334": "patil2024gorilla",
    "2606.25605": "constrainttax2026",
    "2510.07248": "toolschemas2025",
    # Small models as agents; efficiency
    "2506.02153": "belcak2025slmagents",
    "2502.02737": "allal2025smollm2",
    "2412.15115": "qwen2024qwen25",
    "2505.09388": "qwen2025qwen3",
    "2407.21783": "grattafiori2024llama3",
    # Benchmarks and context
    "2108.07732": "austin2021mbpp",
    "2310.06770": "jimenez2024swebench",
    "2308.03688": "liu2024agentbench",
    "2406.12045": "yao2024taubench",
    "2307.13854": "zhou2024webarena",
    "2311.12983": "mialon2024gaia",
    "2407.18901": "trivedi2024appworld",
    "2307.03172": "liu2024lostmiddle",
    # Test-time compute
    "2408.03314": "snell2024testtime",
    "2411.17501": "stroebl2024inferenceflaws",
    "2407.21787": "brown2024monkeys",
}


def fetch(arxiv_ids: List[str], chunk: int = 8) -> Dict[str, dict]:
    out: Dict[str, dict] = {}
    for i in range(0, len(arxiv_ids), chunk):
        batch = arxiv_ids[i : i + chunk]
        url = f"{API}?{urllib.parse.urlencode({'id_list': ','.join(batch), 'max_results': len(batch)})}"
        for attempt in range(4):
            try:
                with urllib.request.urlopen(url, timeout=90) as resp:
                    xml = resp.read()
                break
            except Exception as exc:  # noqa: BLE001
                print(f"  retry {attempt+1} for {batch}: {exc}", file=sys.stderr)
                time.sleep(5 * (attempt + 1))
        else:
            continue
        root = ET.fromstring(xml)
        for entry in root.findall(f"{ATOM}entry"):
            eid = entry.findtext(f"{ATOM}id") or ""
            m = re.search(r"abs/([^v]+)v?(\d*)", eid)
            if not m:
                continue
            key = m.group(1)
            title = " ".join((entry.findtext(f"{ATOM}title") or "").split())
            if not title or title.lower().startswith("error"):
                continue
            authors = [
                " ".join((a.findtext(f"{ATOM}name") or "").split())
                for a in entry.findall(f"{ATOM}author")
            ]
            published = entry.findtext(f"{ATOM}published") or ""
            out[key] = {
                "arxiv_id": key,
                "title": title,
                "authors": authors,
                "year": published[:4],
                "published": published,
                "url": f"https://arxiv.org/abs/{key}",
                "primary_category": (
                    entry.find(f"{ATOM}primary_category").get("term")
                    if entry.find(f"{ATOM}primary_category") is not None
                    else ""
                ),
                "doi": entry.findtext("{http://arxiv.org/schemas/atom}doi") or "",
                "comment": " ".join(
                    (entry.findtext("{http://arxiv.org/schemas/atom}comment") or "").split()
                ),
            }
        time.sleep(3)  # be polite to the API
    return out


def to_bibtex(key: str, rec: dict) -> str:
    authors = " and ".join(rec["authors"]) if rec["authors"] else "Unknown"
    title = rec["title"].replace("{", "").replace("}", "")
    lines = [
        f"@article{{{key},",
        f"  title = {{{title}}},",
        f"  author = {{{authors}}},",
        f"  journal = {{arXiv preprint arXiv:{rec['arxiv_id']}}},",
        f"  year = {{{rec['year']}}},",
        f"  eprint = {{{rec['arxiv_id']}}},",
        "  archivePrefix = {arXiv},",
        f"  url = {{{rec['url']}}}",
        "}",
    ]
    return "\n".join(lines)


def main() -> None:
    # Windows consoles default to a legacy code page; author names are not ASCII.
    try:
        sys.stdout.reconfigure(encoding="utf-8", errors="replace")
    except Exception:  # noqa: BLE001
        pass
    ids = list(CANDIDATES)
    print(f"verifying {len(ids)} arXiv ids...")
    recs = fetch(ids)
    missing = [i for i in ids if i not in recs]
    print(f"resolved {len(recs)}; unresolved: {missing}")

    os.makedirs("paper", exist_ok=True)
    audit = {
        "resolved": {CANDIDATES[k]: v for k, v in recs.items()},
        "unresolved": [{"arxiv_id": i, "intended_key": CANDIDATES[i]} for i in missing],
        "fetched_at": time.strftime("%Y-%m-%dT%H:%M:%S"),
    }
    with open("paper/references_audit.json", "w", encoding="utf-8") as fh:
        json.dump(audit, fh, indent=2, ensure_ascii=False)

    with open("paper/references.bib", "w", encoding="utf-8") as fh:
        fh.write(
            "% Auto-generated by scripts/fetch_citations.py.\n"
            "% Every entry below was retrieved from the arXiv API; see\n"
            "% paper/references_audit.json for the raw records and fetch time.\n"
            "% Do not hand-edit: re-run the script instead.\n\n"
        )
        for aid in ids:
            if aid in recs:
                fh.write(to_bibtex(CANDIDATES[aid], recs[aid]) + "\n\n")

    for aid in ids:
        if aid in recs:
            r = recs[aid]
            first = r["authors"][0].split()[-1] if r["authors"] else "?"
            print(f"  OK  {aid:12s} {CANDIDATES[aid]:28s} {first} {r['year']}  {r['title'][:70]}")
    for aid in missing:
        print(f"  ??  {aid:12s} {CANDIDATES[aid]:28s} UNRESOLVED")


if __name__ == "__main__":
    main()
