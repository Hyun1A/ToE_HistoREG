import torch
import pandas as pd
import random
import h5py
from torch import Tensor, einsum, nn
import numpy as np
import json
import re
from dataclasses import dataclass



def format_report(rec: dict, include_no_diag_note: bool = False) -> str:
    organ = rec.get("organ", "").strip()
    proc = rec.get("procedure", "").strip()
    items = rec.get("sub_feature", []) or []

    lines = [f"{organ}, {proc};"]
    n = len(items)

    def format_recursive(feature, d_line):
        d_line.append(feature["principal"])

        if len(feature["scores"]) > 0:
            if len(feature["sub_feature"]) > 0:
                d_line.append(str(feature["scores"])+",")
            else:
                d_line.append(str(feature["scores"]))

        if len(feature["sub_feature"]) > 0:
            for sub_feature in feature["sub_feature"]:
                d_line = format_recursive(sub_feature, d_line)

        return d_line


    for item in items:
        diag_line = []

        format_recursive(item, diag_line)
        # rev_diag_line = diag_line[::-1]        
        lines.append(" ".join(diag_line))


    if len(lines) == 1:
        report = lines[0]

    elif len(lines) == 2:
        report = "\n  ".join(lines)

    # multiple diagnoses
    elif len(lines) > 2:
        for idx, line in enumerate(lines[1:], start=1):
            lines[idx] = f"{idx}. {line}"

        report = "\n  ".join(lines)

    return report

_DEFAULT_PATTERN_BY_SCORE = {6:"3+3", 7:"3+4", 8:"4+4", 9:"4+5", 10:"5+5"}
_GS7_BY_GRADE_GROUP = {2:"3+4", 3:"4+3"}

def _insert_gleason_parenthetical(text: str) -> str:
    if re.search(r"Gleason'?s?\s*score\s*\d+\s*\(\d\+\d\)", text, flags=re.I):
        return text
    m_score = re.search(r"(Gleason'?s?\s*score\s*)(\d{1,2})", text, flags=re.I)
    if not m_score: return text
    score = int(m_score.group(2))
    m_group = re.search(r"grade\s*group\s*(\d+)", text, flags=re.I)
    gg = int(m_group.group(1)) if m_group else None
    pat = (_GS7_BY_GRADE_GROUP.get(gg) if score == 7 and gg in _GS7_BY_GRADE_GROUP
           else _DEFAULT_PATTERN_BY_SCORE.get(score))
    if not pat: return text
    return re.sub(r"(Gleason'?s?\s*score\s*)(\d{1,2})(?!\s*\()",
                  lambda m: f"{m.group(1)}{m.group(2)} ({pat})",
                  text, count=1, flags=re.I)

def _ensure_pattern4_placeholder_after_grade_group(text: str) -> str:
    # Only for GS=7
    m_score = re.search(r"Gleason'?s?\s*score\s*(\d{1,2})", text, flags=re.I)
    if not m_score or int(m_score.group(1)) != 7:
        return text
    # If already present right after grade group, do nothing
    if re.search(r"grade\s*group\s*\d+\s*\(Gleasn\s*pattern\s*4\s*:\s*\)", text, flags=re.I) \
       or re.search(r"grade\s*group\s*\d+\s*\(Gleason\s*pattern\s*4\s*:\s*\)", text, flags=re.I):
        return text
    # Insert placeholder before any immediate comma
    def repl(m):
        core, tail = m.group(1), (m.group(2) or "")
        return f"{core} (Gleason pattern 4: ){tail}"
    return re.sub(r"(grade\s*group\s*\d+)(\s*,?)", repl, text, count=1, flags=re.I)

def format_prostate_report(rec: dict, include_no_diag_note: bool = False) -> str:
    organ = rec.get("organ", "").strip()
    proc = rec.get("procedure", "").strip()
    items = rec.get("sub_feature", []) or []

    lines = [f"{organ}, {proc};"]

    def format_recursive(feature, d_line):
        d_line.append(feature["principal"])
        if feature.get("scores"):
            d_line.append((str(feature["scores"]) + ",") if feature.get("sub_feature") else str(feature["scores"]))
        for sub_feature in feature.get("sub_feature", []) or []:
            d_line = format_recursive(sub_feature, d_line)
        return d_line

    for item in items:
        diag_line = []
        format_recursive(item, diag_line)
        line = " ".join(diag_line)
        line = _insert_gleason_parenthetical(line)          # add (p1+p2)
        line = _ensure_pattern4_placeholder_after_grade_group(line)  # add (Gleasn pattern 4: )
        lines.append(line)

    if len(lines) == 1:
        return lines[0]
    elif len(lines) == 2:
        return "\n  ".join(lines)
    else:
        numbered = [f"{i}. {l}" for i, l in enumerate(lines[1:], start=1)]
        return "\n  ".join([lines[0]] + numbered)

def format_colon_report(rec: dict, include_no_diag_note: bool = False) -> str:
    organ = rec.get("organ", "").strip()
    proc = rec.get("procedure", "").strip()
    items = rec.get("sub_feature", []) or []


    """
    Things to keep in mind
    * when there is more than 3 principles, it is in form 1, 2. Note doesn't take place
    """

    lines = [f"{organ}, {proc};"]
    n = len(items)

    def format_recursive(feature, d_line):
        d_line.append(feature["principal"])

        if len(feature["scores"]) > 0:
            if len(feature["sub_feature"]) > 0:
                d_line.append(str(feature["scores"])+",")
            else:
                d_line.append(str(feature["scores"]))

        if len(feature["sub_feature"]) > 0:
            for sub_feature in feature["sub_feature"]:
                d_line = format_recursive(sub_feature, d_line)

        return d_line


    for item in items:
        diag_line = []
        format_recursive(item, diag_line)        
        lines.append(" ".join(diag_line))

    if len(lines) == 1:
        report = lines[0]

    elif len(lines) >= 2:
        report = "\n  ".join(lines)

    report = re.sub(r'(?<!\bwith\s)\b(low|high)\b', r'with \1', report, flags=re.IGNORECASE)
    return report

def format_bladder_report(rec: dict, include_no_diag_note: bool = False) -> str:
    organ = rec.get("organ", "").strip()
    proc = rec.get("procedure", "").strip()
    items = rec.get("sub_feature", []) or []


    """
    Things to keep in mind
    * when there is more than 3 principles, it is in form 1, 2. Note doesn't take place
    """

    lines = [f"{organ}, {proc};"]
    n = len(items)

    def format_recursive(feature, d_line):
        d_line.append(feature["principal"])

        if len(feature["scores"]) > 0:
            if len(feature["sub_feature"]) > 0:
                d_line.append(str(feature["scores"])+",")
            else:
                d_line.append(str(feature["scores"]))

        if len(feature["sub_feature"]) > 0:
            for sub_feature in feature["sub_feature"]:
                d_line = format_recursive(sub_feature, d_line)

        return d_line


    for item in items:
        diag_line = []
        format_recursive(item, diag_line)        
        lines.append(" ".join(diag_line))

    lines = [lines[0]] + lines[1:][::-1]

    if len(lines) == 1:
        report = lines[0]

    elif len(lines) >= 2 and len(lines) <= 3:
        report = "\n  ".join(lines)

    # multiple diagnoses
    elif len(lines) >= 4:
        for idx, line in enumerate(lines[1:], start=1):
            if "Note" in line:
                lines[idx] = f"{line}"
                break
            lines[idx] = f"{idx}. {line}"

        report = "\n  ".join(lines)
    # breakpoint()

    return report



def format_breast_report(rec: dict, include_no_diag_note: bool = False) -> str:
    organ = rec.get("organ", "").strip()
    proc = rec.get("procedure", "").strip()
    items = rec.get("sub_feature", []) or []

    def is_ic_nst(name: str) -> bool:
        return name == "Invasive carcinoma of no special type"

    def is_dcis(name: str) -> bool:
        return name == "Ductal carcinoma in situ"

    def ic_nst_scores_block(sf: dict):
        wanted = {"Tubule formation", "Nuclear grade", "Mitoses"}
        scores_map = {}
        for ch in sf.get("sub_feature", []):
            nm = ch.get("principal", "")
            if nm in wanted:
                scores_map[nm] = (ch.get("scores", "") or "").strip()
        if len(scores_map) != 3:
            return None, scores_map

        def to_int(s):
            try: return int(s)
            except: return None

        vals = [to_int(scores_map.get(k, "")) for k in ("Tubule formation", "Nuclear grade", "Mitoses")]
        if any(v is None for v in vals):
            return None, scores_map
        ssum = sum(vals)
        if 3 <= ssum <= 5: grade = "grade I"
        elif 6 <= ssum <= 7: grade = "grade II"
        elif 8 <= ssum <= 9: grade = "grade III"
        else: grade = None
        return grade, scores_map

    def dcis_detail_lines(sf: dict) -> list[str]:
        value_map = {"Type": None, "Nuclear grade": None, "Necrosis": None}
        for ch in sf.get("sub_feature", []):
            fname = ch.get("principal", "")
            if fname in value_map:
                sub = ch.get("sub_feature", [])
                if sub:
                    value_map[fname] = (sub[0].get("principal", "") or "").strip()
        lines = []
        for fname in ("Type", "Nuclear grade", "Necrosis"):
            if value_map[fname]:
                lines.append(f"  - {fname}: {value_map[fname]}")
        return lines

    lines = [f"{organ}, {proc};"]
    n = len(items)

    if n == 0:
        if include_no_diag_note:
            lines.append("  (no specific diagnosis reported)")
        return "\n".join(lines)

    if n == 1:
        sf = items[0]
        name = sf.get("principal", "")
        if is_ic_nst(name):
            grade, smap = ic_nst_scores_block(sf)
            if grade:
                tf = smap.get("Tubule formation", "")
                ng = smap.get("Nuclear grade", "")
                mt = smap.get("Mitoses", "")
                lines.append(f"  {name}, {grade} (Tubule formation: {tf}, Nuclear grade: {ng}, Mitoses: {mt})")
            else:
                lines.append(f"  {name}")
        elif is_dcis(name):
            lines.append(f"  {name}")
            lines.extend(dcis_detail_lines(sf))
        else:
            lines.append(f"  {name}")
        return "\n".join(lines)

    # multiple diagnoses
    for idx, sf in enumerate(items, start=1):
        name = sf.get("principal", "")
        if is_ic_nst(name):
            grade, smap = ic_nst_scores_block(sf)
            if grade:
                tf = smap.get("Tubule formation", "")
                ng = smap.get("Nuclear grade", "")
                mt = smap.get("Mitoses", "")
                lines.append(f"  {idx}. {name}, {grade} (Tubule formation: {tf}, Nuclear grade: {ng}, Mitoses: {mt})")
            else:
                lines.append(f"  {idx}. {name}")
        else:
            lines.append(f"  {idx}. {name}")
    return "\n".join(lines)


def _reformat_cervix_hsil_cin(text: str) -> str:
    """
    Convert '(HSIL) CIN 3' or '(LSIL) CIN 1' → '(HSIL; CIN 3)' / '(LSIL; CIN 1)'.
    Digits only (1–3). No Roman numerals.
    Safely no-ops if already '(HSIL; CIN N)'.
    """
    t = text.replace("（", "(").replace("）", ")")  # normalize full-width parens

    # already in desired form? do nothing
    if re.search(r"\((HSIL|LSIL)\s*;\s*CIN\s*[1-3]\)", t, flags=re.I):
        return t

    # '(HSIL) CIN 3' / '(LSIL)CIN1' / '(HSIL)  CIN-2'
    return re.sub(
        r"\(\s*(HSIL|LSIL)\s*\)\s*CIN\s*-?\s*([1-3])\b",
        r"(\1; CIN \2)",
        t,
        flags=re.I,
    )


def format_cervix_report(rec: dict, include_no_diag_note: bool = False) -> str:
    """
    Uterine cervix report formatter.
    - Header: "Uterine cervix, <procedure>;"
    - Each diagnosis line is built recursively from principal/scores/sub_features (like your other formatters)
    - HSIL/LSIL + CIN is normalized to "(HSIL; CIN N)" / "(LSIL; CIN N)"
    - If multiple diagnoses, number them 1., 2., ...
    """
    organ = (rec.get("organ") or "").strip()
    proc = (rec.get("procedure") or "").strip()
    items = rec.get("sub_feature", []) or []

    lines = [f"{organ}, {proc};"]

    def format_recursive(feature, d_line):
        # principal
        d_line.append(feature.get("principal", ""))

        # scores (if present)
        scores = feature.get("scores")
        subf = feature.get("sub_feature", []) or []
        if scores:
            d_line.append((str(scores) + ",") if subf else str(scores))

        # children
        for ch in subf:
            d_line = format_recursive(ch, d_line)

        return d_line

    for item in items:
        diag_chunks = []
        format_recursive(item, diag_chunks)
        line = " ".join(diag_chunks).strip()

        # Cervix-specific compaction of HSIL/LSIL + CIN
        line = _reformat_cervix_hsil_cin(line)
        lines.append(line)

    # Output rules (match your style)
    if len(lines) == 1:
        # no diagnoses captured
        if include_no_diag_note:
            return "\n  ".join([lines[0], "(no specific diagnosis reported)"])
        return lines[0]

    if len(lines) == 2:
        return "\n  ".join(lines)

    # multiple diagnoses → number them
    numbered = [f"{i}. {l}" for i, l in enumerate(lines[1:], start=1)]
    return "\n  ".join([lines[0]] + numbered)



# format_report

FORMAT_REGISTRY={'breast': format_breast_report,
                    'uterine cervix': format_cervix_report, 
                    'colon': format_colon_report, 
                    'rectum': format_colon_report, 
                    'lung': format_report, 
                    'stomach': format_report,
                    'prostate': format_prostate_report, 
                    'urinary bladder': format_bladder_report,
                 }

