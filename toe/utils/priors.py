# utils/priors.py
import json
import torch

def load_hier_priors(path: str) -> dict:
    with open(path, "r", encoding="utf-8") as f:
        return json.load(f)

def align_priors_to_schema(schema, priors: dict):
    """
    priors(JSON)의 분포를 schema의 vocabulary 순서에 맞춰 torch.Tensor로 정렬해 반환.
    반환 dict:
      - organ: Tensor[C_org]
      - procedure: { organ: Tensor[C_proc(org)] }
      - top: { organ: Tensor[C_top(org)] }
      - sub: { head_key: Tensor[C_labels(head)] }
    빈 vocab은 빈 텐서로 반환.
    """
    organs = list(getattr(schema, "organs_vocab"))

    # organ
    organ_vec = []
    pmap_org = priors.get("organ", {})
    Corg = max(1, len(organs))
    for org in organs:
        organ_vec.append(pmap_org.get(org, 1.0 / Corg))
    out_organ = torch.tensor(organ_vec, dtype=torch.float32)

    # procedure (organ별)
    out_proc = {}
    proc_vocab = getattr(schema, "procedures_vocab")
    pmap_all_proc = priors.get("procedure", {})
    for org in organs:
        labels = proc_vocab.get(org, [])
        if len(labels) == 0:
            out_proc[org] = torch.empty(0, dtype=torch.float32)
            continue
        m = pmap_all_proc.get(org, {})
        vec = [m.get(lbl, 1.0 / len(labels)) for lbl in labels]
        out_proc[org] = torch.tensor(vec, dtype=torch.float32)

    # top (organ별)
    out_top = {}
    top_vocab = getattr(schema, "top_diagnoses")
    pmap_all_top = priors.get("top", {})
    for org in organs:
        labels = top_vocab.get(org, [])
        if len(labels) == 0:
            out_top[org] = torch.empty(0, dtype=torch.float32)
            continue
        m = pmap_all_top.get(org, {})
        vec = [m.get(lbl, 1.0 / len(labels)) for lbl in labels]
        out_top[org] = torch.tensor(vec, dtype=torch.float32)

    # sub (schema.heads 중 organ/procedure/top 제외: head_key 기준)
    out_sub = {}
    pmap_all_sub = priors.get("sub", {})
    for hs in getattr(schema, "heads"):
        p = getattr(hs, "path", None)
        if p is None or len(p) < 2:
            continue
        if p[1] in ("__top__", "__procedure__"):
            continue
        labels = schema.labels_for(hs.path)
        if len(labels) == 0:
            out_sub[hs.key] = torch.empty(0, dtype=torch.float32)
            continue
        m = pmap_all_sub.get(hs.key, {})
        vec = [m.get(lbl, 1.0 / len(labels)) for lbl in labels]
        out_sub[hs.key] = torch.tensor(vec, dtype=torch.float32)

    return {"organ": out_organ, "procedure": out_proc, "top": out_top, "sub": out_sub}
