from __future__ import annotations
import torch, os
import json
import pandas as pd
import random
from transformers import AutoModel
import h5py
from torch import Tensor, einsum, nn
import numpy as np

import torch.nn as nn
from collections import OrderedDict
from typing import Dict, List, Any, Tuple

import json
import re
from dataclasses import dataclass
from typing import Dict, List, Tuple, Any, Optional

import torch
import torch.nn as nn
import torch.nn.functional as F


# -----------------------------
# Utilities
# -----------------------------

def snake(s: str) -> str:
    s = s.strip()
    s = s.replace("/", "_")
    s = re.sub(r"[^0-9A-Za-z]+", "_", s)
    s = re.sub(r"_+", "_", s)
    return s.strip("_").lower() or "empty"



def proc_normalize(p: str) -> str:
    """Normalize procedure strings to a canonical key and map common variants."""
    raw = (p or "").strip()
    key = raw.lower()
    key = key.replace("-", " ").replace("_", " ")
    key = re.sub(r"\s+", " ", key).strip()

    aliases = {
        # core
        "core needle biopsy": "core-needle biopsy",
        "core needle": "core-needle biopsy",
        # sono guided core
        "sono guided core biopsy": "sono-guided core biopsy",
        "ultrasound guided core biopsy": "sono-guided core biopsy",
        # mammotome / vacuum assisted
        "mammotome biopsy": "mammotome biopsy",
        "sono guided mammotome biopsy": "sono-guided mammotome biopsy",
        "vacuum assisted biopsy": "vacuum-assisted biopsy",
        # encor spelling variants
        "encore biopsy": "encor biopsy",
        "encor biopsy": "encor biopsy",
        # generic / others
        "biopsy": "biopsy",
        "lumpectomy": "lumpectomy",
    }
    return aliases.get(key, raw)


def get_all_procs_from_schema(schema: "HierSchema") -> List[str]:
    """Union of procedures across organs (canonicalized)."""
    allp: List[str] = []
    seen = set()
    for plist in schema.procedures_vocab.values():
        for p in plist:
            canon = proc_normalize(p)
            if canon not in seen:
                seen.add(canon)
                allp.append(canon)
    allp.sort()
    return allp





Path = Tuple[str, ...]  # e.g., ("breast", "ductal carcinoma in situ", "type")

# -----------------------------
# Schema builder
# -----------------------------

@dataclass
class HeadSpec:
    path: Path                 # full hierarchical path
    name: str                  # short name for readability
    n_classes: int             # output dimension
    task: str                  # "multilabel", "multiclass"
    always_selected: bool = False  # if True, selection is deterministic when parent present

    @property
    def key(self) -> str:
        return "///".join(self.path)


class HierSchema:
    """Builds heads from an organ tree spec.

    Expected organ_tree format (per user example):
    {
        "breast": {
            "procedures": [...],
            "sub_feature": [ {diagnosis dict}, ... ]
        }
    }
    Each diagnosis dict has keys: principal, scores, sub_feature(list of dicts)."""

    def __init__(self, organ_tree: Dict[str, Any], organs_vocab: List[str], extra_procedures: Optional[List[str]] = None):
        self.organ_tree = organ_tree
        self.organs_vocab = organs_vocab
        self.procedures_vocab: Dict[str, List[str]] = {}
        self.top_diagnoses: Dict[str, List[str]] = {}
        self.heads: List[HeadSpec] = []

        for organ_name, tree in organ_tree.items():
            # Procedures vocab per organ (union with extra if provided)
            procs = tree.get("procedures", [])
            
            if extra_procedures:
                for p in extra_procedures:
                    if p not in procs:
                        procs.append(p)
            self.procedures_vocab[organ_name] = procs

            # Top-level multi-label head over diagnoses            
            diag_list = [d["principal"] for d in tree.get("sub_feature", [])]

            self.top_diagnoses[organ_name] = diag_list
            self.heads.append(
                HeadSpec(path=(organ_name, "__top__"), name=f"{organ_name}_top", n_classes=len(diag_list), task="multilabel")
            )

            # Children heads per diagnosis
            for diag in tree.get("sub_feature", []):
                self._add_children_heads(organ_name, diag)


    def _add_children_heads(self, organ_name: str, node: Dict[str, Any], parent_path: Optional[Path] = None):
        # if organ_name == "Prostate":
        #     breakpoint()
        
        parent_path = parent_path or (organ_name, node["principal"])  # include diagnosis principal in path
        principal = node["principal"]
        children: List[Dict[str, Any]] = node.get("sub_feature", [])
        scores = node.get("scores", "")

        # Special rules (user's 규칙 3)
        if principal == "Invasive carcinoma of no special type":
            # Always selected sub-features: 3 score heads with classes {1,2,3}
            for name in ["Tubule formation", "Nuclear grade", "Mitoses"]:
                self.heads.append(
                    HeadSpec(
                        path=(organ_name, principal, name),
                        name=snake(name),
                        n_classes=3,
                        task="multiclass",
                        always_selected=True,
                    )
                )
            return

        elif principal == "Ductal carcinoma in situ":
            # Always selected: Type / Nuclear grade / Necrosis (each multiclass)
            # Children structure: one empty child with those 3 items as its sub_feature
            # We aggregate all terminal labels under each of the three fields.
            type_labels: List[str] = []
            ng_labels: List[str] = []
            nec_labels: List[str] = []
            for ch in children:
                for field in ch.get("sub_feature", []):
                    field_name = field["principal"]
                    terms = [t["principal"] for t in field.get("sub_feature", [])]
                    if field_name == "Type":
                        type_labels.extend(terms)
                    elif field_name == "Nuclear grade":
                        ng_labels.extend(terms)
                    elif field_name == "Necrosis":
                        nec_labels.extend(terms)
            # Deduplicate while preserving order
            def uniq(xs: List[str]) -> List[str]:
                return list(dict.fromkeys(xs))
            type_labels, ng_labels, nec_labels = uniq(type_labels), uniq(ng_labels), uniq(nec_labels)

            self.heads.append(HeadSpec(path=(organ_name, principal, "Type"), name="dcis_type", n_classes=len(type_labels), task="multiclass", always_selected=True))
            self.heads.append(HeadSpec(path=(organ_name, principal, "Nuclear grade"), name="dcis_ng", n_classes=len(ng_labels), task="multiclass", always_selected=True))
            self.heads.append(HeadSpec(path=(organ_name, principal, "Necrosis"), name="dcis_nec", n_classes=len(nec_labels), task="multiclass", always_selected=True))
            # Save label lists for later decoding
            self._save_field_labels((organ_name, principal, "Type"), type_labels)
            self._save_field_labels((organ_name, principal, "Nuclear grade"), ng_labels)
            self._save_field_labels((organ_name, principal, "Necrosis"), nec_labels)
            return

        else:
        # elif principal == "Acinar adenocarcinoma":
            feature = node
            feat_path = ( organ_name, f'{node["principal"]}')
            self.search_heads(feature, feat_path)



    def search_heads(self, feature, path):
        if len(feature["scores"]) > 0:
            feature_name = f'{feature["principal"]}__scores'
            labels = feature["scores"]
            path_score=tuple( list(path) + ["score"] )
            self.heads.append(HeadSpec(path=path_score, name=feature_name, n_classes=len(labels), \
                                task="multiclass", always_selected=True))
            self._save_field_labels(path_score, labels)



        if len(feature["sub_feature"]) > 0:
            feature_name = f'{feature["principal"]}__child'
            labels = [c["principal"] for c in feature["sub_feature"]]
            self.heads.append(HeadSpec(path=path, name=feature_name, n_classes=len(labels), \
                                task="multiclass", always_selected=True))
            self._save_field_labels(path, labels)

            for sub_feature in feature["sub_feature"]:
                sub = sub_feature["principal"]
                sub_path = tuple( list(path) + [sub] )

                self.search_heads(sub_feature, sub_path)





    # ---------- label vocabulary per head ----------
    def _save_field_labels(self, path: Path, labels: List[str]):
        if not hasattr(self, "field_label_vocab"):
            self.field_label_vocab: Dict[Path, List[str]] = {}
        self.field_label_vocab[path] = labels

    def labels_for(self, path: Path) -> List[str]:
        return self.field_label_vocab.get(path, [])


# -----------------------------
# Model definition
# -----------------------------

@dataclass
class ForwardOutputs:
    organ_logits: torch.Tensor
    colon_logits: torch.Tensor
    procedure_logits: Dict[str, torch.Tensor]
    top_diag_logits: Dict[str, torch.Tensor]  # organ -> logits (B, n_top)
    head_logits: Dict[str, torch.Tensor]      # path_key -> logits (B, C)





class MLP(nn.Module):
    def __init__(self, in_features, out_features, dropout=0.1):
        super().__init__()
        self.proj1 = nn.Linear(in_features, in_features)
        self.act = nn.GELU()
        self.drop1 = nn.Dropout(dropout)

        self.proj2 = nn.Linear(in_features, out_features)

    def forward(self, x):
        return self.proj2(self.drop1(self.act(self.proj1(x))))



# class MLP(nn.Module):
#     def __init__(self, in_features, out_features, hidden_features=None, dropout=0.2):
#         super().__init__()
#         hidden_features = hidden_features or in_features

#         self.fc1 = nn.Linear(in_features, hidden_features)
#         self.norm1 = nn.LayerNorm(hidden_features)
#         self.act1 = nn.GELU()
#         self.drop1 = nn.Dropout(dropout)

#         self.fc2 = nn.Linear(hidden_features, hidden_features//4)
#         self.norm2 = nn.LayerNorm(hidden_features//4)
#         self.act2 = nn.GELU()
#         self.drop2 = nn.Dropout(dropout)

#         self.fc_out = nn.Linear(hidden_features//4, out_features)

#     def forward(self, x):
#         x = self.fc1(x)
#         x = self.norm1(x)
#         x = self.act1(x)
#         x = self.drop1(x)

#         x = self.fc2(x)
#         x = self.norm2(x)
#         x = self.act2(x)
#         x = self.drop2(x)

#         x = self.fc_out(x)
#         return x





class MLP_Deep(nn.Module):
    def __init__(self, in_features, out_features):
        super().__init__()

        self.n_interm_layers = 3
        hidden_dim = in_features // 2

        # Layer 1
        self.proj1 = nn.Linear(in_features, hidden_dim, bias=True)
        self.bn1 = nn.BatchNorm1d(hidden_dim)

        # Intermediate layers
        self.projs_interm = nn.ModuleList()
        self.bns_interm = nn.ModuleList()
        for _ in range(self.n_interm_layers):
            layer = nn.Linear(hidden_dim, hidden_dim, bias=True)
            bn = nn.BatchNorm1d(hidden_dim)
            self.projs_interm.append(layer)
            self.bns_interm.append(bn)

        # Output layer
        self.proj2 = nn.Linear(hidden_dim, out_features, bias=True)

        # Activation
        self.act = nn.GELU()

        # Initialization
        self._init_weights()

    def _init_weights(self):
        # Xavier init for GELU
        nn.init.xavier_normal_(self.proj1.weight)
        nn.init.constant_(self.proj1.bias, 0.01)

        for layer in self.projs_interm:
            nn.init.xavier_normal_(layer.weight)
            nn.init.constant_(layer.bias, 0.01)

        nn.init.xavier_normal_(self.proj2.weight)
        nn.init.constant_(self.proj2.bias, 0.01)

    def forward(self, x):
        x = self.act(self.bn1(self.proj1(x)))
        for proj, bn in zip(self.projs_interm, self.bns_interm):
            x = self.act(bn(proj(x)))
        return self.proj2(x)
    


class HierarchicalClassifier(nn.Module):
    def __init__(self, embed_dim: int, schema: HierSchema, probe_type: str):
        super().__init__()
        self.schema = schema

        if probe_type == "linear":
            probe_module = nn.Linear
        elif probe_type == "mlp":
            probe_module = MLP
        elif probe_type == "mlp_deep":
            probe_module = MLP_Deep


        # Organ and procedure heads
        self.organ_head = probe_module(embed_dim, len(self.schema.organs_vocab))
        # We use a union of procedures across organs for a single head; alternatively, per-organ heads could be used
        self.colon_head = probe_module(embed_dim, 2) # 2: Colon, 3: Rectum


        self.procedure_head = nn.ModuleDict({
            organ: probe_module(embed_dim, len(proc_list))
            for organ, proc_list in self.schema.procedures_vocab.items()
        })

        # Top-level diagnosis heads (per organ)
        self.top_diag_heads = nn.ModuleDict()
        for organ, diags in self.schema.top_diagnoses.items():
            self.top_diag_heads[organ] = probe_module(embed_dim, len(diags))

        # Sub-heads
        self.sub_heads = nn.ModuleDict()
        for hs in self.schema.heads:
            if hs.path[1] == "__top__":
                continue
            self.sub_heads[hs.key] = probe_module(embed_dim, hs.n_classes)

    # ---- forward ----
    def forward(self, x: torch.Tensor, organ_hint: Optional[str] = None) -> ForwardOutputs:
        # x: (B, D)
        h = x

        # Choose organ head
        organ_logits = self.organ_head(h)  # (B, |ORGANS|)
        colon_logits = self.colon_head(h)

        proc_logits: Dict[str, torch.Tensor] = {}
        for key in self.schema.procedures_vocab.keys():
            head = self.procedure_head[key]
            proc_logits[key] = head(h)

        top_logits: Dict[str, torch.Tensor] = {}
        for key in self.schema.top_diagnoses.keys():
            head = self.top_diag_heads[key]
            top_logits[key] = head(h)


        head_logits: Dict[str, torch.Tensor] = {}
        for hs in self.schema.heads:
            if hs.path[1] == "__top__":
                continue
            head = self.sub_heads[hs.key]
            head_logits[hs.key] = head(h)
        
        return ForwardOutputs(
            organ_logits=organ_logits,
            colon_logits=colon_logits,
            procedure_logits=proc_logits,
            top_diag_logits=top_logits,
            head_logits=head_logits,
        )

    # ---- decoding with rules ----
    def decode(self, outs: ForwardOutputs, threshold: float = 0.5) -> List[Dict[str, Any]]:
        B = outs.organ_logits.size(0)
        results: List[Dict[str, Any]] = []
        for b in range(B):
            organ_idx = int(outs.organ_logits[b].argmax().item())
            organ = self.schema.organs_vocab[organ_idx]

            if organ in ["Colon", "Rectum"]:
                colon_idx = int(outs.colon_logits[b].argmax().item())
                organ = "Colon" if colon_idx==0 else "Rectum"

            proc_idx = int(outs.procedure_logits[organ][b].argmax().item())
            procedure = self.schema.procedures_vocab[organ][proc_idx]

            # top-level multi-label
            top_logits = outs.top_diag_logits[organ][b]
            top_probs = torch.sigmoid(top_logits)
            present_mask = (top_probs >= threshold)
            if present_mask.sum() == 0:
                idx_max = top_probs.argmax()
                present_mask[idx_max] = True
            diag_names = self.schema.top_diagnoses[organ]
            present_diags = [d for i, d in enumerate(diag_names) if bool(present_mask[i])]

            # Build JSON
            sample = {"organ": organ, "procedure": procedure, "sub_feature": []}

            # Iterate top diags with rules
            for diag in present_diags:
                if diag == "Invasive carcinoma of no special type":
                    entry = {"principal": diag, "scores": "", "sub_feature": []}
                    for name in ["Tubule formation", "Nuclear grade", "Mitoses"]:
                        key = "///".join((organ, diag, name))
                        logits = outs.head_logits[key][b]
                        idx = int(logits.argmax().item())
                        score = str(idx + 1)  # classes are 3 -> map to 1..3
                        entry["sub_feature"].append({"principal": name, "scores": score, "sub_feature": []})
                    sample["sub_feature"].append(entry)
                    continue

                if diag == "Ductal carcinoma in situ":
                    entry = {"principal": diag, "scores": "", "sub_feature": []}
                    # Type
                    key = "///".join((organ, diag, "Type"))
                    if key in outs.head_logits:
                        idx = int(outs.head_logits[key][b].argmax().item())
                        label = self.schema.labels_for((organ, diag, "Type"))[idx]
                        entry["sub_feature"].append({"principal": "Type", "scores": "", "sub_feature": [{"principal": label, "scores": "", "sub_feature": []}]})
                    # Nuclear grade
                    key = "///".join((organ, diag, "Nuclear grade"))
                    if key in outs.head_logits:
                        idx = int(outs.head_logits[key][b].argmax().item())
                        label = self.schema.labels_for((organ, diag, "Nuclear grade"))[idx]
                        entry["sub_feature"].append({"principal": "Nuclear grade", "scores": "", "sub_feature": [{"principal": label, "scores": "", "sub_feature": []}]})
                    # Necrosis
                    key = "///".join((organ, diag, "Necrosis"))
                    if key in outs.head_logits:
                        idx = int(outs.head_logits[key][b].argmax().item())
                        label = self.schema.labels_for((organ, diag, "Necrosis"))[idx]
                        entry["sub_feature"].append({"principal": "Necrosis", "scores": "", "sub_feature": [{"principal": label, "scores": "", "sub_feature": []}]})
                    sample["sub_feature"].append(entry)
                    continue


                # Generic one-level children multiclass
                # Determine if a child head exists for this diagnosis
                def decode_recursive(organ, path_, comp):
                    entry = {"principal": comp, "scores": "", "sub_feature": []}

                    key_sub = "///".join(path_)
                    path_score = tuple( list(path_)+["score"] )
                    key_score = "///".join(path_score)

                    if key_score in outs.head_logits.keys():
                        logits = outs.head_logits[key_score][b]
                        idx = int(logits.argmax().item())
                        label_list = self.schema.labels_for(path_score)
                        label = label_list[idx] if idx < len(label_list) else ""
                        entry["scores"] = label

                    if key_sub in outs.head_logits.keys(): 
                        logits = outs.head_logits[key_sub][b]
                        idx = int(logits.argmax().item())
                        label_list = self.schema.labels_for(path_)
                        label = label_list[idx] if idx < len(label_list) else ""

                        sub_path = tuple( list(path_) + [label] )

                        entry["sub_feature"].append(decode_recursive(organ, sub_path, label))

                    return entry


                path = (organ, diag)
                sample["sub_feature"].append(decode_recursive(organ, path, diag))


            results.append(sample)
        return results


# -----------------------------
# Label building helpers (optional)
# -----------------------------

@dataclass
class BuiltTargets:
    organ: torch.Tensor           # (B,)
    procedure: Dict[str, torch.Tensor]       # (B,)
    top_diag: Dict[str, torch.Tensor]      # (B, n_top) multi-hot
    heads: Dict[str, torch.Tensor]


def build_targets(records: List[Dict[str, Any]],
                  schema: HierSchema,
                  device: str,
                  organs_vocab: List[str]) -> BuiltTargets:
    """
    Turn JSON records into training targets aligned with the schema.

    - Organ: index in organs_vocab
    - Procedure: index in model-style union of procedures: sorted(set(schema.procedures_vocab[...] ...))
    - Top diagnoses: multi-hot over organ-specific top vocab, padded to max_top across organs
    - Sub heads: CrossEntropy targets (index), -1 = ignore
    """
    B = len(records)
    organ_ids = torch.zeros(B, dtype=torch.long).to(device)

    proc_ids_all = {organ: torch.full((B,), -1, dtype=torch.long).to(device) \
                    for organ in organs_vocab}

    # ---- helper: case-insensitive schema organ key lookup ----
    def match_schema_organ(organ_str: str) -> str:
        for k in schema.top_diagnoses.keys():
            if k.lower() == organ_str.lower():
                return k
        # fallback: return as-is (will likely error later if truly unknown)
        return organ_str

    # ---- procedure vocab used by the *model* (must mirror model's construction) ----

    model_all_procs = dict()
    for key,val in schema.procedures_vocab.items():
        proc_index = {p: i for i, p in enumerate(val)}
        model_all_procs[key] = proc_index

    # simple alias normalizer (inline to keep function self-contained)
    def proc_normalize(p: str) -> str:
        raw = (p or "").strip()
        key = raw.lower().replace("-", " ").replace("_", " ")
        key = re.sub(r"\s+", " ", key).strip()
        aliases = {
            # core
            "core needle biopsy": "core-needle biopsy",
            "core needle": "core-needle biopsy",
            # sono guided core
            "sono guided core biopsy": "sono-guided core biopsy",
            "ultrasound guided core biopsy": "sono-guided core biopsy",
            # mammotome / vacuum assisted
            "mammotome biopsy": "mammotome biopsy",
            "sono guided mammotome biopsy": "sono-guided mammotome biopsy",
            "vacuum assisted biopsy": "vacuum-assisted biopsy",
            # encor spelling variants
            "encore biopsy": "encor biopsy",
            "encor biopsy": "encor biopsy",
            # generic / others
            "biopsy": "biopsy",
            "lumpectomy": "lumpectomy",
        }
        return aliases.get(key, raw)


    # ---- top diag tensor (pad to max across organs) ----
    top_all = dict()

    for key, val in schema.top_diagnoses.items():
        len_top = len(val)
        top_all[key] = torch.full((B, len_top), -1, dtype=torch.float32).to(device)           

    # ---- sub-head targets ----
    head_tgts: Dict[str, torch.Tensor] = {}
    for hs in schema.heads:
        if hs.path[1] == "__top__":
            continue
        head_tgts[hs.key] = torch.full((B,), -1, dtype=torch.long).to(device)  # -1: ignore

    # ---- iterate samples ----
    for i, rec in enumerate(records):
        organ_raw = rec.get("organ", "").strip()
        if organ_raw not in organs_vocab:
            # allow mismatch in case if vocab contains capitalized organ names
            if not any(organ_raw.lower() == ov.lower() for ov in organs_vocab):
                raise ValueError(f"Unknown organ '{organ_raw}' in record {rec.get('id','?')}")
            # map to first case-insensitive match
            organ_idx = next(j for j, ov in enumerate(organs_vocab) if ov.lower() == organ_raw.lower())
        else:
            organ_idx = organs_vocab.index(organ_raw)
        organ_ids[i] = organ_idx

        schema_organ = match_schema_organ(organs_vocab[organ_idx])

        # ---- procedure index ----
        proc_raw = rec.get("procedure", "").strip()
        # direct match first
        for key, val in model_all_procs.items():

            if organ_raw.lower() == key.lower():
                if proc_raw in val.keys():
                    proc_ids_all[key][i] = val[proc_raw]
                else:
                    # try alias normalization
                    canon = proc_normalize(proc_raw)
    
                    if canon in val:
                        proc_ids_all[key][i] = val[canon]
                    else:
                        # SAFETY: fall back to 0 to avoid dimension mismatch with model head.
                        # (If you want strict mode, raise an error here.)
                        proc_ids_all[key][i] = 0



        # ---- top-level multi-hot ----   
        top_vocab = schema.top_diagnoses[schema_organ]
        present = [sf.get("principal", "") for sf in rec.get("sub_feature", [])]
        
        top_all[schema_organ][i] = 0
        for p in present:
            p = p.strip()
            if p in top_vocab:
                top_all[schema_organ][i, top_vocab.index(p)] = 1.0

        # ---- helper to find a diagnosis block ----
        def find_diag(name: str) -> Optional[Dict[str, Any]]:
            for sf in rec.get("sub_feature", []):
                if sf.get("principal") == name:
                    return sf
            return None

        # ---- IC-NST scores (always-present fields, we only score 1..3) ----
        ic = find_diag("Invasive carcinoma of no special type")
        if ic is not None:
            for jname in ["Tubule formation", "Nuclear grade", "Mitoses"]:
                key = "///".join((schema_organ, "Invasive carcinoma of no special type", jname))
                if key in head_tgts:
                    for ch in ic.get("sub_feature", []):
                        if ch.get("principal") == jname:
                            s = (ch.get("scores", "") or "").strip()
                            if s in {"1", "2", "3"}:
                                head_tgts[key][i] = int(s) - 1
                            break

        # ---- DCIS fields (Type / Nuclear grade / Necrosis) ----
        dcis = find_diag("Ductal carcinoma in situ")
        if dcis is not None:
            for field in ["Type", "Nuclear grade", "Necrosis"]:
                key = "///".join((schema_organ, "Ductal carcinoma in situ", field))
                labels = schema.labels_for((schema_organ, "Ductal carcinoma in situ", field))
                if key in head_tgts and labels:
                    leaf = None
                    for item in dcis.get("sub_feature", []):
                        if item.get("principal") == field:
                            sub = item.get("sub_feature", [])
                            if sub:
                                leaf = sub[0].get("principal")
                            break
                    if leaf and leaf in labels:
                        head_tgts[key][i] = labels.index(leaf)


        # ---- Generic one-level children ----
        def build_recursive(organ, path_, feature):
            if len(feature["scores"]) > 0:
                path_score=tuple( list(path_) + ["score"] )
                key_score = "///".join(path_score)

                labels = schema.labels_for(path_score)
                if key_score in head_tgts and labels:
                    child_label = feature.get("scores", "")
                    
                    if child_label in labels:
                        head_tgts[key_score][i] = labels.index(child_label)

            if len(feature["sub_feature"]) > 0:
                for sub_feature in feature["sub_feature"]:

                    key = "///".join(path_)

                    labels = schema.labels_for(path_)
                    if key in head_tgts.keys() and labels:
                        child_label = sub_feature.get("principal", "")
                        
                        if child_label in labels:
                            head_tgts[key][i] = labels.index(child_label)
                            # print("subclasses assigned!!")

                    sub = sub_feature["principal"]
                    sub_path = tuple( list(path_) + [sub] )

                    build_recursive(organ, sub_path, sub_feature)


        for diag in schema.top_diagnoses[schema_organ]:
            if diag in ["Invasive carcinoma of no special type", "Ductal carcinoma in situ"]:
                continue
            diag_sf = find_diag(diag)
            if diag_sf is None:
                continue

            path = (schema_organ, diag)
            build_recursive(schema_organ, path, diag_sf)


    return BuiltTargets(
        organ=organ_ids,
        procedure=proc_ids_all,
        top_diag=top_all,
        heads=head_tgts,
    )


# -----------------------------
# Loss
# -----------------------------

class HierLoss(nn.Module):
    def __init__(self, schema: HierSchema, organ_weight: float = 0.2, proc_weight: float = 0.2, top_weight: float = 1.0, sub_weight: float = 1.0):
        super().__init__()
        self.schema = schema
        self.top_w = top_weight
        self.organ_w = organ_weight
        self.proc_w = proc_weight
        self.sub_w = sub_weight
        self.bce = nn.BCEWithLogitsLoss(reduction='none')
        self.ce = nn.CrossEntropyLoss(ignore_index=-1, reduction="none")

    def forward(self, outs: ForwardOutputs, tgts: BuiltTargets) -> Dict[str, torch.Tensor]:
        losses = {}
        losses["organ"] = self.ce(outs.organ_logits, tgts.organ).mean()

        losses["procedure"] = 0
        for organ in tgts.procedure.keys():
            valid_label = tgts.procedure[organ] != -1
            organ_loss = self.ce(outs.procedure_logits[organ][valid_label], tgts.procedure[organ][valid_label])
            losses["procedure"] += organ_loss.sum()

        losses["procedure"] /= outs.organ_logits.size(0)

        # top-level
        losses["top"] = 0
        for organ in tgts.top_diag.keys():
            valid_label = tgts.top_diag[organ].sum(dim=-1) > 0
            organ_loss = self.bce(outs.top_diag_logits[organ][valid_label], tgts.top_diag[organ][valid_label])
            
            losses["top"] += organ_loss.mean(dim=-1).sum()

        losses["top"] /= outs.organ_logits.size(0)

        # sub-heads
        sub_total = 0.0
        n = 0

        for key, logits in outs.head_logits.items():
            tgt = tgts.heads.get(key)
            valid_idx = (tgt!=-1)
            if valid_idx.sum() != 0:
                sub_total += self.ce(logits[valid_idx], tgt[valid_idx]).mean()
            n += 1

        losses["sub"] = sub_total / max(n, 1)

        total = self.organ_w * losses["organ"] + self.proc_w * losses["procedure"] + self.top_w * losses["top"] + self.sub_w * losses["sub"]
        losses["total"] = total

        return losses




class HierLossMixup(nn.Module):
    def __init__(self, schema: HierSchema, organ_weight: float = 0.2, proc_weight: float = 0.2, top_weight: float = 1.0, sub_weight: float = 1.0):
        super().__init__()
        self.schema = schema
        self.top_w = top_weight
        self.organ_w = organ_weight
        self.proc_w = proc_weight
        self.sub_w = sub_weight
        self.bce = nn.BCEWithLogitsLoss(reduction='none')
        self.ce = nn.CrossEntropyLoss(ignore_index=-1, reduction="none")

    def forward(self, outs, tgts_list, ratio) -> Dict[str, torch.Tensor]:


        losses = {}

        losses_list = []
        losses_colon_list = []
        for tgts in tgts_list:
            loss = self.ce(outs.organ_logits, tgts.organ)

            losses_list.append(loss)

            is_colon = torch.logical_or(tgts.organ==2, tgts.organ==3)
            loss_colon = torch.zeros(outs.organ_logits.size(0)).to(tgts.organ.device)
            if is_colon.sum().item() > 0: 
                loss_colon[is_colon] = self.ce(outs.colon_logits[is_colon], tgts.organ[is_colon]-2)
            losses_colon_list.append(loss_colon)

        losses["organ"] = ( ratio * losses_list[0] + (1-ratio) * losses_list[1] ).mean()
        losses["colon"] = ( ratio * losses_colon_list[0] + (1-ratio) * losses_colon_list[1] ).mean()


        losses["procedure"] = 0
        for t_idx, tgts in enumerate(tgts_list):
            for organ in tgts.procedure.keys():
                valid_label = tgts.procedure[organ] != -1
                valid_ratio = ratio[valid_label] if t_idx == 0 else (1-ratio[valid_label])

                organ_loss = valid_ratio * self.ce(outs.procedure_logits[organ][valid_label], tgts.procedure[organ][valid_label])

                losses["procedure"] += organ_loss.sum()

        losses["procedure"] /= outs.organ_logits.size(0)


        # top-level
        losses["top"] = 0
        for t_idx, tgts in enumerate(tgts_list):
            for organ in tgts.top_diag.keys():
                valid_label = tgts.top_diag[organ].sum(dim=-1) > 0
                valid_ratio = ratio[valid_label] if t_idx == 0 else (1-ratio[valid_label])
                
                organ_loss = valid_ratio[:, None] * self.bce(outs.top_diag_logits[organ][valid_label], tgts.top_diag[organ][valid_label])

                losses["top"] += organ_loss.mean(dim=-1).sum()

        losses["top"] /= outs.organ_logits.size(0)

        # sub-heads
        sub_total = 0.0
        n = 0

        for t_idx, tgts in enumerate(tgts_list):
            for key, logits in outs.head_logits.items():
                tgt = tgts.heads.get(key)
                valid_idx = (tgt!=-1)
                if valid_idx.sum() != 0:
                    valid_ratio = ratio[valid_idx] if t_idx == 0 else (1-ratio[valid_idx])
                    sub_total += (valid_ratio * self.ce(logits[valid_idx], tgt[valid_idx])).mean()
                
                n += 0.5

        losses["sub"] = sub_total / max(n, 1)

        total = self.organ_w * (losses["organ"]+losses["colon"]) + self.proc_w * losses["procedure"] + self.top_w * losses["top"] + self.sub_w * losses["sub"]
        losses["total"] = total


        return losses




# -----------------------------
# Example: build schema for provided BREAST_TREE and create the model
# -----------------------------

def build_schema(organ_tree, ORGANS: List[str], observed_extra_procedures: Optional[List[str]] = None) -> HierSchema:
    return HierSchema(organ_tree=organ_tree, organs_vocab=ORGANS, extra_procedures=observed_extra_procedures)
 

# The following helper demonstrates how to extract additional procedures from a dataset file
def extract_procedures_from_json(path: str) -> List[str]:
    with open(path, "r") as f:
        data = json.load(f)
    procs = []
    for rec in data:
        p = rec.get("procedure", "")
        if p and p not in procs:
            procs.append(p)
    return procs
