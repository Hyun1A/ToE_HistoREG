
import torch, os
import json
import pandas as pd
import random
from transformers import AutoModel
import h5py
from torch import Tensor, einsum, nn
import numpy as np

# Rectum‑tree -------------------------------------------------------------

PROCEDURES = [
    "colonoscopic biopsy",
    "colonoscopic submucosal dissection",
    "colonoscopic mucosal resection",
    "colonoscopic polypectomy",
]

diagnosis1 = {
    "principal": "Adenocarcinoma,",
    "scores": "",
    "sub_feature": [
        {"principal": "well differentiated",       "scores": "", "sub_feature": []},
        {"principal": "moderately differentiated", "scores": "", "sub_feature": []},
        {"principal": "poorly differentiated",     "scores": "", "sub_feature": []},
    ],
}

diagnosis2 = {
    "principal": "Signet‑ring cell carcinoma",
    "scores": "",
    "sub_feature": [],
}

diagnosis3 = {
    "principal": "Tubular adenoma",
    "scores": "",
    "sub_feature": [
        {"principal": "low grade dysplasia",  "scores": "", "sub_feature": []},
        {"principal": "high grade dysplasia", "scores": "", "sub_feature": []},
    ],
}

diagnosis4 = {
    "principal": "Tubulovillous adenoma",
    "scores": "",
    "sub_feature": [
        {"principal": "low grade dysplasia",  "scores": "", "sub_feature": []},
        {"principal": "high grade dysplasia", "scores": "", "sub_feature": []},
    ],
}

diagnosis5  = {"principal": "Hyperplastic polyp",                                           "scores": "", "sub_feature": []}
diagnosis6  = {"principal": "Traditional serrated adenoma with low grade dysplasia",       "scores": "", "sub_feature": []}
diagnosis7  = {"principal": "Serrated serrated lesion with low grade dysplasia",           "scores": "", "sub_feature": []}
diagnosis8  = {"principal": "Chronic nonspecific inflammation",                            "scores": "", "sub_feature": []}
diagnosis9  = {"principal": "Gastrointestinal stromal tumor",                              "scores": "", "sub_feature": []}
diagnosis10 = {"principal": "Malignant lymphoma",                                          "scores": "", "sub_feature": []}

diagnosis11 = {
    "principal": "Extranodal marginal zone B cell lymphoma of mucosa",
    "scores": "",
    "sub_feature": [
        {"principal": "associated lymphoid tissue (MALT lymphoma)", "scores": "", "sub_feature": []},
    ],
}

diagnosis12 = {
    "principal": "Neuroendocrine tumor",
    "scores": "",
    "sub_feature": [
        {"principal": "grade 1",            "scores": "", "sub_feature": []},
        {"principal": "G1",                 "scores": "", "sub_feature": []},
        {"principal": "probably grade 1",   "scores": "", "sub_feature": []},
    ],
}

diagnosis13 = {"principal": "Squamous cell carcinoma",                                      "scores": "", "sub_feature": []}

DIAGNOSES = [
    diagnosis1,  diagnosis2,  diagnosis3,  diagnosis4,  diagnosis5,
    diagnosis6,  diagnosis7,  diagnosis8,  diagnosis9,  diagnosis10,
    diagnosis11, diagnosis12, diagnosis13,
]

RECTUM_TREE = {
    "procedures": PROCEDURES,
    "sub_feature": DIAGNOSES,
}
# -------------------------------------------------------------------------