
import torch, os
import json
import pandas as pd
import random
from transformers import AutoModel
import h5py
from torch import Tensor, einsum, nn
import numpy as np


# Colon‑tree --------------------------------------------------------------

PROCEDURES = [
    'colonoscopic biopsy',
    'colonoscopic polypectomy',
    'colonoscopic mucosal resection',
    'colonoscopic submucosal dissection',
    'olonoscopic mucosal resection'          # original typo kept
]

diagnosis1 = {
    "principal": "Adenocarcinoma,",
    "scores": "",
    "sub_feature": [
        {
            "principal": "moderately differentiated",
            "scores": "",
            "sub_feature": [
                { "principal": "", "scores": "", "sub_feature": [] }
                # { "principal": "mucinous component", "scores": "", "sub_feature": [] }
            ]
        },
        {
            "principal": "well differentiated",
            "scores": "",
            "sub_feature": [
                { "principal": "", "scores": "", "sub_feature": [] }
                # { "principal": "mucinous component", "scores": "", "sub_feature": [] }
            ]
        },
        {
            "principal": "poorly differentiated",
            "scores": "",
            "sub_feature": [
                { "principal": "", "scores": "", "sub_feature": [] }
                # { "principal": "mucinous component", "scores": "", "sub_feature": [] }
            ]
        },
        {
            "principal": "arising from tubulovillous adenoma",
            "scores": "",
            "sub_feature": [
                { "principal": "high grade dysplasia", "scores": "", "sub_feature": [] }
            ]
        }
    ]
}

diagnosis2  = { "principal": "Mucinous carcinoma",                       "scores": "", "sub_feature": [] }
diagnosis3  = { "principal": "Signet‑ring cell carcinoma",               "scores": "", "sub_feature": [] }
diagnosis4  = { "principal": "Tubular adenoma",
                "scores": "",
                "sub_feature": [
                    { "principal": "low grade dysplasia",  "scores": "", "sub_feature": [] },
                    { "principal": "high grade dysplasia", "scores": "", "sub_feature": [] }
                ] }
diagnosis5  = { "principal": "Tubulovillous adenoma",
                "scores": "",
                "sub_feature": [
                    { "principal": "low grade dysplasia",  "scores": "", "sub_feature": [] },
                    { "principal": "high grade dysplasia", "scores": "", "sub_feature": [] }
                ] }
diagnosis6  = { "principal": "Hyperplastic polyp",                       "scores": "", "sub_feature": [] }
diagnosis7  = { "principal": "Sessile serrated lesion",                  "scores": "", "sub_feature": [] }
diagnosis8  = { "principal": "Traditional serrated adenoma",             "scores": "", "sub_feature": [] }
diagnosis9  = { "principal": "Serrated serrated lesion with low grade dysplasia", "scores": "", "sub_feature": [] }
diagnosis10 = { "principal": "Chronic active colitis",                   "scores": "", "sub_feature": [] }
diagnosis11 = { "principal": "Chronic nonspecific inflammation",         "scores": "", "sub_feature": [] }
diagnosis12 = { "principal": "Inflammatory polyp",                       "scores": "", "sub_feature": [] }
diagnosis13 = { "principal": "Malignant lymphoma",                       "scores": "", "sub_feature": [] }
diagnosis14 = { "principal": "Extranodal marginal zone B-cell lymphoma",
                "scores": "",
                "sub_feature": [
                    { "principal": "of mucosa associated lymphoid tissue (MALT lymphoma)",
                      "scores": "", "sub_feature": [] }
                ] }
diagnosis15 = { "principal": "Neuroendocrine tumor",                     "scores": "", "sub_feature": [] }
diagnosis16 = { "principal": "Small cell carcinoma",                     "scores": "", "sub_feature": [] }
diagnosis17 = { "principal": "Sessile serrated adenoma with low grade dysplasia", "scores": "", "sub_feature": [] }
diagnosis18 = { "principal": "Sessile serrated lesion with low grade dysplasia",  "scores": "", "sub_feature": [] }
diagnosis19 = { "principal": "Inflammatory polyp",                       "scores": "", "sub_feature": [] }

DIAGNOSES = [
    diagnosis1, diagnosis2, diagnosis3, diagnosis4, diagnosis5,
    diagnosis6, diagnosis7, diagnosis8, diagnosis9, diagnosis10,
    diagnosis11, diagnosis12, diagnosis13, diagnosis14, diagnosis15,
    diagnosis16, diagnosis17, diagnosis18, diagnosis19
]

COLON_TREE = {
    "procedures": PROCEDURES,
    "sub_feature": DIAGNOSES
}
# -----------------------------------------------------------------------