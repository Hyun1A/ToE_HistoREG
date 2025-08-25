
import torch, os
import json
import pandas as pd
import random
from transformers import AutoModel
import h5py
from torch import Tensor, einsum, nn
import numpy as np

# Cervix‑tree -------------------------------------------------------------

PROCEDURES = [
    "colposcopic biopsy",
    "punch biopsy",
    "polypectomy biopsy",
    "frozen biopsy",
    "Loop Electrosurgical Excision Procedure"  # LEEP
]

diagnosis1 = {
    "principal": "Low-grade squamous intraepithelial lesion (LSIL)",
    "scores": "",
    "sub_feature": [
        {
            "principal": "CIN 1",
            "scores": "",
            "sub_feature": []
        }
    ]
}

diagnosis2 = {
    "principal": "High-grade squamous intraepithelial lesion (HSIL)",
    "scores": "",
    "sub_feature": [
        {
            "principal": "CIN 2",
            "scores": "",
            "sub_feature": [
                {
                    "principal": "",
                    "scores": "",
                    "sub_feature": []
                },
                {
                    "principal": "with glandular involvement",
                    "scores": "",
                    "sub_feature": []
                }
            ]
        },
        {
            "principal": "CIN 3",
            "scores": "",
            "sub_feature": [
                {
                    "principal": "",
                    "scores": "",
                    "sub_feature": []
                },
                {
                    "principal": "with glandular involvement",
                    "scores": "",
                    "sub_feature": []
                }
            ]
        }
    ]
}

diagnosis3  = {"principal": "Chronic nonspecific cervicitis",          "scores": "", "sub_feature": []}
diagnosis4  = {"principal": "Invasive squamous cell carcinoma",        "scores": "", "sub_feature": []}
diagnosis5  = {"principal": "Endocervical adenocarcinoma in situ (AIS)","scores": "", "sub_feature": []}
diagnosis6  = {"principal": "Endocervical adenocarcinoma",             "scores": "", "sub_feature": []}
diagnosis7  = {"principal": "Adenocarcinoma",                          "scores": "", "sub_feature": []}
diagnosis8  = {"principal": "Endometrioid carcinoma",                  "scores": "", "sub_feature": []}
diagnosis9  = {"principal": "Endocervical polyp",                      "scores": "", "sub_feature": []}
diagnosis10 = {"principal": "Microglandular hyperplasia",              "scores": "", "sub_feature": []}
diagnosis11 = {"principal": "Chronic active cervicitis",               "scores": "", "sub_feature": []}
diagnosis12 = {"principal": "Endometrioid adenocarcinoma",             "scores": "", "sub_feature": []}
diagnosis13 = {"principal": "Adenosquamous carcinoma",                 "scores": "", "sub_feature": []}
diagnosis14 = {"principal": "Large cell neuroendocrine carcinoma",     "scores": "", "sub_feature": []}
diagnosis15 = {"principal": "Metastatic high grade serous carcinoma",  "scores": "", "sub_feature": []}

DIAGNOSES = [
    diagnosis1, diagnosis2, diagnosis3,  diagnosis4,  diagnosis5,
    diagnosis6, diagnosis7, diagnosis8,  diagnosis9,  diagnosis10,
    diagnosis11, diagnosis12, diagnosis13, diagnosis14, diagnosis15
]

CERVIX_TREE = {
    "procedures": PROCEDURES,
    "sub_feature": DIAGNOSES
}
# ------------------------------------------------------------------------