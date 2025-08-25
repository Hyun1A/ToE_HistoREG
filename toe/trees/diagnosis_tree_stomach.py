
import torch, os
import json
import pandas as pd
import random
from transformers import AutoModel
import h5py
from torch import Tensor, einsum, nn
import numpy as np

PROCEDURES = [
    "endoscopic biopsy",
    "endoscopic submucosal dissection",
    "endoscopic mucosal resection",
    "biopsy"
]

diagnosis1 = {
    "principal": "Adenocarcinoma,",
    "scores": "",
    "sub_feature": [
        {"principal": "well differentiated",        "scores": "", "sub_feature": []},
        {"principal": "moderately differentiated",  "scores": "", "sub_feature": []},
        {
            "principal": "poorly differentiated",
            "scores": "",
            "sub_feature": [
                {"principal": "with poorly cohesive carcinoma component", "scores": "", "sub_feature": []},
                {"principal": "", "scores": "", "sub_feature": []}     
            ]
        }
    ]
}

diagnosis2 = {
    "principal": "Poorly cohesive carcinoma,",
    "scores": "",
    "sub_feature": [
        {"principal": "signet ring cell type",     "scores": "", "sub_feature": []},
        {"principal": "not otherwise specified",   "scores": "", "sub_feature": []},
    ]
}

diagnosis3 = {
    "principal": "Tubular adenoma",
    "scores": "",
    "sub_feature": [
        {"principal": "with low grade dysplasia",  "scores": "", "sub_feature": []},
        {"principal": "with high grade dysplasia", "scores": "", "sub_feature": []}
    ]
}

diagnosis4  = {"principal": "Extranodal marginal zone B cell lymphoma of MALT type", "scores": "", "sub_feature": []}
diagnosis5  = {"principal": "Chronic gastritis",                    "scores": "", "sub_feature": []}
diagnosis6  = {
    "principal": "Chronic active gastritis",
    "scores": "",
    "sub_feature": [
        {"principal": "erosion",                       "scores": "", "sub_feature": []},
        {"principal": "interstinal metaplasia",        "scores": "", "sub_feature": []},
        {"principal": "lymphoid aggregates",           "scores": "", "sub_feature": []},
        {"principal": "foveolar epithelial hyperplasia","scores": "", "sub_feature": []},
        {"principal": "CMV-infected cells",            "scores": "", "sub_feature": []},
        {"principal": "", "scores": "", "sub_feature": []}     
    ]
}
diagnosis7  = {"principal": "Gastrointestinal stromal tumor",       "scores": "", "sub_feature": []}
diagnosis8  = {"principal": "Malignant lymphoma",                   "scores": "", "sub_feature": []}
diagnosis9  = {"principal": "Malignant melanoma",                   "scores": "", "sub_feature": []}
diagnosis10 = {"principal": "Neuroendocrine tumor",
               "scores": "",
               "sub_feature": [
                   {"principal": "grade 1",          "scores": "", "sub_feature": []}
               ]}
diagnosis11 = {"principal": "Small cell carcinoma",                 "scores": "", "sub_feature": []}
diagnosis12 = {"principal": "Squamous cell carcinoma",              "scores": "", "sub_feature": []}
diagnosis13 = {"principal": "Amyloidosis",                          "scores": "", "sub_feature": []}
diagnosis14 = {"principal": "Fundic gland polyp",                   "scores": "", "sub_feature": []}
diagnosis15 = {"principal": "Hyperplastic polyp",                   "scores": "", "sub_feature": []}
diagnosis16 = {"principal": "Mucinous adenocarcinoma",              "scores": "", "sub_feature": []}

DIAGNOSES = [
    diagnosis1, diagnosis2, diagnosis3, diagnosis4, diagnosis5,
    diagnosis6, diagnosis7, diagnosis8, diagnosis9, diagnosis10,
    diagnosis11, diagnosis12, diagnosis13, diagnosis14, diagnosis15,
    diagnosis16
]

STOMACH_TREE = {
    "procedures": PROCEDURES,
    "sub_feature": DIAGNOSES
}
# ------------------------------------------------------------------------