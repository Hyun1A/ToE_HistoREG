
import torch, os
import json
import pandas as pd
import random
from transformers import AutoModel
import h5py
from torch import Tensor, einsum, nn
import numpy as np

PROCEDURES = ['biopsy']

diagnosis1 = {
    "principal": "Adenocarcinoma",
    "scores": "",
    "sub_feature": []
}

diagnosis2 = {
    "principal": "Non-small cell carcinoma,",
    "scores": "",
    "sub_feature": [
        {
            "principal": "favor adenocarcinoma",
            "scores": "",
            "sub_feature": []
        },
        {
            "principal": "favor squamous cell carcinoma",
            "scores": "",
            "sub_feature": []
        },
        {
            "principal": "not otherwise specified",
            "scores": "",
            "sub_feature": []
        }
    ]
}

diagnosis3 = {
    "principal": "Invasive mucinous adenocarcinoma",
    "scores": "",
    "sub_feature": []
}

diagnosis4 = {
    "principal": "Squamous cell carcinoma",
    "scores": "",
    "sub_feature": []
}

diagnosis5 = {
    "principal": "Small cell carcinoma",
    "scores": "",
    "sub_feature": []
}

diagnosis6 = {
    "principal": "Fungal ball",
    "scores": "",
    "sub_feature": [
        {
            "principal": "morphologically consistent with Aspergillus spp.",
            "scores": "",
            "sub_feature": []
        },
    ]
}

diagnosis7 = {
    "principal": "Carcinoid/neuroendocrine tumor, NOS",
    "scores": "",
    "sub_feature": []
}

diagnosis8 = {
    "principal": "Chronic granulomatous inflammation with necrosis",
    "scores": "",
    "sub_feature": []
}

diagnosis9 = {
    "principal": "Chronic granulomatous inflammation without necrosis",
    "scores": "",
    "sub_feature": []
}

diagnosis10 = {
    "principal": "Fungal infection",
    "scores": "",
    "sub_feature": [
        {
            "principal": "morphologically consistent with cryptococcus spp.",
            "scores": "",
            "sub_feature": []
        },
    ]
}

diagnosis11 = {
    "principal": "No evidence of malignancy or granuloma",
    "scores": "",
    "sub_feature": []
}

diagnosis12 = {
    "principal": "Chronic inflammation",
    "scores": "",
    "sub_feature": [
        {
            "principal": "type 2 pneumocyte hyperplasia",
            "scores": "",
            "sub_feature": []
        },
        {
            "principal": "organizing fibrosis",
            "scores": "",
            "sub_feature": []
        },
    ]
}

diagnosis13 = {
    "principal": "Metastatic adenocarcinoma",
    "scores": "",
    "sub_feature": [
        {
            "principal": "from colon primary",
            "scores": "",
            "sub_feature": []
        },
    ]
}

diagnosis14 = {
    "principal": "Sclerosing pneumocytoma",
    "scores": "",
    "sub_feature": []
}

diagnosis15 = {
    "principal": "Pulmonary hamartoma",
    "scores": "",
    "sub_feature": []
}

diagnosis16 = {
    "principal": "Large cell neuroendocrine carcinoma",
    "scores": "",
    "sub_feature": []
}

diagnosis17 = {
    "principal": "Malignant lymphoma",
    "scores": "",
    "sub_feature": []
}

diagnosis18 = {
    "principal": "Metastatic carcinoma",
    "scores": "",
    "sub_feature": [
        {
            "principal": "from breast primary",
            "scores": "",
            "sub_feature": []
        },
    ]
}

diagnosis19 = {
    "principal": "Metastatic leiomyoma",
    "scores": "",
    "sub_feature": []
}

diagnosis20 = {
    "principal": "Mucoepidermoid carcinoma",
    "scores": "",
    "sub_feature": []
}

diagnosis21 = {
    "principal": "Pleomorphic carcinoma",
    "scores": "",
    "sub_feature": []
}

DIAGNOSES = [
    diagnosis1, diagnosis2, diagnosis3, diagnosis4, diagnosis5,
    diagnosis6, diagnosis7, diagnosis8, diagnosis9, diagnosis10,
    diagnosis11, diagnosis12, diagnosis13, diagnosis14, diagnosis15,
    diagnosis16, diagnosis17, diagnosis18, diagnosis19, diagnosis20,
    diagnosis21
]

LUNG_TREE = {
    "procedures": PROCEDURES,
    "sub_feature": DIAGNOSES
}