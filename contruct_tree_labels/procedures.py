
import torch, os
import json
import pandas as pd
import random
from transformers import AutoModel
import h5py
from torch import Tensor, einsum, nn
import numpy as np


PROCEDURE_DICT = {
    "breast": [
        "core-needle biopsy",
        "sono-guided core biopsy",
        "mammotome biopsy",
        "encor biopsy",
        "vacuum-assisted biopsy",
        "biopsy",
        "lumpectomy"
    ],
    "urinary bladder": [
        "transurethral resection",
        "punch biopsy",
        "Loop Electrosurgical Excision Procedure",
        "polypectomy biopsy",
        "frozen biopsy",
        "incisional biopsy"
    ],
    "bladder": [
        "transurethral resection",
        "punch biopsy",
        "Loop Electrosurgical Excision Procedure",
        "polypectomy biopsy",
        "frozen biopsy",
        "incisional biopsy"
    ],
    "cervix": [
        "colposcopic biopsy"
    ],
    "uterine cervix": [
        "colposcopic biopsy"
    ],    
    "colon": [
        "colonoscopic biopsy",
        "colonoscopic polypectomy",
        "colonoscopic mucosal resection",
        "colonoscopic submucosal dissection"
    ],
    "rectum": [
        "colonoscopic biopsy",
        "colonoscopic polypectomy",
        "colonoscopic mucosal resection",
        "colonoscopic submucosal dissection"
    ],
    "lung": [
        "biopsy",
        "segmentectomy"
    ],
    "prostate": [
        "biopsy"
    ],
    "stomach": [
        "endoscopic biopsy",
        "endoscopic mucosal resection",
        "endoscopic submucosal dissection",
        "biopsy"
    ]
}
