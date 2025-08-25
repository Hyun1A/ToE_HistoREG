
import torch, os
import json
import pandas as pd
import random
from transformers import AutoModel
import h5py
from torch import Tensor, einsum, nn
import numpy as np


# diagnosis tree 구성의 경우의 수:
# 1) level 3&4 내용이 붙는지?
# 2) 있어도 한 케이스밖에 없는지?
# 3) 여러 개라면 listup 및 classification으로 처리할 수 있는지?
# 4) 혹은 report generation으로 처리해야할지?



PROCEDURES=[
            "transurethral resection"
        ]

diagnosis1 = {
      "principal": "No tumor present",
      "scores":"",
      "sub_feature": []
    }

diagnosis2 = {
      "principal": "Chronic granulomatous inflammation with foreign body reaction",
      "scores":"",
      "sub_feature": []
    }

diagnosis3 = {
      "principal": "Note) The specimen includes muscle proper.",
      "scores":"",
      "sub_feature": []
    }

diagnosis4 = {
      "principal": "Note) The specimen does not include muscle proper.",
      "scores":"",
      "sub_feature": []
    }

diagnosis5 = {
      "principal": "Invasive urothelial carcinoma,",
      "scores":"",
      "sub_feature": [
        {
          "principal": "with involvement of subepithelial connective tissue",
          "scores":"",
          "sub_feature": []
        },
        {
          "principal": "squamous differentiation",
          "scores":"",
          "sub_feature": []
        },
        {
          "principal": "with involvement of muscle proper",
          "scores":"",
          "sub_feature": []
        }
      ]
    }

diagnosis6 = {
      "principal": "Urothelial carcinoma in situ",
      "scores":"",
      "sub_feature": []
    }

diagnosis7 = {
      "principal": "Non-invasive papillary urothelial carcinoma,",
      "scores":"",
      "sub_feature": [
        {
          "principal": "low grade",
          "scores":"",
          "sub_feature": []
        },
        {
          "principal": "high grade",
          "scores":"",
          "sub_feature": []
        },
        {
          "principal": "high grade, with squamous differentiation",
          "scores":"",
          "sub_feature": []
        }
      ]
    }

DIAGNOSES = [
    diagnosis3,  diagnosis4,  diagnosis5,
    diagnosis6, diagnosis7, diagnosis1, diagnosis2
]

BLADDER_TREE = {
    "procedures": PROCEDURES,
    "sub_feature": DIAGNOSES
}