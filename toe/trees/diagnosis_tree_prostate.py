
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
            "biopsy"
        ]

diagnosis1 = {
      "principal": "Acinar adenocarcinoma,",
      "scores": "",
      "sub_feature": [
        {
          "principal": "Gleason's score",
          "scores": ["10","9","6","7","8"],
          "sub_feature": [
            {
              "principal": "grade group",
              "scores": ["2","4","5","3","1"],
              "sub_feature": [
                {
                  "principal": "",
                  "scores": "",
                  "sub_feature": []
                },
                {
                  "principal": "tumor volume:",
                  "scores": ["75%","40%","55%","7%","50%","100%",
                  "78%","45%","30%","95%","20%","25%","70%","33%",
                  "60%","5%","62%","88%","96%","66%","4%","22%",
                  "57%","14%","1%","90%","15%","3%","83%","65%",
                  "80%","35%","85%","10%"],
                  "sub_feature": []
                }
              ]
            }
          ]
        }
      ]
    }

diagnosis2 = {
      "principal": "Chronic granulomatous inflammation without necrosis",
      "scores": "",
      "sub_feature": []
    }

diagnosis3 = {
      "principal": "No tumor present",
      "scores": "",
      "sub_feature": []
    }

diagnosis4 = {
      "principal": "Chronic granulomatous inflammation with necrosis",
      "scores": "",
      "sub_feature": []
    }

diagnosis5 = {
      "principal": "Acute prostatitis",
      "scores": "",
      "sub_feature": []
    }

diagnosis6 = {
      "principal": "Small cell carcinoma",
      "scores": "",
      "sub_feature": []
    }

diagnosis7 = {
      "principal": "Malignant lymphoma",
      "scores": "",
      "sub_feature": []
    }

DIAGNOSES = [
    diagnosis1, diagnosis2, diagnosis3,  diagnosis4,  diagnosis5,
    diagnosis6, diagnosis7
]

PROSTATE_TREE = {
    "procedures": PROCEDURES,
    "sub_feature": DIAGNOSES
}