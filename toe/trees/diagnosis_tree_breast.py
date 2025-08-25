
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
            "core-needle biopsy",
            "sono-guided core biopsy",
            "mammotome biopsy",
            "encor biopsy",
            "vacuum-assisted biopsy",
            "biopsy",
            "lumpectomy"
        ]

diagnosis1 = {"principal":"Invasive carcinoma of no special type",
        "scores": "",
        "sub_feature": [
        {"principal": ", grade I",
            "scores": "",
            "sub_feature": [
            {
            "principal": "Tubule formation",
            "scores": [1,2,3],
            "sub_feature": []
            },

            {
            "principal": "Nuclear grade",
            "scores": [1,2,3],
            "sub_feature": []
            },

            {
            "principal": "Mitoses",
            "scores": [1,2,3],
            "sub_feature": []
            },
            ]
        },

        {"principal": ", grade II",
            "scores": "",
            "sub_feature": [
            {
            "principal": "Tubule formation",
            "scores": [1,2,3],
            "sub_feature": []
            },

            {
            "principal": "Nuclear grade",
            "scores": [1,2,3],
            "sub_feature": []
            },

            {
            "principal": "Mitoses",
            "scores": [1,2,3],
            "sub_feature": []
            },
            ]
        },

        {"principal": ", grade III",
            "scores": "",
            "sub_feature": [
            {
            "principal": "Tubule formation",
            "scores": [1,2,3],
            "sub_feature": []
            },

            {
            "principal": "Nuclear grade",
            "scores": [1,2,3],
            "sub_feature": []
            },

            {
            "principal": "Mitoses",
            "scores": [1,2,3],
            "sub_feature": []
            },
            ]
        },
        ]
    }

diagnosis2 = {"principal":"Ductal carcinoma in situ",
        "scores": "",
        "sub_feature": [
        {"principal": "",
            "scores": "",
            "sub_feature": [
            {"principal": "Type",
            "scores": "",
            "sub_feature": [
                {"principal": "Cribriform",
                    "scores": "",
                    "sub_feature": []
                },
                {"principal": "Cribriform, Micropapillary",
                    "scores": "",
                    "sub_feature": []
                },
                {"principal": "Flat",
                    "scores": "",
                    "sub_feature": []
                },
                {"principal": "Cribriform and solid",
                    "scores": "",
                    "sub_feature": []
                },
                {"principal": "Solid",
                    "scores": "",
                    "sub_feature": []
                },
                {"principal": "Papillary",
                    "scores": "",
                    "sub_feature": []
                },

                {"principal": "Cribriform and papillary",
                    "scores": "",
                    "sub_feature": []
                },
                {"principal": "Solid and papillary",
                    "scores": "",
                    "sub_feature": []
                },
                {"principal": "Micropapillary",
                    "scores": "",
                    "sub_feature": []
                },
                {"principal": "Cribriform and micropapillary",
                    "scores": "",
                    "sub_feature": []
                },
                {"principal": "Cribriform",
                    "scores": "",
                    "sub_feature": []
                },
                {"principal": "Solid and micropapillary",
                    "scores": "",
                    "sub_feature": []
                },
                {"principal": "Solid and cribriform",
                    "scores": "",
                    "sub_feature": []
                },
                ]
            },

            {"principal": "Nuclear grade",
            "scores": "",
            "sub_feature": [
                {"principal": "Low",
                    "scores": "",
                    "sub_feature": []
                },
                {"principal": "Intermediate",
                    "scores": "",
                    "sub_feature": []
                },
                {"principal": "High",
                    "scores": "",
                    "sub_feature": []
                },
                ]
            },

            {"principal": "Necrosis",
            "scores": "",
            "sub_feature": [
                {"principal": "Present (Comedo-type)",
                    "scores": "",
                    "sub_feature": []
                },
                {"principal": "Present",
                    "scores": "",
                    "sub_feature": []
                },
                {"principal": "Present (Focal)",
                    "scores": "",
                    "sub_feature": []
                },
                {"principal": "Absent",
                    "scores": "",
                    "sub_feature": []
                },
                ]
            },
            ]
        }
        ]
    }

diagnosis3 = {"principal":"Papillary neoplasm",
        "scores": "",
        "sub_feature": [
            {"principal": "",
            "scores": "",
            "sub_feature": []
            },
            {"principal": "with usual ductal hyperplasia",
                "scores": "",
                "sub_feature": []
            },
            {"principal": "with atypical ductal hyperplasia",
                "scores": "",
                "sub_feature": []
            },
            {"principal": "with apocrine metaplasia",
                "scores": "",
                "sub_feature": []
            },
            ]
        }

diagnosis4 = {"principal":"Microcalcification",
        "scores": "",
        "sub_feature": []
        }

diagnosis5 = {"principal":"Fibroepithelial tumor",
        "scores": "",
        "sub_feature": [
            {"principal": "",
            "scores": "",
            "sub_feature": []
            },
            {"principal": ", favor phyllodes tumor",
            "scores": "",
            "sub_feature": []
            },
            {"principal": ", favor fibroadenoma",
            "scores": "",
            "sub_feature": []
            },
            ]
        }

diagnosis6 = {"principal":"Invasive lobular carcinoma",
        "scores": "",
        "sub_feature": [
            {"principal": "",
            "scores": "",
            "sub_feature": []
            },
            {"principal": ", pleomorphic type",
            "scores": "",
            "sub_feature": []
            },
            ]
        }

diagnosis7 = {"principal":"Mucinous carcinoma",
        "scores": "",
        "sub_feature": []
        }

diagnosis8 = {"principal":"Intraductal papilloma",
        "scores": "",
        "sub_feature": [
            {"principal": "",
            "scores": "",
            "sub_feature": []
            },
            {"principal": "with usual ductal hyperplasia",
            "scores": "",
            "sub_feature": []
            },
            ]
        }

diagnosis9 = {"principal": "Usual ductal hyperplasia",
         "scores": "",
         "sub_feature": []}

diagnosis10 = {"principal": "Atypical ductal hyperplasia",
          "scores": "",
          "sub_feature": []}

diagnosis11 = {"principal": "Lobular carcinoma in situ",
          "scores": "",
          "sub_feature": []}

diagnosis12 = {"principal": "Fibroadenomatoid change",
           "scores": "",
           "sub_feature": []}

diagnosis13 = {"principal": "Fibroadenoma",
           "scores": "",
           "sub_feature": []}

diagnosis14 = {"principal": "Invasive carcinoma with features of mucinous carcinoma",
           "scores": "",
           "sub_feature": []}

diagnosis15 = {"principal": "Micro-invasive carcinoma",
           "scores": "",
           "sub_feature": []}

diagnosis16 = {"principal": "Metaplastic carcinoma",
           "scores": "",
           "sub_feature": []}

diagnosis17 = {"principal": "Tubular carcinoma",
           "scores": "",
           "sub_feature": []}

diagnosis18 = {"principal": "Fibroepithelial lesion, favor fibroadenoma",
           "scores": "",
           "sub_feature": []}

diagnosis19 = {"principal": "Columnar cell lesion",                            "scores": "", "sub_feature": []}  # n=45
diagnosis20 = {"principal": "Fibrocystic change",                              "scores": "", "sub_feature": []}  # n=35
diagnosis21 = {"principal": "No evidence of tumor",                            "scores": "", "sub_feature": []}  # n=34
diagnosis22 = {"principal": "Fibroepithelial tumor, favor fibroadenoma",       "scores": "", "sub_feature": []}  # n=30
diagnosis23 = {"principal": "Sclerosing adenosis",                             "scores": "", "sub_feature": []}  # n=25
diagnosis24 = {"principal": "Apocrine metaplasia",                             "scores": "", "sub_feature": []}  # n=25
diagnosis25 = {"principal": "Intraductal papilloma with usual ductal hyperplasia","scores": "", "sub_feature": []}  # n=20
diagnosis26 = {"principal": "Duct ectasia",                                    "scores": "", "sub_feature": []}  # n=17
diagnosis27 = {"principal": "Pseudoangiomatous stromal hyperplasia",           "scores": "", "sub_feature": []}  # n=9
diagnosis28 = {"principal": "Invasive micropapillary carcinoma",               "scores": "", "sub_feature": []}  # n=8
diagnosis29 = {"principal": "Columnar cell lesion with atypia",                "scores": "", "sub_feature": []}  # n=7
diagnosis30 = {"principal": "Mammary duct ectasia",                            "scores": "", "sub_feature": []}  # n=6
diagnosis31 = {"principal": "Invasive cribriform carcinoma",                   "scores": "", "sub_feature": []}  # n=2
diagnosis32 = {"principal": "Papillary carcinoma",                             "scores": "", "sub_feature": []}  # n=2
diagnosis33 = {"principal": "Solid papillary carcinoma in situ",               "scores": "", "sub_feature": []}  # n=2
diagnosis34 = {"principal": "Fat necrosis",                                    "scores": "", "sub_feature": []}  # n=2
diagnosis35 = {"principal": "Radiation-related atypia",                        "scores": "", "sub_feature": []}  # n=2
diagnosis36 = {"principal": "Atypical lobular hyperplasia",                    "scores": "", "sub_feature": []}  # n=1
diagnosis37 = {"principal": "Intravascular tumor emboli",                      "scores": "", "sub_feature": []}  # n=1
diagnosis38 = {"principal": "Flat epithelial atypia",                          "scores": "", "sub_feature": []}  # n=1
diagnosis39 = {"principal": "Mucocele-like lesion",                            "scores": "", "sub_feature": []}  # n=1
diagnosis40 = {"principal": "Invasive lobular carcinoma, pleomorphic type",    "scores": "", "sub_feature": []}  # n=1
diagnosis44 = {"principal": "Intraductal papilloma with apocrine metaplasia",  "scores": "", "sub_feature": []}  # n=1
diagnosis45 = {"principal": "Carcinoma with apocrine differentiation",         "scores": "", "sub_feature": []}  # n=1
diagnosis46 = {"principal": "Columnar cell hyperplasia",                       "scores": "", "sub_feature": []}  # n=1
diagnosis47 = {"principal": "Solid papillary carcinoma",                       "scores": "", "sub_feature": []}  # n=1
diagnosis48 = {"principal": "Encapsulated papillary carcinoma",                "scores": "", "sub_feature": []}  # n=1
diagnosis49 = {"principal": "Secretory carcinoma",                             "scores": "", "sub_feature": []}  # n=1
diagnosis50 = {"principal": "Phyllodes tumor",                                 "scores": "", "sub_feature": []}  # n=1
diagnosis51 = {"principal": "Columnar cell change",                            "scores": "", "sub_feature": []}  # n=1
diagnosis52 = {"principal": "Sclerosing papilloma",                            "scores": "", "sub_feature": []}  # n=1
diagnosis53 = {"principal": "Foreign body reaction",                           "scores": "", "sub_feature": []}  # n=6
diagnosis54 = {"principal": "Apocrine adenosis",                           "scores": "", "sub_feature": []}  # 
diagnosis55 = {"principal": "Desmoid fibromatosis",                           "scores": "", "sub_feature": []}  # 
diagnosis56 = {"principal": "Ductal carcinoma in situ in intraductal papilloma", "scores": "", "sub_feature": []}  # 
diagnosis57 = {"principal": "Fibroepithelial tumor, favor phyllodes tumor", "scores": "", "sub_feature": []}  # 
diagnosis58 = {"principal": "Granulomatous lobular mastitis", "scores": "", "sub_feature": []}  # 
diagnosis59 = {"principal": "Malignant lymphoma", "scores": "", "sub_feature": []}  # 

DIAGNOSES = [diagnosis1, diagnosis2, diagnosis3, diagnosis4, diagnosis5, diagnosis6,
             diagnosis7, diagnosis8, diagnosis9, diagnosis10, diagnosis11, diagnosis12,
             diagnosis13, diagnosis14, diagnosis15, diagnosis16, diagnosis17, diagnosis18,
             diagnosis19, diagnosis20, diagnosis21, diagnosis22, diagnosis23, diagnosis24,
             diagnosis25, diagnosis26, diagnosis27, diagnosis28, diagnosis29, diagnosis30,
             diagnosis31, diagnosis32, diagnosis33, diagnosis34, diagnosis35, diagnosis36,
             diagnosis37, diagnosis38, diagnosis39, diagnosis40,
             diagnosis44, diagnosis45, diagnosis46, diagnosis47, diagnosis48,
             diagnosis49, diagnosis50, diagnosis51, diagnosis52, diagnosis53, diagnosis54,
             diagnosis55, diagnosis56, diagnosis57, diagnosis58, diagnosis59]


BREAST_TREE={
        "procedures" : PROCEDURES,
        "sub_feature" : DIAGNOSES
        }

