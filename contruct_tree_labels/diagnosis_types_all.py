
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



DIAGNOSIS_DICT = {
    "breast": [
        "Invasive carcinoma of no special type",    
        "Ductal carcinoma in situ",
        "Papillary neoplasm",
        "Microcalcification",
        "Fibroepithelial tumor",                                    # 92
        "Invasive lobular carcinoma",                               # 78
        "Columnar cell lesion",
        "Mucinous carcinoma",                                       # 40
        "Intraductal papilloma",                                    # 40
        "Usual ductal hyperplasia",                                 # 37  
        "Fibrocystic change",      
        "No evidence of tumor",
        "Atypical ductal hyperplasia",                              # 34
        "Lobular carcinoma in situ",                                # 21
        "Fibroadenomatoid change",                                  # 15
        "Apocrine metaplasia",
        "Fibroadenoma",                                             # 23
        "Duct ectasia",
        "Invasive carcinoma with features of mucinous carcinoma",   # 10
        "Micro-invasive carcinoma",                                 # 6
        "Flat epithelial atypia",
        "Metaplastic carcinoma",                                    # 4
        "Tubular carcinoma",                                        # 2
        "Fibroepithelial lesion, favor fibroadenoma",               # 1
        "Granulomatous lobular mastitis",
    ],

    "urinary bladder": [
        "No tumor present",                                     
        "Invasive urothelial carcinoma",                        
        "Non-invasive papillary urothelial carcinoma",         
        "Urothelial carcinoma in situ",                        
        "Chronic granulomatous inflammation with foreign body reaction", 
    ],

    "uterine cervix": [
        "Low-grade squamous intraepithelial lesion (LSIL)",   
        "High-grade squamous intraepithelial lesion (HSIL)",  
        "Chronic nonspecific cervicitis",                     
        "Invasive squamous cell carcinoma (SCC)",             
        "Endocervical adenocarcinoma in situ (AIS)",          # 22
        "Adenosquamous carcinoma",                            # 2
        "Metastatic high grade serous carcinoma",             # 1
        "Endocervical polyp"
    ],

    "colon": [
        "Adenocarcinoma",                                        
        "Tubular adenoma",                                       
        "Hyperplastic polyp",                                    
        "Chronic active colitis",                                
        "Chronic nonspecific inflammation",                      
        "Tubulovillous adenoma",                                 # 15
        "Mucinous carcinoma",                                    # 8
        "Sessile serrated lesion",                               # 4
        "Traditional serrated adenoma",                          # 3
        "Small cell carcinoma",                                  # 3
        "Neuroendocrine tumor",                                  # 2
        "Signet-ring cell carcinoma",                            # 2
        "Serrated serrated lesion with low grade dysplasia",     # 1
    ],

    "rectum": [
        "Adenocarcinoma",                                        
        "Tubular adenoma",                                       
        "Chronic nonspecific inflammation",                      
        "Neuroendocrine tumor",                                  # 10
        "Squamous cell carcinoma",                               # 5
        "Hyperplastic polyp",                                    # 5
        "Extranodal marginal zone B cell lymphoma",              # 2
        "Tubulovillous adenoma with high grade dysplasia",       # 2
        "Traditional serrated adenoma with low grade dysplasia", # 1
        "Serrated serrated lesion with low grade dysplasia",     # 1
        "Signet-ring cell carcinoma",                            # 1
    ],

    "lung": [
        "Adenocarcinoma",                                      
        "Squamous cell carcinoma",                             
        "Non-small cell carcinoma",                            
        "Small cell carcinoma",    
        "No evidence of malignancy or granuloma",              # 69
        "Chronic granulomatous inflammation with necrosis",    # 29
        "Fungal infection",                                    # 25
        "Chronic inflammation",                                # 22
        "Invasive mucinous adenocarcinoma",                    # 21
        "Metastatic adenocarcinoma, from colon primary",       # 13
        "Carcinoid/neuroendocrine tumor, NOS",                 # 5
        "Fungal ball",                                         # 4
        "Sclerosing pneumocytoma",                             # 3
        "Large cell neuroendocrine carcinoma",                 # 1
    ],

    "prostate": [
        "Acinar adenocarcinoma",
        "No tumor present",
        "Chronic granulomatous inflammation without necrosis",
        "Acute prostatitis",
    ],

    "stomach": [
        "Adenocarcinoma",                                 # 경우의수 많음 
        "Tubular adenoma with low grade dysplasia",    
        "Chronic gastritis",                                        # 경우의 수 많음
        "Extranodal marginal zone B cell lymphoma of MALT type",    # 95
        "Chronic active gastritis",                                 # 79    # 경우의 수 많음               
        "Malignant lymphoma",                                       # 25               
        "Poorly cohesive carcinoma",                                # 26    # 2종류                         
        "Gastrointestinal stromal tumor",                           # 10
        "Small cell carcinoma",                                     # 7
        "Malignant melanoma",                                       # 6
        "Squamous cell carcinoma",                                  # 5
        "Amyloidosis",                                              # 4
        "Neuroendocrine tumor",                                     # 3
        "Fundic gland polyp",                                       # 3
        "Hyperplastic polyp",                                       # 3
        "Mucinous adenocarcinoma",                                  # 2
    ]
}
