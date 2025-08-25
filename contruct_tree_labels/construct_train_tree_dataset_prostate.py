
import torch, os
import json
import pandas as pd
import random
from transformers import AutoModel
import h5py
from torch import Tensor, einsum, nn
import numpy as np
from pathlib import Path
import re

from procedures import PROCEDURE_DICT
from diagnosis_types_all import DIAGNOSIS_DICT


## 완전하게 recursive 하게는 못할 것 같고 
## 일단 \n 이거 다 지우기 
### , 나 with 로 처음은 나누기
### 그다음은 1) 2) 3) 혹은 1. 2. 3. 으로 나누는 것도 있음.. 

def generate_sample_case1(samp, overall_diag, principal):
    overall_diag = overall_diag.replace("\n", "")
    overall_diag = re.sub(r' {2,}', ' ', overall_diag)
    diag = overall_diag.replace(f"{principal}, ", "").replace(f"{principal} with ", "").replace(f"{principal}", "")

    d_idx = 1
    features = []
    
    samp['sub_feature'].append(
            {
            "principal": principal,
            "scores": "",
            "sub_feature" : []
            }
        )

    if principal == "Acinar adenocarcinoma":
        sub_feature = [{
            "principal": principal,
            "scores": "",
            "sub_feature": []
        }]
        
        diag_list = diag.split(', ')
        ### gleason's score
        gleason = diag_list[0]
        gleason_info = {
            "principal": "Gleanson\'s score",
            "scores": gleason.split(' ')[2],
            "sub_feature": [
                {
                    "principal": "grade group",
                    "scores": diag_list[1].split(' ')[2],
                    "sub_feature": [{
                    "principal": "tumor volume",
                    "scores": diag_list[2].split(' ')[2],
                    "sub_feature": []
                }]
                }
                
            ]
        }
        sub_feature[0]['sub_feature'].append(gleason_info)
        samp["sub_feature"] = sub_feature
    else:
        ## 숫자로 되어 있는지,
        pattern = r'\d+[.)]\s*(.*?)\s*(?=\d+[.)]|$)'
        matches = re.findall(pattern, diag, flags=re.DOTALL)
        features =  [m.replace('\n', ' ').strip() for m in matches]
        
        if len(features) != 0:
        
            for f in features:
                samp[f"sub_feature"][-1]['sub_feature'].append({
                    "principal": f,
                    "scores": "",
                    "sub_features": []
                })
        elif len(diag) != 0:
            samp[f"sub_feature"][-1]['sub_feature'].append({
                "principal": diag,
                "scores": "",
                "sub_features": []
            })

    #samp[f"diag_{d_idx}"]["auxiliary"] = auxiliary
    #amp[f"diag_{d_idx}"]["sub_feature"] = samp

    return samp




if __name__ == '__main__':
    annotation_path = f"../data/train.json"
    train_save_path = f"../data/tree_samples/train_prostate.json"

    path = Path(train_save_path)
    path.parent.mkdir(parents=True, exist_ok=True)


    procedures = PROCEDURE_DICT["prostate"]
    prostate_diags = DIAGNOSIS_DICT["prostate"]

    with open(annotation_path, "r", encoding="utf-8") as f:
        annotations = json.load(f)


    anno_organ = []
    for item in annotations:
        organ = item["report"].split(",")[0]
        
        if organ.lower() == 'prostate':
            anno_organ.append(item)


    train_organ_train = []

    for idx, item in enumerate(anno_organ):

        slide_id = item["id"]
        organ = item["report"].split(",")[0]
        proc = item["report"].split(",")[1].split(";")[0].strip()
        overall_diag =  item["report"].split(";")[1][1:].strip()

        print(idx, slide_id)

        samp = {"id": slide_id,
                "organ": organ,
                "procedure": proc,
                "sub_feature": []
                }

        ## 여기 앞에서 잡으면 될듯
        rows_diag = overall_diag.split("\n")
        
        for row in rows_diag:
            row = re.sub(r'^\s*\d+[.)]\s*', '', row)
            if "Acinar adenocarcinoma" in row:
                samp = generate_sample_case1(samp, row, "Acinar adenocarcinoma")
            else:
                samp = generate_sample_case1(samp, row, row.split(',')[0])



        train_organ_train.append(samp)
        with path.open("w", encoding="utf-8") as f:
            json.dump(train_organ_train, f, ensure_ascii=False, indent=4)
