
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



def generate_sample_case1(samp, overall_diag, principal):
    overall_diag = overall_diag.replace("\n", "")
    overall_diag = re.sub(r' {2,}', ' ', overall_diag)
    diag = overall_diag.replace(f"{principal}, ", "").replace(f"{principal} with ", "").replace(f"{principal}", "")

    d_idx = 1
    features = []
    
    samp.update({
        f"sub_feature": [{
            "principal": principal,
            "scores": "",
            "sub_feature" : []
            }]
        })

    if principal == "Adenocarcinoma":
        sub_feature = [{
            "principal": diag.split(' with ')[0],
            "scores": "",
            "sub_feature": []
        }]
        
        if len(diag.split(' with ')) > 1:
            features = diag.split(' with ')[1].split(' and ')            
            sub_feature[0]["sub_feature"] = [{"principal": f, "scores":"", "sub_features": []} for f in features]
        
        samp["sub_feature"][0]["sub_feature"] = sub_feature
    else:
        pattern = r'\d+[.)]\s*(.*?)\s*(?=\d+[.)]|$)'
        matches = re.findall(pattern, diag, flags=re.DOTALL)
        features =  [m.replace('\n', ' ').strip() for m in matches]
        
        if len(features) != 0:
        
            for f in features:
                samp[f"sub_feature"][0]['sub_feature'].append({
                    "principal": f,
                    "scores": "",
                    "sub_features": []
                })
        elif len(diag) != 0:
            samp[f"sub_feature"][0]['sub_feature'].append({
                "principal": diag,
                "scores": "",
                "sub_features": []
            })

    #samp[f"diag_{d_idx}"]["auxiliary"] = auxiliary
    #amp[f"diag_{d_idx}"]["sub_feature"] = samp

    return samp


def generate_sample_case2(samp, overall_diag, breast_diags):
    diags = overall_diag.split(", ")

    d_idx = 1
    diag = diags[1]
    principal = "Poorly cohesive carcinoma"
    features = []
    
    samp.update({
        f"diag_{d_idx}": {
            "principal": principal,
            "auxiliary": "",
            "scores": "",
            "features" : []
            }
        })

    auxiliary = diag.split(' with ')[0]
    if len(diag.split(' with ')) > 1:
        features = diag.split(' with ')[1].split(' and ')

    samp[f"diag_{d_idx}"]["auxiliary"] = auxiliary
    samp[f"diag_{d_idx}"]["features"] = features

    return samp


def generate_sample_case3(samp, overall_diag, breast_diags):
    diags = [s.replace("\n", "").strip() for s in re.split(r'(?m)^\s*\d+\.\s*', overall_diag) if s.strip()]

    d_idx = 1
    for diag in diags:
        diag = diag.strip()

        principal, auxiliary = "", ""
        for d_case in breast_diags:
            if diag.startswith(d_case):
                principal = d_case
                break
        
        if principal != "":
            auxiliary = diag.replace(principal, "").strip()
            
            samp.update({
                f"diag_{d_idx}": {
                    "principal": principal,
                    "auxiliary": auxiliary,
                    "scores": "",
                    }
                })
            
            print(f"{idx}: ", overall_diag)
            print(diag)
            print(principal)
            print(auxiliary)
            print()            

            d_idx+=1

    return samp








if __name__ == '__main__':
    annotation_path = f"../data/train.json"
    train_save_path = f"../data/tree_samples/train_prostate.json"

    path = Path(train_save_path)
    path.parent.mkdir(parents=True, exist_ok=True)


    procedures = PROCEDURE_DICT["stomach"]
    stomach_diags = DIAGNOSIS_DICT["stomach"]

    with open(annotation_path, "r", encoding="utf-8") as f:
        annotations = json.load(f)


    anno_organ = []
    for item in annotations:
        organ = item["report"].split(",")[0]
        
        if organ.lower() == 'stomach':
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
                "procedure": proc
                }

        if "Adenocarcinoma" in overall_diag:
            samp = generate_sample_case1(samp, overall_diag, "Adenocarcinoma")
            
        elif "Poorly cohesive carcinoma" in overall_diag: # 만성 + 급성 염증 세포 침식 (이걸 따로 해야하나 고민)
            samp = generate_sample_case1(samp, overall_diag, "Poorly cohesive carcinoma")
            
        elif "Chronic gastritis" in overall_diag.split("\n")[0]: # 만성적 염증세포
            samp = generate_sample_case1(samp, overall_diag, "Chronic gastritis")
            
        elif "Chronic active gastritis" in overall_diag.split("\n")[0]: # 만성 + 급성 염증 세포 침식 (이걸 따로 해야하나 고민)
            samp = generate_sample_case1(samp, overall_diag, "Chronic active gastritis")
        
        elif "Tubular adenoma with" in overall_diag.split("\n")[0]: # 만성 + 급성 염증 세포 침식 (이걸 따로 해야하나 고민)
            overall_diag = overall_diag.replace(" with ", ", ")
            samp = generate_sample_case1(samp, overall_diag, "Tubular adenoma")
        else:
            samp = generate_sample_case1(samp, overall_diag, overall_diag.split(',')[0])



        train_organ_train.append(samp)
        with path.open("w", encoding="utf-8") as f:
            json.dump(train_organ_train, f, ensure_ascii=False, indent=4)
