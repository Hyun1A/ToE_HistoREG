
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



def generate_sample_case1(samp, overall_diag, breast_diags):
    diags = overall_diag.split("\n")

    d_idx = 1
    diag = diags[0]
    principal = "Invasive carcinoma of no special type"

    samp.update({
        f"diag_{d_idx}": {
            "principal": principal,
            "auxiliary": "",
            "scores": "",
            }
        })

    additional = diag.replace(principal, " ")
    auxiliary = additional.split(",")[1].split("(")[0].strip()
    scores = additional.split("(")[1][:-1].strip()
    scores = scores.strip().split(",")

    scores_dict = {}
    for sc in scores:
        bio, grade = sc.strip().split(":")
        scores_dict[bio.strip()] = grade.strip()

    samp[f"diag_{d_idx}"]["auxiliary"] = auxiliary
    samp[f"diag_{d_idx}"]["scores"] = scores_dict


    d_idx+=1
    for diag in diags[1:]:
        diag = diag.strip()[2:].strip()

        principal, auxiliary, scores = "", "", ""
        for d_case in breast_diags:
            if diag.startswith(d_case):
                principal = d_case
                break
        
        if principal != "":
            auxiliary = diag.replace(principal, "")
            
            samp.update({
                f"diag_{d_idx}": {
                    "principal": principal,
                    "auxiliary": auxiliary,
                    "scores": "",
                    }
                })

            d_idx+=1

    return samp


def generate_sample_case2(samp, overall_diag, breast_diags):
    diags = [s.replace("\n", "").strip() for s in re.split(r'(?m)^\s*\d+\.\s*', overall_diag) if s.strip()]

    d_idx = 1
    diag = diags[0]
    principal = "Ductal carcinoma in situ"

    samp.update({
        f"diag_{d_idx}": {
            "principal": principal,
            "auxiliary": "",
            "scores": "",
            }
        })

    additional = diag.replace(principal, " ")
    auxiliary = additional.split("-")[0].strip()

    scores = [s.strip() for s in additional.split(" -")[1:]]

    scores_dict = {}
    for sc in scores:
        bio, grade = sc.strip().split(":")
        
        scores_dict[bio.strip()] = grade.strip()

    samp[f"diag_{d_idx}"]["auxiliary"] = auxiliary
    samp[f"diag_{d_idx}"]["scores"] = scores_dict


    d_idx+=1
    for diag in diags[1:]:
        diag = diag.strip()

        principal, auxiliary, scores = "", "", ""
        for d_case in breast_diags:
            if diag.startswith(d_case):
                principal = d_case
                break
        
        if principal != "":
            auxiliary = diag.replace(principal, "")
            
            samp.update({
                f"diag_{d_idx}": {
                    "principal": principal,
                    "auxiliary": auxiliary,
                    "scores": "",
                    }
                })

            d_idx+=1

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
    train_save_path = f"../data/tree_samples/train_breast.json"

    path = Path(train_save_path)
    path.parent.mkdir(parents=True, exist_ok=True)


    procedures = PROCEDURE_DICT["breast"]
    breast_diags = DIAGNOSIS_DICT["breast"]

    with open(annotation_path, "r", encoding="utf-8") as f:
        annotations = json.load(f)


    anno_organ = []
    for item in annotations:
        organ = item["report"].split(",")[0]
        
        if organ.lower() == 'breast':
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

        if "Invasive carcinoma of no special type" in overall_diag:
            samp = generate_sample_case1(samp, overall_diag, breast_diags)

        elif "Ductal carcinoma in situ" in overall_diag.split("\n")[0]:
            samp = generate_sample_case2(samp, overall_diag, breast_diags)

        else:
            samp = generate_sample_case3(samp, overall_diag, breast_diags)



        train_organ_train.append(samp)
        with path.open("w", encoding="utf-8") as f:
            json.dump(train_organ_train, f, ensure_ascii=False, indent=4)
