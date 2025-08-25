import os
from typing import List, Dict, Any
import torch
from torch.utils.data import Dataset
import json
import pandas as pd
import random
import h5py
import numpy as np
from tqdm.auto import tqdm


ORGANS = ['Breast', 'Lung', 'Colon', 'Rectum', \
          'Stomach', 'Uterine cervix', 'Prostate', 'Urinary bladder']


class HierDatasetNoisyMixup(Dataset):
    """
    Minimal skeleton.
    Expect each sample to have:
      - 'feat': 1D array-like of shape [D]
      - (optional) 'organ', 'procedure', 'top_diagnoses', 'diagnosis_node_map'
    """
    def __init__(self, annotation_path, val_anno_path, raw_feature_path, struct_reports_path, feature_fn, model, mode="train", alpha=0.2, lam=0.01):
        self.raw_feature_path = raw_feature_path
        self.feature_fn = feature_fn
        self.fns = os.listdir(raw_feature_path)
        self.report_model = model
        self.alpha = alpha
        self.lam = lam

        with open(annotation_path, "r", encoding="utf-8") as f:
            self.annotations = json.load(f)    
        with open(val_anno_path, "r", encoding="utf-8") as f:
            self.val_annotations = json.load(f)    

        self.annotations = [item for item in self.annotations \
                            if item["report"].split(",")[0].lower() in [item.lower() for item in ORGANS]]
        self.val_annotations = [item for item in self.val_annotations \
                            if item["report"].split(",")[0].lower() in [item.lower() for item in ORGANS]] 

        self.true_reports = {item["id"]:item["report"] for item in self.annotations}          
        self.val_true_reports = {item["id"]:item["report"] for item in self.val_annotations}  


        # anno_data = [item for item in os.listdir(struct_reports_path) if ".json" in item]
        anno_fns = ['train_breast.json', 'train_lung.json', 'train_colon.json', 'train_rectum.json', \
                    'train_stomach.json', 'train_cervix.json', 'train_prostate.json', 'train_bladder.json']
    
        self.struct_true_reports = dict()
        for anno_fn in anno_fns:
            struct_reps = {item["id"]: item for item in json.load(open(f"{struct_reports_path}/{anno_fn}"))}
            self.struct_true_reports.update(struct_reps)

        self.features_all = self.prepare_features()

        self.ids = []
        self.image_features = []
        self.text_features = []
        self.struct_reports = []
        self.reports = []
        
        for key, val in self.features_all.items():
            struct_report = self.struct_true_reports[key]
            report = self.true_reports[key]

            if (mode == "train") and not (key in self.val_true_reports.keys()):
                self.ids.append(key)
                self.image_features.append(val['image'])
                self.text_features.append(val['text'])
                self.struct_reports.append(struct_report)
                self.reports.append(report)

                
            elif (mode == "val") and (key in self.val_true_reports.keys()):
                self.ids.append(key)
                self.image_features.append(val['image'])
                self.text_features.append(val['text'])
                self.struct_reports.append(struct_report)
                self.reports.append(report)


    def __len__(self) -> int:
        return len(self.ids)

    def __getitem__(self, idx1: int):
        idx2 = torch.randint(len(self.ids), (1,))

        idx_list = [idx1, idx2.item()]

        samples = []

        for idx in idx_list:
            case_id = self.ids[idx]
            image_feature = self.image_features[idx]
            text_feature = self.text_features[idx]
            struct_report = self.struct_reports[idx]
            report = self.reports[idx]
                
            noise_img = self.lam * torch.randn_like(image_feature) / (image_feature.norm()*image_feature.size(1)**0.5)
            image_feature = image_feature + noise_img

            noise_txt = self.lam * torch.randn_like(text_feature) / (text_feature.norm()*text_feature.size(1)**0.5)
            text_feature = text_feature + noise_txt

            samples.append( {"case_id": case_id, \
                    "image_feat": image_feature, "text_feat": text_feature, \
                    "struct_report": struct_report, "report": report} )


        ratio = torch.distributions.Beta(self.alpha, self.alpha).sample().item() if self.alpha > 0 else 1


        samples_mixed = dict()
        for key in samples[0].keys():
            if key == "image_feat" or key == "text_feat":
                samples_mixed[key] = ratio*samples[0][key] + (1-ratio)*samples[1][key]

        return samples[0], samples[1], samples_mixed, ratio


    def prepare_features(self):
        if os.path.isfile(self.feature_fn):
            features = torch.load(self.feature_fn)
        else:
            features = {}

            pbar = tqdm(self.annotations, desc="extract", leave=False)
            for idx, report in enumerate(pbar, start=1):
                case_id = report['id']
                fn = case_id.split(".")[0]+".pt"

                if not fn in self.fns:
                    continue

                if fn.split(".")[1] == "pt":
                    embedding = torch.load(f"{self.raw_feature_path}/{fn}")
                elif fn.split(".")[1] == "h5":
                    embedding = torch.tensor(np.array(h5py.File(f"{self.raw_feature_path}/{fn}")["features"]))

                tile_embeddings = embedding.unsqueeze(0).to('cuda')

                # Compute slide embedding and latents. Only Perceiver is evaluated.
                # We highly recommend running the model on a GPU in mixed precision (`fp16`)
                with torch.autocast('cuda', torch.float16), torch.inference_mode():
                    reprs = self.report_model.slide_representations(tile_embeddings)

                # Do zero-shot prediction using the slide embedding.
                with torch.autocast('cuda', torch.float16):
                    image_proj, text_proj = self.report_model.get_proj_embs(reprs['image_embedding'], prompts=ORGANS)

                feature = {"image": image_proj.detach().cpu(), "text": text_proj.detach().cpu()}
                features[case_id] = feature

                if idx % 500 == 0:
                    torch.save(features, self.feature_fn)

            torch.save(features, self.feature_fn)
        return features



def collate_hier_mixup(batch):
    
    stacked_samples = []

    for s_idx in range(len(batch[0])-1):
        image_feats = torch.stack([b[s_idx]["image_feat"] for b in batch], dim=0)
        text_feats = torch.stack([b[s_idx]["text_feat"] for b in batch], dim=0)

        if s_idx < 2:
            ids = [b[s_idx]["case_id"] for b in batch]
            struct_reps = [b[s_idx]["struct_report"] for b in batch]
            reps = [b[s_idx]["report"] for b in batch]

            stacked_samples.append({"case_id": ids, \
                "image_feat": image_feats, "text_feat": text_feats, \
                "struct_report": struct_reps, "report": reps})

        else:
            stacked_samples.append({"image_feat": image_feats, \
                                     "text_feat": text_feats})


    ratios = [b[-1] for b in batch]

    return stacked_samples[0], stacked_samples[1], stacked_samples[2], ratios

