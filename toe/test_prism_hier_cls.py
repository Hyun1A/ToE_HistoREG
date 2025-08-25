import torch, os
import argparse
import json
import pandas as pd
import random
from transformers import AutoModel
import h5py
from torch import Tensor, einsum, nn
from torch.utils.data import DataLoader
import numpy as np
from tqdm.auto import tqdm, trange

from models.hier_classifiers import build_schema, extract_procedures_from_json
from models.hier_classifiers import HierarchicalClassifier, HierLoss, HierLossMixup
from models.hier_classifiers import build_targets
from utils.optimizer import *
from utils.formatting import FORMAT_REGISTRY
from utils.dataset import HierDatasetTest, collate_hier_test

from trees.diagnosis_tree_breast import BREAST_TREE
from trees.diagnosis_tree_lung import LUNG_TREE
from trees.diagnosis_tree_colon import COLON_TREE
from trees.diagnosis_tree_rectum import RECTUM_TREE
from trees.diagnosis_tree_stomach import STOMACH_TREE
from trees.diagnosis_tree_cervix import CERVIX_TREE
from trees.diagnosis_tree_prostate import PROSTATE_TREE
from trees.diagnosis_tree_bladder import BLADDER_TREE
from reg.metric.eval import REG_Evaluator


ORGANS = ['Breast', 'Lung', 'Colon', 'Rectum', \
          'Stomach', 'Uterine cervix', 'Prostate', 'Urinary bladder']

TREES = [BREAST_TREE, LUNG_TREE, COLON_TREE, RECTUM_TREE, \
          STOMACH_TREE, CERVIX_TREE, PROSTATE_TREE, BLADDER_TREE]


os.environ["TOKENIZERS_PARALLELISM"] = "false"

def get_scores(gt_data, pred_data):
    print("Computing scores...")

    EMBEDDING_MODEL = 'dmis-lab/biobert-v1.1'
    SPACY_MODEL = 'en_core_sci_lg'

    evaluator = REG_Evaluator(embedding_model=EMBEDDING_MODEL, spacy_model=SPACY_MODEL)

    pred_dict = {item['id']: item['report'] for item in pred_data}

    eval_pairs = []
    for item in gt_data:
        gt_id = item['id']
        gt_report = item['report']
        
        pred_report = pred_dict.get(gt_id, "") 
        
        eval_pairs.append((gt_report, pred_report))

    scores = evaluator.evaluate_compwise(eval_pairs)

    print(f"\n Average Ranking Score: {scores[0]:.4f}")
    print(f"Average ROUGE Score: {scores[1]:.4f}")
    print(f"Average BLEU Score: {scores[2]:.4f}")
    print(f"Average KEY Score: {scores[3]:.4f}")
    print(f"Average EMB Score: {scores[4]:.4f}")

    return scores[0]



def test(val_loader, classifier, criterion, threshold):
    classifier.eval()

    print(f"Test")
    
    pbar = tqdm(val_loader, desc="test", leave=False)

    case_ids_all = []
    organs_all = []
    gen_report_all = []

    with torch.no_grad():
        for idx, data in enumerate(pbar, start=1):

            case_id = data["case_id"]
            image_proj = data["image_feat"][:,0,:].to("cuda")
            
            outs = classifier(image_proj)

            preds = classifier.decode(outs, threshold=threshold)

            organs = [pred["organ"] for pred in preds]
            gen_report = []
            for pred in preds:
                pred_organ = pred['organ'].lower()
                rep = FORMAT_REGISTRY[pred_organ](pred)
                gen_report.append(rep)

            case_ids_all += case_id
            organs_all += organs
            gen_report_all += gen_report
            
    eval_results = {"case_ids": case_ids_all, \
            "organs": organs_all, \
            "gen_reports":gen_report_all \
            }
    
    pred_data = [{"id":case_id, "report": report} for (case_id, report) in zip(eval_results["case_ids"], eval_results["gen_reports"])]

    df = pd.DataFrame(eval_results)
    os.makedirs("./results/tmp", exist_ok=True)
    test_output_csv = f"./results/tmp/valid_outputs.csv"
    df.to_csv(test_output_csv, index=False)

    return pred_data



if __name__ == "__main__":

    parser = argparse.ArgumentParser()

    parser.add_argument("--aggregator_path", type=str,
                        default="../Prism")
    parser.add_argument("--raw_feature_path", type=str, 
                        default="/data/REG_TEST/test_data/features_virchow")
    parser.add_argument("--feature_fn", type=str, 
                        default="/home2/hyun/REG/PRISM/proposed/exp_hier_cls_base/data/feature_cache/data/features_all_test_phase1.pt")
    parser.add_argument("--test_output_path", type=str, 
                        default="./results/exp_final_test")
    parser.add_argument("--ckpt_path", type=str, 
                        default="./ckpt/ckpt_final.pt")
    parser.add_argument("--probe_type", type=str, default="linear")  
    parser.add_argument("--threshold", type=float, default=0.5)



    args = parser.parse_args()
        



    ################################################
    ####### prepare PRISM and hier cls model #######
    ## Load PRISM model.
    model = AutoModel.from_pretrained(args.aggregator_path, trust_remote_code=True, local_files_only=True)
    model = model.to('cuda')
    model.eval()

    ## construct schema and model
    # ORGANS = ['Breast', '']


    tree_dict = {organ:tree for (organ, tree) in zip(ORGANS, TREES)}

    # observed = extract_procedures_from_json(args.records_path)
    schema = build_schema(tree_dict, ORGANS)

    classifier = HierarchicalClassifier(embed_dim=5120, schema=schema, probe_type=args.probe_type)
    criterion = HierLoss(schema)

    ckpt = torch.load(args.ckpt_path)
    classifier.load_state_dict(ckpt)
    classifier.to("cuda")
    ####### prepare PRISM and hier cls model #######
    ################################################


    ###############################################
    ####### prepare tree-structured dataset #######
    test_dataset = HierDatasetTest(args.raw_feature_path, args.feature_fn, model)

    test_loader = DataLoader(test_dataset, batch_size=1, shuffle=False,\
                            num_workers=1, collate_fn=collate_hier_test, pin_memory=True)

    os.makedirs(args.test_output_path, exist_ok=True)
    ####### prepare tree-structured dataset #######
    ###############################################


    ###################
    #### evalaute #####
    classifier.eval()
    pred_data = test(test_loader, classifier, criterion, threshold=args.threshold)
    #### evalaute #####
    ###################


    #######################
    #### save results #####
    os.makedirs(args.test_output_path, exist_ok=True)
    test_output_path_json = f"{args.test_output_path}/predictions.json"

    with open(test_output_path_json, "w", encoding="utf-8") as f:
        json.dump(pred_data, f, ensure_ascii=False, indent=2)

    #### save results #####
    #######################

