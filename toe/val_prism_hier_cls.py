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
from utils.dataset import HierDataset, HierDatasetOrganwise, collate_hier

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




def get_scores_record(gt_data, pred_data):
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

    scores = evaluator.evaluate_compwise_record(eval_pairs)

    score_dict = scores[-1]
    
    print(f"\n Average Ranking Score: {scores[0]:.4f}")
    print(f"Average ROUGE Score: {scores[1]:.4f}")
    print(f"Average BLEU Score: {scores[2]:.4f}")
    print(f"Average KEY Score: {scores[3]:.4f}")
    print(f"Average EMB Score: {scores[4]:.4f}")

    return scores[0], score_dict





def validate(val_loader, classifier, criterion, threshold):
    classifier.eval()

    print(f"Validate")
    
    pbar = tqdm(val_loader, desc="evaluate", leave=False)

    case_ids_all = []
    organs_all = []
    gen_report_all = []
    true_report_all = []

    with torch.no_grad():
        for idx, data in enumerate(pbar, start=1):
            case_id = data["case_id"]
            image_proj = data["image_feat"][:,0,:].to("cuda")
            true_report = data["report"]
            
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
            true_report_all += true_report

    eval_results = {"case_ids": case_ids_all, \
            "organs": organs_all, \
            "true_reports":true_report_all, \
            "gen_reports":gen_report_all \
            }
    

    gt_data = [{"id":case_id, "report": report} for (case_id, report) in zip(eval_results["case_ids"], eval_results["true_reports"])]
    pred_data = [{"id":case_id, "report": report} for (case_id, report) in zip(eval_results["case_ids"], eval_results["gen_reports"])]


    with open(f"./results/tmp/gt_data.json", "w", encoding="utf-8") as f:
        json.dump(gt_data, f, ensure_ascii=False, indent=2)
    with open(f"./results/tmp/pred_data.json", "w", encoding="utf-8") as f:
        json.dump(pred_data, f, ensure_ascii=False, indent=2)
        

    final_score = get_scores(gt_data, pred_data)

    df = pd.DataFrame(eval_results)
    os.makedirs("./results/tmp", exist_ok=True)
    test_output_csv = f"./results/tmp/valid_outputs.csv"
    df.to_csv(test_output_csv, index=False)

    return eval_results, final_score





def validate_organwise(val_loader, classifier, criterion, threshold, organ, args):
    classifier.eval()

    print()
    print('#'*30)
    print(f"Validate for {organ}")
    print('#'*30)

    pbar = tqdm(val_loader, desc="evaluate", leave=False)

    case_ids_all = []
    organs_all = []
    gen_report_all = []
    true_report_all = []

    with torch.no_grad():
        for idx, data in enumerate(pbar, start=1):

            case_id = data["case_id"]
            image_proj = data["image_feat"][:,0,:].to("cuda")
            true_report = data["report"]
            
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
            true_report_all += true_report

    eval_results = {"case_ids": case_ids_all, \
            "organs": organs_all, \
            "true_reports":true_report_all, \
            "gen_reports":gen_report_all \
            }
    

    gt_data = [{"id":case_id, "report": report} for (case_id, report) in zip(eval_results["case_ids"], eval_results["true_reports"])]
    pred_data = [{"id":case_id, "report": report} for (case_id, report) in zip(eval_results["case_ids"], eval_results["gen_reports"])]

    final_score, score_dict = get_scores_record(gt_data, pred_data)

    eval_results.update(score_dict)

    df = pd.DataFrame(eval_results)
    test_output_csv = f"{args.test_output_path}/valid_outputs_{organ}.csv"
    df.to_csv(test_output_csv, index=False)

    return eval_results, final_score





if __name__ == "__main__":

    parser = argparse.ArgumentParser()

    parser.add_argument("--aggregator_path", type=str,
                        default="../Prism")
    parser.add_argument("--annotation_path", type=str, 
                        default="/home2/hyun/REG/data/reg_data/train.json")
    parser.add_argument("--val_anno_path", type=str, 
                        default="/home2/hyun/REG/proposed/exp_hier_cls_base/data/val_samples.json")
    parser.add_argument("--raw_feature_path", type=str, 
                        default="/home2/hyun/REG/data/patches_224/reg_data/virchow/pt_files")
    parser.add_argument("--feature_fn", type=str, 
                        default="/home2/hyun/REG/proposed/exp_hier_cls_base/data/feature_cache/features_all.pt")
    parser.add_argument("--records_path", type=str, 
                        default='/home2/hyun/REG/proposed/exp_hier_cls_base/data/tree_samples')
    parser.add_argument("--test_output_path", type=str, 
                        default="./results/exp_final_val")
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
    tree_dict = {organ:tree for (organ, tree) in zip(ORGANS, TREES)}
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
    val_dataset_all = HierDataset(args.annotation_path, args.val_anno_path,
                    args.raw_feature_path, args.records_path, args.feature_fn, model, mode="val")

    val_loader_all = DataLoader(val_dataset_all, batch_size=1, shuffle=False,\
                            num_workers=1, collate_fn=collate_hier, pin_memory=True)


    val_loaders = dict()

    for organ in ORGANS:
        val_dataset_organ = HierDatasetOrganwise(args.annotation_path, args.val_anno_path,
                        args.raw_feature_path, args.records_path, args.feature_fn, model, organ, mode="val")

        val_loader = DataLoader(val_dataset_organ, batch_size=1, shuffle=False,\
                                num_workers=1, collate_fn=collate_hier, pin_memory=True)

        val_loaders[organ] = val_loader



    os.makedirs(args.test_output_path, exist_ok=True)
    ####### prepare tree-structured dataset #######
    ###############################################


    ###################
    #### evalaute #####
    classifier.eval()
    eval_results, final_score = validate(val_loader_all, classifier, criterion, threshold=args.threshold)


    final_scores = dict()
    for organ, val_loader in val_loaders.items():
        eval_results, final_score = validate_organwise(val_loader, classifier, criterion, threshold=args.threshold, organ=organ, args=args)
        final_scores[organ] = final_score



    #### evalaute #####
    ###################

    #######################
    #### save results #####
    df = pd.DataFrame(eval_results)
    os.makedirs(args.test_output_path, exist_ok=True)
    test_output_csv = f"{args.test_output_path}/valid_outputs.csv"
    df.to_csv(test_output_csv, index=False)
    #### save results #####
    #######################

