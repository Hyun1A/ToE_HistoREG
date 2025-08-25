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
import wandb

from models.hier_classifiers import build_schema, extract_procedures_from_json
from models.hier_classifiers import HierarchicalClassifier, HierLoss, HierLossMixup
from models.hier_classifiers import build_targets
from utils.optimizer import *
from utils.formatting import FORMAT_REGISTRY
from utils.dataset import HierDataset, HierDatasetOrganwise, HierDatasetNoisyMixup, collate_hier, collate_hier_mixup

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

    return scores




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

    return scores, score_dict



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




def train(max_epochs, train_loader, val_loader, classifier, criterion, optimizer, scheduler, threshold, args):
    for epoch in range(max_epochs):
        classifier.train()

        print(f"Epoch: {epoch}/{max_epochs}")
        
        pbar = tqdm(train_loader, desc="train", leave=False)

        losses_log = {"organ": [], "colon": [], "procedure": [], "top": [], "sub": [], "total": []}
        for idx, data_pair in enumerate(pbar, start=1):

            ratio = torch.tensor(data_pair[-1]).to("cuda")

            true_report_list = []
            tgt_list = []

            for data in data_pair[:2]:
                record = data["struct_report"]
                true_report = data["report"]
                tgt = build_targets(record, schema, "cuda", ORGANS)
                true_report_list.append(true_report)
                tgt_list.append(tgt)

            image_proj_mixup = data_pair[2]["image_feat"][:,0,:].to("cuda")
            outs = classifier(image_proj_mixup)

            losses = criterion(outs, tgt_list, ratio)

            loss = losses['total']

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            
            for key, val in losses.items():
                losses_log[key].append(val.detach().item())

            if args.show_rep_period > 0 and ((epoch) % args.show_rep_period==0) and (idx == len(train_loader)):

                idx_large = ( ratio <= 0.5 ).int().tolist()
                true_report = [true_report_list[idx_large[n]][n] for n in range(len(idx_large))]

                preds = classifier.decode(outs, threshold=0.5)

                gen_report = []
                for pred in preds:
                    pred_organ = pred['organ'].lower()
                    rep = FORMAT_REGISTRY[pred_organ](pred)
                    gen_report.append(rep)

                for i in range(5): print(f"{i}, Generated report"); print(gen_report[i]); print(); print(f"{i}, True report"); print(true_report[i]); print()

        for key, val in losses_log.items():
            losses_log[key] = torch.tensor(val).mean().item()

        if args.use_wandb:
            wandb.log(
                {
                    "epoch": epoch + 1,
                    "train/loss_organ":     losses_log["organ"],
                    "train/loss_procedure": losses_log["procedure"],
                    "train/loss_top":       losses_log["top"],
                    "train/loss_sub":       losses_log["sub"],
                    "train/loss_total":     losses_log["total"],
                    "batchsize": args.batch_size,
                    "lr": optimizer.param_groups[0]["lr"],
                },
                step=epoch + 1,       
            )


        scheduler.step()

        if epoch % args.val_period == 0 or epoch == max_epochs:
        # if False:
            print("Eval val data`")
            eval_results, final_score = validate(val_loader, classifier, criterion, threshold=args.threshold)

            if args.use_wandb:
                wandb.log(
                    {
                        "val/Ranking": final_score[0],
                        "val/ROUGE": final_score[1],
                        "val/BLEU": final_score[2],
                        "val/KEY": final_score[3],
                        "val/EMB": final_score[4],
                    },
                    step=epoch + 1,       
                )

            torch.save(classifier.cpu().state_dict(), args.ckpt_path)
            classifier.to("cuda")



    return classifier



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
                        default="./results/exp_final_train")
    parser.add_argument("--ckpt_path", type=str, 
                        default="./ckpt/ckpt_final.pt")
    parser.add_argument("--project_name", type=str, default="REG-hier-classifier")
    parser.add_argument("--exp_name", type=str, default="all_organs")    
    parser.add_argument("--probe_type", type=str, default="linear")    

    parser.add_argument("--batch_size", type=int, default=1024)
    parser.add_argument("--max_epochs", type=int, default=1000)
    parser.add_argument("--threshold", type=float, default=0.5)
    parser.add_argument("--show_rep_period", type=int, default=50)
    parser.add_argument("--val_period", type=int, default=200)
    parser.add_argument("--alpha", type=float, default=0.2)
    parser.add_argument("--lam", type=float, default=0.01)

    parser.add_argument("--lr", type=float, default=3e-3)
    parser.add_argument("--wd", type=float, default=1e0)
    parser.add_argument("--use_wandb", type=bool, default=True)


    args = parser.parse_args()
        

    if args.use_wandb:
        wandb.init(
            project=args.project_name,    
            name=args.exp_name,
        )


    ################################################
    ####### prepare PRISM and hier cls model #######
    ## Load PRISM model.

    model = AutoModel.from_pretrained(args.aggregator_path, trust_remote_code=True, local_files_only=True)
    model = model.to('cuda')
    model.eval()

    ## construct schema and model
    tree_dict = {organ:tree for (organ, tree) in zip(ORGANS, TREES)}
    schema = build_schema(tree_dict, ORGANS)

    classifier = HierarchicalClassifier(embed_dim=5120, schema=schema, probe_type=args.probe_type).to("cuda")
    criterion = HierLossMixup(schema, organ_weight=0.2, proc_weight=0.5, top_weight=2.0, sub_weight=1.0)
    ####### prepare PRISM and hier cls model #######
    ################################################


    ###############################################
    ####### prepare tree-structured dataset #######
    train_dataset = HierDatasetNoisyMixup(args.annotation_path, args.val_anno_path,
                    args.raw_feature_path, args.records_path, args.feature_fn, model, mode="train", alpha=args.alpha, lam=args.lam)


    train_loader = DataLoader(train_dataset, batch_size=args.batch_size, shuffle=True,\
                            num_workers=4, collate_fn=collate_hier_mixup, pin_memory=True)

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


    #################################
    ####### prepare optimizer #######
    optimizer = build_optimizer(
        classifier,
        backbone_lr=args.lr,
        head_lr=args.lr,      
        weight_decay=args.wd,
    )

    scheduler = build_scheduler(optimizer, max_epochs=args.max_epochs)
    ####### prepare optimizer #######
    #################################


    ################
    #### train #####
    classifier = train(args.max_epochs, train_loader, val_loader_all, classifier, \
                    criterion, optimizer, scheduler, threshold=args.threshold, args=args)
    #### train #####
    ################


    ###################
    #### evalaute #####
    classifier.eval()
    eval_results, final_score_all = validate(val_loader_all, classifier, criterion, threshold=args.threshold)

    if args.use_wandb:
        wandb.log(
            {
                "val/Ranking": final_score_all[0],
                "val/ROUGE": final_score_all[1],
                "val/BLEU": final_score_all[2],
                "val/KEY": final_score_all[3],
                "val/EMB": final_score_all[4],
            },
            step=args.max_epochs + 1,       
        )


    #######################
    #### save results #####
    df = pd.DataFrame(eval_results)
    os.makedirs(args.test_output_path, exist_ok=True)
    test_output_csv = f"{args.test_output_path}/valid_outputs.csv"
    df.to_csv(test_output_csv, index=False)
    #### save results #####
    #######################
    

    final_scores = dict()
    for organ, val_loader in val_loaders.items():
        eval_results, final_score = validate_organwise(val_loader, classifier, criterion, threshold=args.threshold, organ=organ, args=args)
        final_scores[organ] = final_score[:-1]
    final_scores["All"] = final_score_all

    metrics = ["Ranking Score", "ROUGE Score", "BLEU Score", "KEY Score", "EMB Score"]

    df = pd.DataFrame(final_scores, index=metrics)
    df.reset_index(inplace=True)
    df.rename(columns={'index': 'Metric'}, inplace=True)
    df.to_csv(f"{args.test_output_path}/valid_outputs_organwise.csv", index=False)
    print(df)
    #### evalaute #####
    ###################

    torch.save(classifier.cpu().state_dict(), args.ckpt_path)

    if args.use_wandb:
        wandb.finish()


