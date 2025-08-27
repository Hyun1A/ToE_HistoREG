## 🚀 Intallation

### Setup for experiments

**OS**: Ubuntu 20.04.6 LTS

**Python**: 3.10.18

Install conda an enviroment
<pre>
conda create -n toe python=3.10.18
</pre>

Install pytorch
<pre>
pip install torch==2.3.0 torchvision==0.18.0 torchaudio==2.3.0 --index-url https://download.pytorch.org/whl/cu121
</pre>

Please install packages in requirements.txt
<pre>
pip install -r requirements.txt
</pre>


## 💻 Preparing tree labels and a pretrained model

### Constructing for tree labels
We already constructed the tree labels for training dataset saved in `./toe/data/tree_samples` from the root directory. If you want to regenerate them, please refer to `toe/construct_tree_labels`. For example, `toe/construct_tree_labelsconstruct_train_tree_dataset_lung.py` aims to construct the tree labels for training samples whose organ labels are "lung".

### Perparing slide encoder of Prism [1] (Perceiver) [2] 
Please pull the repository of [PRISM](https://huggingface.co/paige-ai/Prism) from huggingface and replace the empty "Prism" folder in `toe`. (The repository must be moved exactly as it is.)



## ▶️ Usage


### Train

Run the scripts after specifying the directories of patch features, model path of PRISM, tree samples, and path to save slide features in `.sh` files. The options are:

<pre>
    --aggregator_path ./Prism \
    --raw_feature_path {TRAIN_PATCH_PATH} \
    --val_anno_path ./data/val_samples.json \
    --feature_fn ./data/feature_cache/features_all.pt \
    --records_path ./data/tree_samples \
</pre>
The option to specify in practice is `raw_feature_path`, the path to extracted train patch features


<pre>
bash train_hier_cls_mlp.sh
</pre>


### Test

Run the scripts after specifying the directories of patch features, model path of PRISM, tree samples, and path to save slide features in `.sh` files. The options are:

<pre>
    --aggregator_path ./Prism \
    --raw_feature_path {TEST_PATCH_PATH} \
    --feature_fn ./data/feature_cache/features_test_phase2.pt \
</pre>
The option to specify in practice is `raw_feature_path`, the path to extracted test patch features.

<pre>
bash test_hier_cls_mlp.sh
</pre>

**For test without training**, we provided checkpoints and test slide features through [https://huggingface.co/Hyun1A/ToE_HistoREG/tree/main](hugginface).
Put the checkpoint `ckpt_mlp.sh` in `toe/ckpt` and the test slide features `features_test_phase2.pt` in `toe/data/feature_cache`



### References
[1] Shaikovski, George, et al. "Prism: A multi-modal generative foundation model for slide-level histopathology." arXiv preprint arXiv:2405.10254 (2024).

[2] Jaegle, Andrew, et al. "Perceiver: General perception with iterative attention." International conference on machine learning. PMLR, 2021.


## 📜 License

Please check the [PRISM repository](https://huggingface.co/paige-ai/Prism) for license details.
