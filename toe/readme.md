
## Getting Started

## Setup for experiments

**OS**: Ubuntu 20.04.6 LTS
**Python**: 3.10.18

<pre>
conda create -n toe python=3.10.18
</pre>

Please install packages in requirements.txt
<pre>
pip install -r requirements.txt
</pre>


## Constructing for tree labels
We already constructed the tree labels for training dataset saved in *data/tree_samples* from the root directory. If you want to regenerate them, please refer to *toe/construct_tree_labels*

For example, *toe/construct_tree_labelsconstruct_train_tree_dataset_lung.py* aims to construct the tree labels for training samples whose organ labels are "lung"


## Perparing slide encoder of Prism [1] (Perceiver) [2] 
Please pull the repository of [PRISM](https://huggingface.co/paige-ai/Prism) from huggingface and replace the empty "Prism" folder in *toe*. (The repository must be moved exactly as it is.)

## Training
<pre>
bash train_hier_cls_mlp.sh
</pre>


## Test
<pre>
bash test_hier_cls_mlp.sh
</pre>



## References
[1] Shaikovski, George, et al. "Prism: A multi-modal generative foundation model for slide-level histopathology." arXiv preprint arXiv:2405.10254 (2024).
[2] Jaegle, Andrew, et al. "Perceiver: General perception with iterative attention." International conference on machine learning. PMLR, 2021.