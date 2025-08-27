## TRIDENT

[TRIDENT](https://github.com/mahmoodlab/TRIDENT) is a tool for efficient processing and analysis of whole-slide images (WSIs).  
This repository provides simple shell scripts to streamline training and testing workflows.

---

## 🚀 Installation

First, clone the original TRIDENT repository and set up the environment.

```bash
# Clone TRIDENT
git clone https://github.com/mahmoodlab/TRIDENT.git
cd TRIDENT

# Create and activate conda environment
conda create -n trident python=3.10 -y
conda activate trident

# Install dependencies
pip install -r requirements.txt
```

---

## 📂 Shell Scripts

This project includes a `shell_scripts/` folder with example scripts for running TRIDENT on **training** and **testing** datasets.

- `run_train_data.sh`  
- `run_test_data.sh`

Each `.sh` file contains example commands with predefined arguments.  
👉 **Important**: You need to modify the file paths inside each `.sh` script to match your local environment.

---

## ▶️ Usage

Run the scripts after editing the dataset paths:

```bash
# For training data
bash shell_scripts/run_train_data.sh

# For testing data
bash shell_scripts/run_test_data.sh
```

---

## ⚠️ Notes

- Make sure the **WSI data paths** and **output directories** inside the `.sh` files are properly set before running.
- Scripts are intended as **templates** — adjust hyperparameters or flags as needed.
- For more details on TRIDENT features, please refer to the [official TRIDENT repo](https://github.com/mahmoodlab/TRIDENT).

---

## 📜 License

Please check the [TRIDENT repository](https://github.com/mahmoodlab/TRIDENT) for license details.
