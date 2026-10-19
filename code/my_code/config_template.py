"""
Configuration template for LNP transfection efficiency prediction.
Copy this file and modify as needed for each dataset.
"""
import os

# ============== Data Paths ==============
# For LUMI-lab dataset
INPUT_DIR = "/home/data/zhengxiao/disk1/my_project/lnpModel/other_model/LUMI-lab/data/LUMI-lab"
INPUT_FILE = os.path.join(INPUT_DIR, "4CR-1920.csv")
OUTPUT_DIR = "/home/data/zhengxiao/disk1/my_project/lnpModel/code/my_code/results_lumi"

# For original dataset (uncomment to use)
# INPUT_DIR = "/home/data/zhengxiao/disk1/my_project/lnpModel/data/raw_data/大库材料+转染结果/大库材料+转染结果"
# INPUT_FILE = os.path.join(INPUT_DIR, "大库材料的体外转染结果.xlsx")
# MOL_DIR = os.path.join(INPUT_DIR, "大库材料的分子结构")
# OUTPUT_DIR = "/home/data/zhengxiao/disk1/my_project/lnpModel/code/my_code/results"

# ============== Data Columns ==============
SMILES_COL = "mol"          # Column name for SMILES
TARGET_COL = "RLU (log2)"   # Column name for target value
TARGET_LOG = False           # Whether to apply log10 transform to target

# ============== Model Settings ==============
TEST_SIZE = 0.2
RANDOM_STATE = 42

# ============== UniMol2 Settings ==============
PRETRAINED_DIR = "/home/data/zhengxiao/disk1/my_project/Uni-Mol2_Toxicity_Prediction/2_models/pretrained"
UNIMOL_MODEL_SIZE = "84m"
UNIMOL_BATCH_SIZE = 4
UNIMOL_EPOCHS = 50
UNIMOL_LEARNING_RATE = 1e-4
UNIMOL_EARLY_STOPPING = 10