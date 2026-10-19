"""
01 - Data Exploration and Visualization
Usage: python 01_explore_data.py --input_file <path> [--smiles_col <col>] [--target_col <col>] [--output_dir <dir>]

python 01_explore_data.py --input_file <path> [--smiles_col <col>] [--target_col <col>] [--output_dir <dir>]

"""
import os, sys, argparse
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from utils import Config, load_data, load_mol_smiles, merge_data
import warnings
warnings.filterwarnings('ignore')


def main():
    parser = argparse.ArgumentParser(description='Explore data for LNP transfection prediction')
    parser.add_argument('--input_file', type=str, required=True, help='Path to input CSV/XLSX file')
    parser.add_argument('--smiles_col', type=str, default='SMILES', help='SMILES column name')
    parser.add_argument('--target_col', type=str, default='TARGET', help='Target column name')
    parser.add_argument('--mol_dir', type=str, default=None, help='Directory with .mol files (optional)')
    parser.add_argument('--name_col', type=str, default=None, help='Name column for matching mol files')
    parser.add_argument('--target_log', action='store_true', help='Apply log10 transform to target')
    parser.add_argument('--output_dir', type=str, default=None, help='Output directory')
    args = parser.parse_args()

    cfg = Config(
        INPUT_FILE=args.input_file,
        SMILES_COL=args.smiles_col,
        TARGET_COL=args.target_col,
        MOL_DIR=args.mol_dir,
        TARGET_LOG=args.target_log,
        OUTPUT_DIR=args.output_dir or os.path.join(os.path.dirname(os.path.abspath(__file__)), 'data', 'explore_output'),
    )
    os.makedirs(cfg.OUTPUT_DIR, exist_ok=True)

    print("=" * 60)
    print("Loading data...")
    df = load_data(cfg)
    print(f"Rows with labels: {len(df)}")

    if cfg.MOL_DIR:
        print("\nLoading mol files and converting to SMILES...")
        name_to_smiles = load_mol_smiles(cfg)
        name_col = args.name_col or 'name_ionziable_lipid'
        df = merge_data(df, name_to_smiles, name_col)
        print(f"After merge: {len(df)} samples")

    # Save merged data
    merged_path = os.path.join(cfg.OUTPUT_DIR, 'merged_data.csv')
    df.to_csv(merged_path, index=False)
    print(f"\nSaved merged data to {merged_path}")

    print("\n" + "=" * 60)
    print("Target Distribution:")
    print(df['TARGET_log10'].describe(percentiles=[.1, .25, .5, .75, .9, .95, .99]))

    # Plot
    fig, axes = plt.subplots(1, 2, figsize=(14, 5))
    axes[0].hist(df['TARGET'], bins=50, alpha=0.7, color='steelblue', edgecolor='white')
    axes[0].set_title('Raw Target Distribution', fontsize=14)
    axes[0].set_xlabel(args.target_col)
    axes[0].set_ylabel('Count')

    axes[1].hist(df['TARGET_log10'], bins=50, alpha=0.7, color='coral', edgecolor='white')
    axes[1].set_title('Log10 Target Distribution', fontsize=14)
    axes[1].set_xlabel('log10(' + args.target_col + ')')
    axes[1].set_ylabel('Count')

    plt.tight_layout()
    plot_path = os.path.join(cfg.OUTPUT_DIR, 'target_distribution.png')
    plt.savefig(plot_path, dpi=150)
    print(f"\nSaved plot to {plot_path}")

    print("\n" + "=" * 60)
    print("SMILES examples:")
    print(df[['SMILES', 'TARGET', 'TARGET_log10']].head(10).to_string())

    # Check SMILES validity
    from rdkit import Chem
    invalid = sum(1 for smi in df['SMILES'] if Chem.MolFromSmiles(smi) is None)
    print(f"\nInvalid SMILES: {invalid}/{len(df)}")

    print("\nDone!")


if __name__ == '__main__':
    main()