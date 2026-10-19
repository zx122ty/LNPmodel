"""
02 - Training: 1D-QSAR, 2D-QSAR, and UniMol2-based models
Usage: python 02_train_models.py --input_file <path> --output_dir <dir> [--smiles_col <col>] [--target_col <col>]
"""
import os, sys, argparse
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import json
import warnings
warnings.filterwarnings('ignore')

from utils import (Config, load_data, calculate_1dqsar_repr, calculate_2dqsar_repr,
                   get_unimol2_repr, evaluate)


def main():
    parser = argparse.ArgumentParser(description='Train QSAR models for LNP transfection prediction')
    parser.add_argument('--input_file', type=str, required=True, help='Path to input CSV file (merged with SMILES)')
    parser.add_argument('--output_dir', type=str, required=True, help='Output directory for results')
    parser.add_argument('--smiles_col', type=str, default='SMILES', help='SMILES column name')
    parser.add_argument('--target_col', type=str, default='TARGET', help='Target column name')
    parser.add_argument('--target_log', action='store_true', help='Apply log10 transform')
    parser.add_argument('--test_size', type=float, default=0.2, help='Test split ratio')
    parser.add_argument('--random_state', type=int, default=42, help='Random seed')
    parser.add_argument('--unimol_size', type=str, default='84m', help='UniMol2 model size')
    parser.add_argument('--unimol_batch_size', type=int, default=4, help='UniMol2 batch size')
    parser.add_argument('--skip_unimol', action='store_true', help='Skip UniMol2 repr (no GPU)')
    args = parser.parse_args()

    cfg = Config(
        INPUT_FILE=args.input_file,
        SMILES_COL=args.smiles_col,
        TARGET_COL=args.target_col,
        TARGET_LOG=args.target_log,
        TEST_SIZE=args.test_size,
        RANDOM_STATE=args.random_state,
        UNIMOL_MODEL_SIZE=args.unimol_size,
        UNIMOL_BATCH_SIZE=args.unimol_batch_size,
        OUTPUT_DIR=args.output_dir,
    )
    os.makedirs(cfg.OUTPUT_DIR, exist_ok=True)
    os.makedirs(os.path.join(cfg.OUTPUT_DIR, 'data'), exist_ok=True)

    # ============== Load Data ==============
    print("Loading data...")
    df = pd.read_csv(args.input_file)
    print(f"Dataset: {len(df)} samples")

    # Use TARGET_log10 if exists, else TARGET
    if 'TARGET_log10' in df.columns:
        target_col = 'TARGET_log10'
        print("Using TARGET_log10 column")
    else:
        df['TARGET_log10'] = df['TARGET'].values.copy()
        target_col = 'TARGET_log10'

    from sklearn.model_selection import train_test_split
    train_df, test_df = train_test_split(df, test_size=cfg.TEST_SIZE, random_state=cfg.RANDOM_STATE)
    print(f"Train: {len(train_df)}, Test: {len(test_df)}")

    train_y = train_df[target_col].values
    test_y = test_df[target_col].values

    results = {}

    # ============== 1D-QSAR ==============
    print("\n" + "=" * 60)
    print("1D-QSAR: Calculating molecular descriptors...")
    train_df['1d_mr'] = train_df['SMILES'].apply(calculate_1dqsar_repr)
    test_df['1d_mr'] = test_df['SMILES'].apply(calculate_1dqsar_repr)
    train_x_1d = np.array(train_df['1d_mr'].tolist())
    test_x_1d = np.array(test_df['1d_mr'].tolist())
    print(f"1D feature shape: {train_x_1d.shape}")

    # ============== 2D-QSAR ==============
    print("\n" + "=" * 60)
    print("2D-QSAR: Calculating Morgan fingerprints...")
    train_df['2d_mr'] = train_df['SMILES'].apply(calculate_2dqsar_repr)
    test_df['2d_mr'] = test_df['SMILES'].apply(calculate_2dqsar_repr)
    train_x_2d = np.array(train_df['2d_mr'].tolist())
    test_x_2d = np.array(test_df['2d_mr'].tolist())
    print(f"2D feature shape: {train_x_2d.shape}")

    # ============== ML Models ==============
    from sklearn.ensemble import RandomForestRegressor, GradientBoostingRegressor
    from xgboost import XGBRegressor
    from lightgbm import LGBMRegressor
    from sklearn.svm import SVR
    from sklearn.neural_network import MLPRegressor
    from sklearn.neighbors import KNeighborsRegressor

    ml_models = [
        ("Random Forest", RandomForestRegressor(n_estimators=200, max_depth=20, random_state=42, n_jobs=-1)),
        ("XGBoost", XGBRegressor(n_estimators=200, max_depth=10, learning_rate=0.05, random_state=42, n_jobs=-1)),
        ("LightGBM", LGBMRegressor(n_estimators=200, max_depth=10, learning_rate=0.05, random_state=42, n_jobs=-1, verbose=-1)),
        ("Gradient Boosting", GradientBoostingRegressor(n_estimators=200, max_depth=5, learning_rate=0.05, random_state=42)),
        ("SVR", SVR(kernel='rbf', C=10, gamma='scale')),
        ("KNN", KNeighborsRegressor(n_neighbors=5)),
        ("MLP", MLPRegressor(hidden_layer_sizes=(128, 64, 32), learning_rate_init=0.001,
                              max_iter=1000, random_state=42, early_stopping=True)),
    ]

    def train_and_eval(name, model, train_x, test_x, prefix):
        model.fit(train_x, train_y)
        pred_train = model.predict(train_x)
        pred_test = model.predict(test_x)
        key = f"{prefix}-{name}"
        results[key] = {'train': evaluate(train_y, pred_train), 'test': evaluate(test_y, pred_test)}
        print(f"  [{key}] Test R2={results[key]['test']['R2']:.4f}, RMSE={results[key]['test']['RMSE']:.4f}")

    # 1D-QSAR
    print("\n1D-QSAR: Training ML models...")
    for name, model in ml_models:
        train_and_eval(name, model, train_x_1d, test_x_1d, "1D-QSAR")

    # 2D-QSAR
    print("\n2D-QSAR: Training ML models...")
    for name, model in ml_models:
        train_and_eval(name, model, train_x_2d, test_x_2d, "2D-QSAR")

    # ============== UniMol2 + ML ==============
    if not args.skip_unimol:
        print("\n" + "=" * 60)
        print("UniMol2 Representation + ML models...")
        all_smiles = df['SMILES'].tolist()
        print(f"Generating UniMol2 representations ({len(all_smiles)} molecules)...")
        all_repr = get_unimol2_repr(all_smiles, cfg)
        print(f"UniMol2 repr shape: {all_repr.shape}")

        train_idx = train_df.index
        test_idx = test_df.index
        train_x_um = all_repr[train_idx]
        test_x_um = all_repr[test_idx]

        print("UniMol2 + ML: Training models...")
        for name, model in ml_models:
            train_and_eval(name, model, train_x_um, test_x_um, "UniMol2-ML")

        # Save UniMol2 representations
        np.save(os.path.join(cfg.OUTPUT_DIR, 'data', 'unimol2_repr.npy'), all_repr)

    # ============== Save Results ==============
    results_df = pd.DataFrame({
        k: {
            'test_MSE': v['test']['MSE'],
            'test_RMSE': v['test']['RMSE'],
            'test_MAE': v['test']['MAE'],
            'test_R2': v['test']['R2'],
            'test_Spearman': v['test']['Spearman'],
        }
        for k, v in results.items()
    }).T.sort_values('test_R2', ascending=False)

    results_df.to_csv(os.path.join(cfg.OUTPUT_DIR, 'model_comparison.csv'))
    print("\n" + "=" * 60)
    print("Results:")
    print(results_df.to_string())

    # Plot
    fig, ax = plt.subplots(figsize=(12, 6))
    colors = ['#1f77b4' if '1D' in k else '#ff7f0e' if '2D' in k else '#2ca02c' for k in results_df.index]
    ax.barh(range(len(results_df)), results_df['test_R2'].values, color=colors)
    ax.set_yticks(range(len(results_df)))
    ax.set_yticklabels(results_df.index, fontsize=9)
    ax.set_xlabel('Test R² Score')
    ax.set_title('Model Comparison')
    ax.axvline(0, color='gray', linestyle='-', linewidth=0.5)
    from matplotlib.patches import Patch
    ax.legend(handles=[
        Patch(facecolor='#1f77b4', label='1D-QSAR'),
        Patch(facecolor='#ff7f0e', label='2D-QSAR'),
        Patch(facecolor='#2ca02c', label='UniMol2 + ML'),
    ], loc='lower right')
    plt.tight_layout()
    plt.savefig(os.path.join(cfg.OUTPUT_DIR, 'model_comparison.png'), dpi=150)
    print(f"\nSaved results to {cfg.OUTPUT_DIR}")


if __name__ == '__main__':
    main()