"""
03 - UniMol2 Fine-tuning (end-to-end with MolTrain/MolPredict)
Usage: python 03_unimol2_finetune.py --input_file <path> --output_dir <dir>
"""
import os, sys, argparse, shutil, yaml
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import warnings
warnings.filterwarnings('ignore')

from utils import Config, evaluate


def main():
    parser = argparse.ArgumentParser(description='UniMol2 fine-tuning for LNP transfection prediction')
    parser.add_argument('--input_file', type=str, required=True, help='Path to input CSV with SMILES')
    parser.add_argument('--output_dir', type=str, required=True, help='Output directory')
    parser.add_argument('--smiles_col', type=str, default='SMILES', help='SMILES column name')
    parser.add_argument('--target_col', type=str, default='TARGET', help='Target column name')
    parser.add_argument('--test_size', type=float, default=0.2, help='Test split ratio')
    parser.add_argument('--random_state', type=int, default=42, help='Random seed')
    parser.add_argument('--unimol_size', type=str, default='84m', help='UniMol2 model size')
    parser.add_argument('--batch_size', type=int, default=4, help='Training batch size')
    parser.add_argument('--epochs', type=int, default=50, help='Max epochs')
    parser.add_argument('--learning_rate', type=float, default=1e-4, help='Learning rate')
    parser.add_argument('--early_stopping', type=int, default=10, help='Early stopping patience')
    args = parser.parse_args()

    cfg = Config(
        INPUT_FILE=args.input_file,
        SMILES_COL=args.smiles_col,
        TARGET_COL=args.target_col,
        TEST_SIZE=args.test_size,
        RANDOM_STATE=args.random_state,
        UNIMOL_MODEL_SIZE=args.unimol_size,
        UNIMOL_BATCH_SIZE=args.batch_size,
        UNIMOL_EPOCHS=args.epochs,
        UNIMOL_LEARNING_RATE=args.learning_rate,
        UNIMOL_EARLY_STOPPING=args.early_stopping,
        OUTPUT_DIR=args.output_dir,
    )
    os.makedirs(cfg.OUTPUT_DIR, exist_ok=True)
    os.makedirs(os.path.join(cfg.OUTPUT_DIR, 'data'), exist_ok=True)

    # ============== Prepare Data ==============
    print("Loading data...")
    df = pd.read_csv(args.input_file)
    print(f"Dataset: {len(df)} samples")

    if 'TARGET_log10' in df.columns:
        target_col = 'TARGET_log10'
    else:
        target_col = 'TARGET'

    from sklearn.model_selection import train_test_split
    train_df, test_df = train_test_split(df, test_size=cfg.TEST_SIZE, random_state=cfg.RANDOM_STATE)
    print(f"Train: {len(train_df)}, Test: {len(test_df)}")

    data_dir = os.path.join(cfg.OUTPUT_DIR, 'data')
    train_csv = os.path.join(data_dir, 'unimol2_train.csv')
    test_csv = os.path.join(data_dir, 'unimol2_test.csv')
    train_df[['SMILES', target_col]].to_csv(train_csv, index=False)
    test_df[['SMILES', target_col]].to_csv(test_csv, index=False)

    # ============== Fine-tuning ==============
    print("\n" + "=" * 60)
    print("UniMol2 Fine-tuning with MolTrain...")

    from unimol_tools import MolTrain, MolPredict

    save_path = os.path.join(cfg.OUTPUT_DIR, 'unimol2_finetune')

    size_upper = cfg.UNIMOL_MODEL_SIZE.upper()
    # Use LipidFM v2 checkpoint instead of the original 84M base
    LIPIDFM_CKPT = "/home/data/zhengxiao/disk1/my_project/lnpModel/code/01_continous_training/2_models/continued_pretrain/unimol2_lipids_84m_v2/checkpoints/checkpoint_best.pt"
    model_path = LIPIDFM_CKPT
    dict_path = f'{cfg.PRETRAINED_DIR}/dict.txt'

    clf = MolTrain(
        task='regression',
        data_type='molecule',
        model_name='unimolv2',
        model_size=cfg.UNIMOL_MODEL_SIZE,
        epochs=cfg.UNIMOL_EPOCHS,
        learning_rate=cfg.UNIMOL_LEARNING_RATE,
        batch_size=cfg.UNIMOL_BATCH_SIZE,
        early_stopping=cfg.UNIMOL_EARLY_STOPPING,
        metrics='mse',
        split='random',
        save_path=save_path,
        max_atoms=256,
        pretrained_model_path=model_path,
        pretrained_dict_path=dict_path,
    )

    print("Starting training...")
    clf.fit(train_csv)

    # ============== Prediction ==============
    print("\n" + "=" * 60)
    print("Predicting...")

    # Need to fix kfold count in config if early stopping cut it short
    config_path = os.path.join(save_path, 'config.yaml')
    with open(config_path, 'r') as f:
        config = yaml.safe_load(f)

    actual_kfold = config.get('kfold', 5)
    model_files = [f for f in os.listdir(save_path) if f.endswith('.pth')]
    n_models = len(model_files)
    if n_models < actual_kfold:
        tmp_save = save_path + '_tmp'
        os.makedirs(tmp_save, exist_ok=True)
        for fname in model_files:
            shutil.copy2(os.path.join(save_path, fname), tmp_save)
        if os.path.exists(os.path.join(save_path, 'target_scaler.ss')):
            shutil.copy2(os.path.join(save_path, 'target_scaler.ss'), tmp_save)
        config['kfold'] = n_models
        with open(os.path.join(tmp_save, 'config.yaml'), 'w') as f:
            yaml.dump(config, f)
        predm = MolPredict(load_model=tmp_save)
        shutil.rmtree(tmp_save)
    else:
        predm = MolPredict(load_model=save_path)

    pred_train_y = predm.predict(train_csv).reshape(-1)
    pred_test_y = predm.predict(test_csv).reshape(-1)

    train_metrics = evaluate(train_df[target_col].values, pred_train_y)
    test_metrics = evaluate(test_df[target_col].values, pred_test_y)

    print(f"\nTrain: {train_metrics}")
    print(f"Test:  {test_metrics}")

    # Save results
    flat_results = {}
    for k, v in train_metrics.items():
        flat_results[f'train_{k}'] = v
    for k, v in test_metrics.items():
        flat_results[f'test_{k}'] = v
    pd.DataFrame([flat_results]).to_csv(
        os.path.join(cfg.OUTPUT_DIR, 'unimol2_finetune_results.csv'), index=False)

    # Scatter plot
    fig, axes = plt.subplots(1, 2, figsize=(12, 5))
    for ax, y_true, y_pred, title in [
        (axes[0], train_df[target_col].values, pred_train_y, 'Train'),
        (axes[1], test_df[target_col].values, pred_test_y, 'Test')
    ]:
        ax.scatter(y_true, y_pred, alpha=0.6, s=20)
        lims = [min(y_true.min(), y_pred.min()), max(y_true.max(), y_pred.max())]
        ax.plot(lims, lims, 'r--', alpha=0.5)
        ax.set_xlabel('True')
        ax.set_ylabel('Predicted')
        ax.set_title(f'{title} (R²={test_metrics["R2"] if title=="Test" else train_metrics["R2"]:.3f})')
    plt.tight_layout()
    plt.savefig(os.path.join(cfg.OUTPUT_DIR, 'unimol2_finetune_scatter.png'), dpi=150)
    print("Saved scatter plot")


if __name__ == '__main__':
    main()