"""
04 - Predict new molecules (inference)
Usage: python 04_predict.py --model_dir <dir> --input_file <path> --output <path>
"""
import os, sys, argparse
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import pandas as pd
import numpy as np
import warnings
warnings.filterwarnings('ignore')


def main():
    parser = argparse.ArgumentParser(description='Predict transfection efficiency')
    parser.add_argument('--model_dir', type=str, required=True, help='Directory with trained model')
    parser.add_argument('--input_file', type=str, help='CSV file with SMILES')
    parser.add_argument('--smiles', type=str, nargs='+', help='SMILES strings to predict')
    parser.add_argument('--smiles_col', type=str, default='SMILES', help='SMILES column name')
    parser.add_argument('--method', type=str, default='finetune', choices=['finetune', 'ml'],
                        help='Prediction method')
    parser.add_argument('--output', type=str, default=None, help='Output CSV path')
    args = parser.parse_args()

    if args.smiles:
        smiles_list = args.smiles
    elif args.input_file:
        df = pd.read_csv(args.input_file)
        smiles_list = df[args.smiles_col].tolist()
    else:
        print("Please provide --smiles or --input_file")
        sys.exit(1)

    print(f"Predicting {len(smiles_list)} molecules...")

    if args.method == 'finetune':
        from unimol_tools import MolPredict
        import tempfile, yaml, shutil

        # Check if kfold needs fixing
        config_path = os.path.join(args.model_dir, 'config.yaml')
        with open(config_path, 'r') as f:
            config = yaml.safe_load(f)
        model_files = [f for f in os.listdir(args.model_dir) if f.endswith('.pth')]
        n_models = len(model_files)
        actual_kfold = config.get('kfold', 5)

        if n_models < actual_kfold:
            tmp_save = args.model_dir + '_tmp_predict'
            os.makedirs(tmp_save, exist_ok=True)
            for fname in model_files:
                shutil.copy2(os.path.join(args.model_dir, fname), tmp_save)
            if os.path.exists(os.path.join(args.model_dir, 'target_scaler.ss')):
                shutil.copy2(os.path.join(args.model_dir, 'target_scaler.ss'), tmp_save)
            config['kfold'] = n_models
            with open(os.path.join(tmp_save, 'config.yaml'), 'w') as f:
                yaml.dump(config, f)
            predm = MolPredict(load_model=tmp_save)
            shutil.rmtree(tmp_save)
        else:
            predm = MolPredict(load_model=args.model_dir)

        import tempfile
        tmpdir = tempfile.mkdtemp()
        tmp_csv = os.path.join(tmpdir, 'tmp.csv')
        pd.DataFrame({args.smiles_col: smiles_list}).to_csv(tmp_csv, index=False)
        pred = predm.predict(tmp_csv).reshape(-1)
    else:
        from utils import get_unimol2_repr, Config
        cfg = Config()
        reprs = get_unimol2_repr(smiles_list, cfg)
        from joblib import load
        model_path = os.path.join(args.model_dir, 'best_model.joblib')
        if os.path.exists(model_path):
            model = load(model_path)
            pred = model.predict(reprs)
        else:
            print(f"ML model not found at {model_path}")
            return

    results = pd.DataFrame({
        args.smiles_col: smiles_list,
        'prediction': pred,
    })

    if args.output:
        results.to_csv(args.output, index=False)
        print(f"Saved predictions to {args.output}")
    else:
        print(results.to_string())


if __name__ == '__main__':
    main()