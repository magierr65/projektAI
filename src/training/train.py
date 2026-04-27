import sys
import os
import argparse

sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from src.preprocessing.feature_engineering import load_and_preprocess_data, load_and_preprocess_rule_aided_data
from src.models.mlp_model import run_mlp_training
from src.models.committee_model import run_committee_training
from src.models.modular_model import run_modular_training
from src.models.rule_aided_model import run_rule_aided_training

def main():
    parser = argparse.ArgumentParser(description="Train a selected model.")
    parser.add_argument('model', type=str, choices=['mlp', 'committee', 'modular', 'rule_aided'],
                        help="The model to train.")
    args = parser.parse_args()

    print(f"Starting training for {args.model} model...")

    if args.model == 'rule_aided':
        df = load_and_preprocess_rule_aided_data()
        run_rule_aided_training(df)
    else:
        df = load_and_preprocess_data()
        if args.model == 'mlp':
            run_mlp_training(df)
        elif args.model == 'committee':
            run_committee_training(df)
        elif args.model == 'modular':
            run_modular_training(df)

    print(f"Finished training for {args.model} model.")

if __name__ == '__main__':
    main()
