from datasets import load_dataset, Sequence, ClassLabel
import argparse

def main():
    ap = argparse.ArgumentParser("Upload JSONL file to huggingface.")
    ap.add_argument("--jsonl", type=str, help="The input file.", required=True)
    args = ap.parse_args()

    dataset = load_dataset("json", data_files=args.jsonl)
    
    features = dataset['train'].features.copy()
    features["highlight_labels"] = Sequence(feature=ClassLabel(names=["no", "yes"]), id=None)
    features["underline_labels"] = Sequence(feature=ClassLabel(names=["no", "yes"]), id=None)
    features["emphasis_labels"] = Sequence(feature=ClassLabel(names=["no", "yes"]), id=None)

    dataset = dataset['train'].map(lambda x : x, features=features, batched=True)

    dataset.push_to_hub(repo_id="hspolicy/ndca_openev_2024_cards")