import argparse
from debate_utils.models import Card
import hashlib
from typing import List
import tqdm

STRATEGIES = ["exact_text"]

def main():
    ap = argparse.ArgumentParser("Compute embeddings for cards")
    ap.add_argument("--jsonl", type=str, help="The input file.", required=True)
    ap.add_argument("--strategy", type=str, help="The input file.", choices=STRATEGIES, required=True)
    ap.add_argument("--outfile", type=str, help="The output file.", required=True)
    args = ap.parse_args()

    if args.strategy == "exact_text":
        exact_text_strategy(args.jsonl, args.outfile)
    else:
        print(f"Strategy not yet implemented: {args.strategy}")

def exact_text_strategy(jsonl_file: str, outfile: str):
    total = unique = mismatch = 0 
    with open(outfile, "w") as ofh:
        exact_hashes = set()
        tags = {}
        for card_json in tqdm.tqdm(open(jsonl_file).readlines()):
            total += 1
            card = Card.from_json(card_json)
            exact_hash = hashlib.md5(card.text_plain().encode()).hexdigest()
            if exact_hash in exact_hashes:
                if card.tag != tags[exact_hash]:
                    print(f"WARNING: tags disagree.\nRetained tag:\n\t{tags[exact_hash]}\nDiscarded tag:\n\t{card.tag}")
                    mismatch += 1
                continue
            else:
                unique += 1
                tags[exact_hash] = card.tag
                exact_hashes.add(exact_hash)
                ofh.write(card_json)
    print(f"Deduplication of {jsonl_file} is complete.\n\tRetained: {round(unique / total, 3) * 100}% ({unique} cards)\n\tDropped: {total - unique} cards\n\tMismatched tags: {mismatch} cards\nSaved in: {outfile}")