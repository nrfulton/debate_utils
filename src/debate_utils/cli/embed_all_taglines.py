import argparse
from debate_utils.models import Card
from sentence_transformers import SentenceTransformer
import pickle
from typing import List, Dict
import numpy
import abc
from sklearn.metrics.pairwise import cosine_similarity

EMBEDDING_MODEL_CHOICES = ["minilm_l12_v2"]

def main():
    ap = argparse.ArgumentParser("Compute embeddings for cards")
    ap.add_argument("--model", type=str, help="The model to use.", choices=EMBEDDING_MODEL_CHOICES, required=True)
    ap.add_argument("--outfile", type=str, help="The output file.", required=True)
    args = ap.parse_args()
    cards = list(Card.from_hf_dataset())

    embeddings = None
    if args.model == "minilm_l12_v2":
        embeddings = MiniLM.embed(cards)
        for card in MiniLM.search(embeddings, cards, "nuke war leads to human extinction", 10):
            print(card)
    
    pickle.dump(embeddings, open(args.outfile, 'wb'))


class TagEmbedding(abc.ABC):
    @classmethod
    @abc.abstractmethod
    def embed(self, cards: List[Card]) -> numpy.ndarray:
        ...

    @classmethod
    @abc.abstractmethod
    def search(self, embeddings: numpy.ndarray, cards: List[Card], search_term: str, top_n=10):
        ...


class MiniLM(TagEmbedding):
    model_id = 'sentence-transformers/all-MiniLM-L12-v2'

    @classmethod
    def embed(cls, cards: List[Card]) -> numpy.ndarray:
        taglines = list(map(lambda x: x.tag, cards))
        model = SentenceTransformer(cls.model_id)
        embeddings = model.encode(taglines, show_progress_bar=True)
        return embeddings

    @classmethod
    def search(cls, embeddings: numpy.ndarray, cards: List[Card], search_text: str, top_n=10):
        key_vector = SentenceTransformer(cls.model_id).encode(search_text).reshape(1,-1)
        similarity_scores = cosine_similarity(key_vector, embeddings)[0]
        top_indices = numpy.argsort(similarity_scores)[-top_n:][::-1]
        return list(map(lambda idx: cards[idx], top_indices))