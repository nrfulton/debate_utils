"""
Usage:

from debate_utils.card_retreival import CardRetreivalDatabase
db = CardRetreivalDatabase()

"""
import sqlite3
import sqlite_vec
from typing import List
from debate_utils.models import Card
from sentence_transformers import SentenceTransformer
import struct

# from https://github.com/asg017/sqlite-vec/blob/main/examples/python-recipes/openai-sample.py
def serialize(vector: List[float]) -> bytes:
    """serializes a list of floats into a compact "raw bytes" format"""
    return struct.pack("%sf" % len(vector), *vector)

class CardRetreivalDatabase():
    MODEL_DIMENSIONS = {
        "sentence-transformers/all-MiniLM-L12-v2": 384
    }

    model: str
    cards: List[Card]
    db: sqlite3.Connection

    def __init__(self, model="sentence-transformers/all-MiniLM-L12-v2", in_memory=True):
        self.model = model
        if self.model != "sentence-transformers/all-MiniLM-L12-v2":
            raise NotImplementedError("TODO: copy over embedding setup from other files...")
        if not in_memory:
            raise NotImplementedError()
        self.cards = list(Card.from_hf_dataset())
        self.db = self._setup_in_memory_embedding_db()
    
    def _setup_in_memory_embedding_db(self):
        db = sqlite3.connect(":memory:")
        db.enable_load_extension(True)
        sqlite_vec.load(db)
        db.enable_load_extension(False)
        db.execute(
            """
                CREATE TABLE tags(
                id INTEGER PRIMARY KEY,
                tag TEXT
                );
            """
        )
        with db:
            for i, card in enumerate(self.cards):
                db.execute("INSERT INTO tags(id, tag) VALUES(?, ?)", [i, card.tag])
        db.execute(
            f"""
                CREATE VIRTUAL TABLE vec_tags USING vec0(
                id INTEGER PRIMARY KEY,
                tag_embedding FLOAT[{self.MODEL_DIMENSIONS[self.model]}]
                );
            """
        )
        with db:
            tag_rows = db.execute("SELECT id, tag FROM tags").fetchall()
            embeddings = SentenceTransformer(self.model).encode([row[1] for row in tag_rows], show_progress_bar=True)
            for (id, _), embedding in zip(tag_rows, embeddings):
                db.execute("INSERT INTO vec_tags(id, tag_embedding) VALUES(?, ?)", [id, serialize(embedding)])
        return db
    
    def search_for_tagline(self, text: str, top_n=1):
        embedding = SentenceTransformer(self.model).encode(text)
        results = self.db.execute("SELECT vec_tags.id, distance, tag FROM vec_tags LEFT JOIN tags ON tags.id = vec_tags.id WHERE tag_embedding MATCH ? AND k = ? ORDER BY distance", [serialize(embedding), top_n]).fetchall()
        return [(self.cards[i], score) for (i, score, tag) in results]
