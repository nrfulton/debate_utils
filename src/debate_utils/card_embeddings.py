from enum import Enum
from sentence_transformers import SentenceTransformer
from typing import Iterable, List, Optional
from debate_utils.models import Card
import numpy as np
from sklearn.manifold import TSNE
from sklearn.cluster import DBSCAN
import plotly.express as px
import plotly.io as pio
import pandas as pd

""" Embedding models supported by this cosebase. """
EmbeddingModel = Enum("SupportedEmbeddingModels", names=["mimilmv2"])


def _sf_model_id(em: EmbeddingModel) -> str:
    """ returns the sentence_transformers model id for an SF model. """
    match em:
        case EmbeddingModel.mimilmv2:
            return "sentence-transformers/all-MiniLM-L12-v2"


def _embed(em:EmbeddingModel, sentences: Iterable[str]):
    # TODO could cache the model loading component of this.
    match em:
        case EmbeddingModel.mimilmv2:
            model = SentenceTransformer(_sf_model_id(em))
            return model.encode(sentences, show_progress_bar=True)


def embed_tags(model: EmbeddingModel, cards: Iterable[Card]):
    return _embed(model, [c.tag for c in cards])


def embed_text(model: EmbeddingModel, cards: Iterable[Card]):
    return _embed(model, [c.text_plain() for c in cards])


def embed_cite(model: EmbeddingModel, cards: Iterable[Card]):
    return _embed(model, [c.cite_plain() for c in cards])


def visualize_embeddings(embeddings: np.ndarray, text_labels: Optional[Iterable[str]]):
    # Perform DBSCAN clustering
    dbscan = DBSCAN(eps=0.5, min_samples=5, n_jobs=-1)
    clusters = dbscan.fit_predict(embeddings)
    # reduce dimesnions
    tsne = TSNE(n_components=3, random_state=0)
    reduced_embeddings = tsne.fit_transform(embeddings)
    # construct a plotly plot.
    if text_labels:
        df = pd.DataFrame({
            'x': reduced_embeddings[:, 0],
            'y': reduced_embeddings[:, 1],
            'z': reduced_embeddings[:, 2],
            'cluster': clusters,
            'label': text_labels
        })
    else:
        df = pd.DataFrame({
            'x': reduced_embeddings[:, 0],
            'y': reduced_embeddings[:, 1],
            'z': reduced_embeddings[:, 2],
            'cluster': clusters,
            'label': text_labels
        })
    fig = px.scatter_3d(df, x='x', y='y', z='z', color='cluster', text='label',
                    title='t-SNE Visualization of Embeddings in 3D with DBSCAN Clusters')
    fig.update_traces(marker=dict(size=5),
                  selector=dict(mode='markers'),
                  hovertemplate='<b>%{text}</b><br><br>Cluster: %{marker.color}')
    pio.write_html(fig, file='embeddings_visualization_3d_clusters.html', auto_open=True)