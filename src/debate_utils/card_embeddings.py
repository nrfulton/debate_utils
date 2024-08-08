from enum import Enum
from sentence_transformers import SentenceTransformer
from typing import Iterable, List, Optional, Any, Dict, Callable
from debate_utils.models import Card
import numpy as np
from sklearn.manifold import TSNE
from sklearn.cluster import DBSCAN
import plotly.express as px
import plotly.io as pio
import pandas as pd
import logging
import os
import hashlib
import pathlib
import pickle


""" The directory in which the embeddings cache is stored. """
EMBEDDINGS_CACHE_DIR = os.environ["HOME"] + os.sep + ".cache" + os.sep + "debate_utils" + os.sep + "card_embeddings"
pathlib.Path(EMBEDDINGS_CACHE_DIR).mkdir(exist_ok=True, parents=True)


""" Embedding models supported by this cosebase. """
EmbeddingModel = Enum("SupportedEmbeddingModels", names=["mimilmv2", "gte_large"])


def _sf_model_id(em: EmbeddingModel) -> str:
    """ returns the sentence_transformers model id for an SF model. """
    match em:
        case EmbeddingModel.mimilmv2:
            return "sentence-transformers/all-MiniLM-L12-v2"
        case EmbeddingModel.gte_large:
            return "thenlper/gte-large"


def cacheable_cards_embedding(fn: Callable):
    def wrapper(model: EmbeddingModel, cards: Iterable[Card]):
        # append the function name to the combined checksum of all cards, then md5sum that for length and use the result as our cache filename.
        combined_checksums: str = "".join([c.checksum() for c in cards]) + fn.__name__
        cache_file = EMBEDDINGS_CACHE_DIR + os.sep + hashlib.md5(combined_checksums.encode()).hexdigest() + ".pickle"
        if os.path.exists(cache_file):
            with open(cache_file, 'rb') as pfh:
                return pickle.load(pfh)
        result = fn(model, cards)
        with open(cache_file, 'wb') as ofh:
            pickle.dump(result, ofh)
        return result
    return wrapper


def _embed(em:EmbeddingModel, sentences: Iterable[str]):
    # TODO could cache the model loading component of this.
    match em:
        case EmbeddingModel.mimilmv2:
            model = SentenceTransformer(_sf_model_id(em))
            return model.encode(sentences, show_progress_bar=True)
        case EmbeddingModel.gte_large:
            model = SentenceTransformer(_sf_model_id(em))
            return model.encode(sentences, show_progress_bar=True)


@cacheable_cards_embedding
def embed_tags(model: EmbeddingModel, cards: Iterable[Card]):
    return _embed(model, [c.tag for c in cards])


@cacheable_cards_embedding
def embed_text(model: EmbeddingModel, cards: Iterable[Card]):
    return _embed(model, [c.text_plain() for c in cards])


@cacheable_cards_embedding
def embed_selected_text(model: EmbeddingModel, cards: Iterable[Card]):
    return _embed(model, [c.selected_text_plain() for c in cards])


@cacheable_cards_embedding
def embed_cite(model: EmbeddingModel, cards: Iterable[Card]):
    return _embed(model, [c.cite_plain() for c in cards])


def group_by_cluster(cluster_labels: np.ndarray, original_items_or_labels: Iterable[Any]) -> Dict[Any, Any]:
    # Sort the original cards into clusters.
    cards_by_cluster = {}
    for cluster_id, item in zip(cluster_labels, original_items_or_labels):
        if cluster_id in cards_by_cluster.keys():
            cards_by_cluster[cluster_id].append(item)
        else:
            cards_by_cluster[cluster_id] = [ item ]
    return cards_by_cluster


def downsample_randomly_but_proportionally(labels: np.ndarray, size=1_000) -> List[int]:
    """ `labels` is a list of classes associated with a dataset. For exaple, the IDs of clusters.
        This function down-samples `labels` to about `size` elements. The elements are randomly selected, with 2 major constraints:
            1. we try to select elements in a way that is approximately proportional.
            2. each label must have at least one elment.
        The return value of this function is a set of selected indices. """
    cluster_ids, cluster_counts = np.unique(labels, return_counts=True)
    print(f"There are {len(cluster_ids)} clusters")
    cluster_proportions = cluster_counts / cluster_counts.sum()
    sampled_indices = []
    for cluster_id, proportion in zip(cluster_ids, cluster_proportions):
        cluster_indices = np.where(labels == cluster_id)[0]
        # ensure that everyone cluster at least one representative, but otherwise allocate data proportionally.
        sample_size_for_this_cluster = max(1, int(size * proportion))
        sampled_indices.extend(np.random.choice(cluster_indices, size=sample_size_for_this_cluster, replace=False))
    assert abs(len(sampled_indices) - size) < size*.1, f"Expected {size} +- {size*.1} (== 10% of size) elements but found {len(sampled_indices)}."
    return sampled_indices # result may be slighly more or less than size.


def _debug_print_grouped_clusters(cluster_labels: np.ndarray, original_items_or_labels: Iterable[Any]) -> None:
    groups = group_by_cluster(cluster_labels, original_items_or_labels)
    for cluster_id in groups.keys():
        print(f"## Cluster {cluster_id}:")
        for item in groups[cluster_id]:
            print(f"\t - {item}")

def visualize_embeddings(embeddings: np.ndarray, text_labels: Optional[Iterable[str]]):
    logging.getLogger().debug("About to DBSCAN+t-SNE+plot. This could take a while. I'll let you know when each phase is done.")
    # Perform DBSCAN clustering
    dbscan = DBSCAN(eps=0.5, min_samples=5, n_jobs=-1)
    clusters = dbscan.fit_predict(embeddings)
    logging.getLogger().debug("Finished DBSCAN clustering.")
    # reduce dimesnions to 3
    tsne = TSNE(n_components=3, random_state=0, verbose=True)
    reduced_embeddings = tsne.fit_transform(embeddings)
    logging.getLogger().debug("Finished t-SNE dimensionality reduction to 3D")
    # downsample data randomly, but insist on a roughly proporational set of representatives from each cluster.
    if len(reduced_embeddings) > 1000:
        indices = downsample_randomly_but_proportionally(clusters, size=1000)
        logging.getLogger().debug(f"Selected {len(indices)} indices:\n{indices}")
        reduced_embeddings = reduced_embeddings[indices]
        clusters = clusters[indices]
        text_labels = [text_labels[i] for i in indices]
        logging.getLogger().debug(f"Selected 1K datapoints at random, and reduced the size of:\n\tembeddings to: {len(reduced_embeddings)}\n\tClusters to: {len(clusters)}\n\ttext labels: {len(text_labels)}")
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
            'cluster': clusters
        })
    
    # render the plots.
    fig = px.scatter_3d(df, x='x', y='y', z='z', color='cluster', hover_data='label',
                    title='Clustered Embeddings of Taglines in the Natioanl Debate Coaches Association Open Evidence Project, 2024-2025')
    fig.update_traces(marker=dict(size=5),
                  selector=dict(mode='markers'))
    pio.write_html(fig, file='embeddings_visualization_3d_clusters.html', auto_open=True)

    logging.getLogger().debug("finished rendering the 3D plot. Starting on the 2D plot.")

    fig2 = px.scatter(df, x='x', y='y', color='cluster', hover_data='label', render_mode='webgl',
                    title='Clustered Embeddings of Taglines in the Natioanl Debate Coaches Association Open Evidence Project, 2024-2025')
    fig2.update_traces(marker=dict(size=5),
                  selector=dict(mode='markers'))
    pio.write_html(fig2, file='embeddings_visualization_2d_clusters.html', auto_open=True)