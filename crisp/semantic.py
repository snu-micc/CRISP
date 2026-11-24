"""Embedding clustering and cluster keywords used in formation-energy rule generation."""
from collections import Counter
import numpy as np
from sklearn.feature_extraction.text import ENGLISH_STOP_WORDS, TfidfVectorizer

def cluster_embeddings(vectors, cfg):
    """Run PCA/K-means on embedding rows, in original rule order; no embedding API call."""
    import numpy as np
    from sklearn.cluster import KMeans
    from sklearn.decomposition import PCA
    from sklearn.preprocessing import normalize

    x = np.asarray(vectors)
    d, k = cfg["pca_dimensions"], cfg["clusters"]
    if x.ndim != 2 or not np.isfinite(x).all() or x.shape[0] <= d or x.shape[1] < d or x.shape[0] < k:
        raise ValueError(f"Expected finite 2D embeddings with > {d} rows, >= {d} columns and >= {k} rows")
    if cfg["l2_before_pca"]:
        x = normalize(x)
    projected = PCA(n_components=d, random_state=cfg["random_seed"]).fit_transform(x)
    if cfg["l2_after_pca"]:
        projected = normalize(projected)
    km = KMeans(n_clusters=k, n_init=cfg["kmeans_n_init"], max_iter=cfg["kmeans_max_iter"],
                random_state=cfg["random_seed"])
    labels = km.fit_predict(projected)
    counts = Counter(map(int, labels))
    if cfg["embedding_input"] == "text":
        # The synthesizability notebook used NumPy's default argsort for tied cluster sizes.
        ids, sizes = np.unique(labels, return_counts=True)
        order = ids[np.argsort(-sizes)].tolist()
    else:
        order = sorted(counts, key=lambda j: (-counts[j], j))
    ranked = {label: f"R{i:02d}" for i, label in enumerate(order, 1)}
    distances = np.linalg.norm(projected - km.cluster_centers_[labels], axis=1)
    return [{"raw_cluster": int(i), "cluster_id": ranked[int(i)], "distance_to_centroid": float(dist)}
            for i, dist in zip(labels, distances)]

GENERIC_STOPWORDS = {
    "formation", "energy", "shear", "modulus", "inorganic", "crystal", "crystals",
    "rule", "rules", "feature", "features", "material", "materials", "important",
    "indicator", "indicators", "prediction", "predictive", "structure", "structures",
}

def top_keywords(texts: list[str], indices: np.ndarray, n: int = 12) -> str:
    vectorizer = TfidfVectorizer(
        lowercase=True,
        stop_words=sorted(set(ENGLISH_STOP_WORDS).union(GENERIC_STOPWORDS)),
        ngram_range=(1, 3),
        min_df=2,
        max_df=0.9,
        max_features=5000,
    )
    matrix = vectorizer.fit_transform(texts)
    names = np.asarray(vectorizer.get_feature_names_out())
    scores = np.asarray(matrix[indices].mean(axis=0)).ravel()
    if not np.any(scores):
        return ""
    order = scores.argsort()[::-1][:n]
    return ", ".join(names[index] for index in order if scores[index] > 0)

import re
RULE_PATTERN = re.compile(
    r"(?:^|\n)\s*(?:\*\*|[-*]\s*)?(\d{1,2})\s*[\.:\)]\s*(?:\*\*)?\s*(.+?)(?=(?:\n\s*(?:\*\*|[-*]\s*)?\d{1,2}\s*[\.:\)])|\Z)",
    flags=re.DOTALL,
)

def parse_rules(text: str) -> list[str]:
    found: list[tuple[int, str]] = []
    for match in RULE_PATTERN.finditer(text.strip()):
        idx = int(match.group(1))
        rule = re.sub(r"\s+", " ", match.group(2)).strip(" -*\t\r\n")
        if 1 <= idx <= 10 and rule:
            found.append((idx, rule))
    by_index = {idx: rule for idx, rule in found}
    if len(by_index) == 10:
        return [by_index[i] for i in range(1, 11)]
    # Some otherwise valid responses place all ten numbered items on one
    # comma-separated line (for example, ``1: ... , 2: ...``).  Parse that
    # formatting variant without changing the returned rule text or relaxing
    # the requirement that indices 1--10 are all present exactly once.
    inline_pattern = re.compile(
        r"(?:^|,\s+)(\d{1,2})\s*[\.:\)]\s*(.+?)(?=(?:,\s+\d{1,2}\s*[\.:\)]\s*)|\Z)",
        flags=re.DOTALL,
    )
    inline_found: list[tuple[int, str]] = []
    for match in inline_pattern.finditer(text.strip()):
        idx = int(match.group(1))
        rule = re.sub(r"\s+", " ", match.group(2)).strip(" -*\t\r\n")
        if 1 <= idx <= 10 and rule:
            inline_found.append((idx, rule))
    inline_by_index = {idx: rule for idx, rule in inline_found}
    if len(inline_by_index) == 10:
        return [inline_by_index[i] for i in range(1, 11)]
    # Conservative line-level fallback. Failure remains visible in the manifest.
    fallback: list[str] = []
    for line in text.splitlines():
        m = re.match(r"\s*(\d{1,2})\s*[\.:\)]\s*(.+?)\s*$", line)
        if m and 1 <= int(m.group(1)) <= 10:
            fallback.append(re.sub(r"\s+", " ", m.group(2)).strip())
    return fallback[:10]
