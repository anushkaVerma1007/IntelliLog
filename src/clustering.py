# src/clustering.py
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.cluster import KMeans

def cluster_anomalies(messages: list, n_clusters: int = 5) -> list:
    """
    Clusters log messages into N groups using TF-IDF and KMeans.

    Args:
        messages (list): A list of normalized log message strings.
        n_clusters (int): The number of clusters to create.

    Returns:
        list: A list of cluster labels corresponding to each message.
    """
    if not messages or len(messages) < n_clusters:
        return []

    # Vectorize the text data
    vectorizer = TfidfVectorizer(max_features=1000)
    X = vectorizer.fit_transform(messages)

    # Apply KMeans clustering
    kmeans = KMeans(
        n_clusters=n_clusters,
        random_state=42,
        n_init=10 # Suppress future warning
    )
    kmeans.fit(X)

    return kmeans.labels_