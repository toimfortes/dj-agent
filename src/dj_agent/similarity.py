"""Track similarity engine — find tracks that sound alike.

Uses audio feature vectors (MFCCs, chroma, spectral) for similarity
scoring.  When Essentia is available, uses MusiCNN embeddings for
higher-quality similarity.
"""

from __future__ import annotations

from pathlib import Path

import librosa
import numpy as np

from .types import TrackInfo


def compute_feature_vector(path: str | Path, method: str = "auto") -> np.ndarray:
    """Compute a feature vector for a track.

    Parameters
    ----------
    method : "auto" (CLAP if available, else librosa), "clap", or "librosa"

    Returns a 1-D numpy array suitable for cosine similarity.
    CLAP produces a 512-dim semantic embedding (captures "vibe").
    librosa produces a 62-dim timbral feature vector (MFCC/chroma/spectral).
    """
    if method == "auto":
        try:
            return _clap_embedding(path)
        except ImportError:
            pass
        return _librosa_features(path)
    elif method == "clap":
        return _clap_embedding(path)
    else:
        return _librosa_features(path)


def _clap_embedding(path: str | Path) -> np.ndarray:
    """Compute a 512-dim CLAP semantic embedding for similarity.

    Captures "vibe" — mood, genre, energy, cultural context — not just timbre.
    Requires: pip install laion-clap
    """
    import laion_clap  # type: ignore[import-untyped]

    model = laion_clap.CLAP_Module(enable_fusion=False, amodel="HTSAT-base")
    model.load_ckpt(ckpt="music_audioset_epoch_15_esc_90.14.pt")

    embed = model.get_audio_embedding_from_filelist(
        x=[str(path)], use_tensor=False,
    )
    return embed[0].astype(np.float32)


def _librosa_features(path: str | Path) -> np.ndarray:
    """Compute a 62-dim timbral feature vector using librosa.

    Features: MFCCs (mean+std), chroma (mean), spectral contrast (mean),
    spectral centroid (mean), spectral rolloff (mean), zero crossing rate.
    """
    y, sr = librosa.load(str(path), sr=22050, mono=True, duration=60)

    features: list[float] = []

    # MFCCs (20 coefficients × 2 stats = 40 dims)
    mfcc = librosa.feature.mfcc(y=y, sr=sr, n_mfcc=20)
    features.extend(np.mean(mfcc, axis=1).tolist())
    features.extend(np.std(mfcc, axis=1).tolist())

    # Chroma (12 dims)
    chroma = librosa.feature.chroma_stft(y=y, sr=sr)
    features.extend(np.mean(chroma, axis=1).tolist())

    # Spectral contrast (7 dims)
    contrast = librosa.feature.spectral_contrast(y=y, sr=sr)
    features.extend(np.mean(contrast, axis=1).tolist())

    # Spectral centroid, rolloff, ZCR (3 dims)
    features.append(float(np.mean(librosa.feature.spectral_centroid(y=y, sr=sr))))
    features.append(float(np.mean(librosa.feature.spectral_rolloff(y=y, sr=sr))))
    features.append(float(np.mean(librosa.feature.zero_crossing_rate(y=y))))

    return np.array(features, dtype=np.float32)


def cosine_similarity(a: np.ndarray, b: np.ndarray) -> float:
    """Cosine similarity between two vectors (0 = orthogonal, 1 = identical)."""
    norm_a = np.linalg.norm(a)
    norm_b = np.linalg.norm(b)
    if norm_a == 0 or norm_b == 0:
        return 0.0
    return float(np.dot(a, b) / (norm_a * norm_b))


def find_similar(
    target_vector: np.ndarray,
    library_vectors: dict[str, np.ndarray],
    top_k: int = 10,
    exclude_id: str | None = None,
) -> list[tuple[str, float]]:
    """Find the most similar tracks by cosine similarity.

    Uses FAISS for fast approximate nearest-neighbor search if available
    (handles 100k+ tracks in milliseconds). Falls back to brute-force
    for small libraries or when FAISS is not installed.

    Parameters
    ----------
    target_vector : 1-D array from compute_feature_vector
    library_vectors : dict of {content_id: feature_vector}
    top_k : number of results
    exclude_id : content ID to exclude (the target track itself)

    Returns list of (content_id, similarity_score) sorted by score desc.
    """
    # Filter out excluded ID
    ids = [cid for cid in library_vectors if cid != exclude_id]
    if not ids:
        return []

    # Try FAISS for large libraries
    if len(ids) > 500:
        try:
            return _find_similar_faiss(target_vector, library_vectors, ids, top_k)
        except ImportError:
            pass

    # Brute-force fallback (fine for <500 tracks)
    scores: list[tuple[str, float]] = []
    for cid in ids:
        sim = cosine_similarity(target_vector, library_vectors[cid])
        scores.append((cid, sim))

    scores.sort(key=lambda x: x[1], reverse=True)
    return scores[:top_k]


def _find_similar_faiss(
    target: np.ndarray,
    library: dict[str, np.ndarray],
    ids: list[str],
    top_k: int,
) -> list[tuple[str, float]]:
    """FAISS-accelerated similarity search using inner product (cosine)."""
    import faiss  # type: ignore[import-untyped]

    # Build matrix and normalize for cosine similarity via inner product
    vectors = np.array([library[cid] for cid in ids], dtype=np.float32)
    norms = np.linalg.norm(vectors, axis=1, keepdims=True)
    norms[norms == 0] = 1.0
    vectors /= norms

    target_norm = target.astype(np.float32).reshape(1, -1)
    t_norm = np.linalg.norm(target_norm)
    if t_norm > 0:
        target_norm /= t_norm

    dim = vectors.shape[1]
    index = faiss.IndexFlatIP(dim)  # inner product = cosine on normalized vectors
    index.add(vectors)

    k = min(top_k, len(ids))
    distances, indices = index.search(target_norm, k)

    return [(ids[idx], float(dist)) for dist, idx in zip(distances[0], indices[0]) if idx >= 0]


def build_embedding_cache(
    tracks: list[TrackInfo],
    cache_path: str | Path | None = None,
) -> dict[str, np.ndarray]:
    """Compute feature vectors for all tracks in a library.

    Optionally saves to disk as a .npz file for fast reloading.
    """
    embeddings: dict[str, np.ndarray] = {}

    for t in tracks:
        p = Path(t.path)
        if not p.exists():
            continue
        try:
            vec = compute_feature_vector(p)
            embeddings[t.db_content_id] = vec
        except Exception:
            continue

    if cache_path:
        cache_path = Path(cache_path)
        cache_path.parent.mkdir(parents=True, exist_ok=True)
        np.savez_compressed(
            str(cache_path),
            ids=list(embeddings.keys()),
            vectors=np.array(list(embeddings.values())),
        )

    return embeddings


def load_embedding_cache(cache_path: str | Path) -> dict[str, np.ndarray]:
    """Load pre-computed embeddings from a .npz file."""
    data = np.load(str(cache_path), allow_pickle=True)
    ids = data["ids"]
    vectors = data["vectors"]
    if len(ids) != len(vectors):
        raise ValueError(
            f"Embedding cache corrupted: {len(ids)} ids but {len(vectors)} vectors"
        )
    return {str(cid): vec for cid, vec in zip(ids, vectors)}
