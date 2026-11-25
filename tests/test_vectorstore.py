import numpy as np

from src.vectorstore import VectorStore


def _toy_store() -> VectorStore:
    store = VectorStore()
    # Three orthogonal-ish 3D vectors standing in for "topics".
    vectors = np.array(
        [
            [1.0, 0.0, 0.0],  # doc A: pure "x"
            [0.0, 1.0, 0.0],  # doc B: pure "y"
            [0.9, 0.1, 0.0],  # doc C: mostly "x", close to A
        ]
    )
    store.add(
        ids=["a", "b", "c"],
        vectors=vectors,
        documents=["doc about x", "doc about y", "doc mostly about x"],
        metadatas=[{"source": "a.md"}, {"source": "b.md"}, {"source": "c.md"}],
    )
    return store


def test_query_returns_nearest_neighbour_first():
    store = _toy_store()
    results = store.query(np.array([1.0, 0.0, 0.0]), k=3)

    assert len(results) == 3
    # Closest to pure-x should be a (identical), then c (mostly x), then b (orthogonal).
    assert [r.id for r in results] == ["a", "c", "b"]
    # scores should be descending
    assert results[0].score >= results[1].score >= results[2].score


def test_query_respects_k():
    store = _toy_store()
    results = store.query(np.array([1.0, 0.0, 0.0]), k=1)
    assert len(results) == 1
    assert results[0].id == "a"


def test_query_on_empty_store_returns_empty_list():
    store = VectorStore()
    assert store.query(np.array([1.0, 0.0, 0.0]), k=3) == []


def test_add_rejects_mismatched_lengths():
    import pytest

    store = VectorStore()
    with pytest.raises(ValueError):
        store.add(
            ids=["a", "b"],
            vectors=np.array([[1.0, 0.0]]),
            documents=["only one doc"],
            metadatas=[{"source": "a.md"}],
        )


def test_save_and_load_round_trip(tmp_path):
    store = _toy_store()
    store.save(tmp_path)

    loaded = VectorStore.load(tmp_path)
    assert len(loaded) == len(store)

    results = loaded.query(np.array([0.0, 1.0, 0.0]), k=1)
    assert results[0].id == "b"
