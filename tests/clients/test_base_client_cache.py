from grammared_language.clients.base_client import BaseClient
from grammared_language.language_tool.output_models import LanguageToolRemoteResult


class RecordingClient(BaseClient):
    def __init__(self):
        super().__init__()
        self.predict_calls = []

    def _predict(self, text):
        self.predict_calls.append(text)
        if isinstance(text, list):
            return [f"corrected: {value}" for value in text]
        return f"corrected: {text}"

    def _pred_postprocess(self, original, pred, **kwargs):
        return LanguageToolRemoteResult(
            language="English", languageCode="en-US", matches=[]
        )


def test_scalar_prediction_uses_sentence_cache():
    client = RecordingClient()

    first = client.predict("A")
    second = client.predict("A")

    assert client.predict_calls == ["A"]
    assert first is second


def test_sentence_cache_uses_lru_eviction(monkeypatch):
    monkeypatch.setenv("GRAMMARED_LANGUAGE__GRPC_SERVER_CACHE_SIZE", "2")
    client = RecordingClient()

    client.predict("A")
    client.predict("B")
    client.predict("A")  # Refresh A so B becomes least recently used.
    client.predict("C")
    client.predict("B")

    assert client.predict_calls == ["A", "B", "C", "B"]


def test_fully_cached_batch_does_not_infer():
    client = RecordingClient()
    for text in ["A", "B", "C"]:
        client.predict(text)
    client.predict_calls.clear()

    results = client.predict(["A", "B", "C"])

    assert len(results) == 3
    assert client.predict_calls == []


def test_partially_cached_batch_sends_only_uncached_sentences_once():
    client = RecordingClient()
    client.predict("A")
    client.predict("C")
    client.predict_calls.clear()

    results = client.predict(["A", "B", "C", "D"])

    assert len(results) == 4
    assert client.predict_calls == [["B", "D"]]


def test_cold_batch_is_sent_as_one_model_batch():
    client = RecordingClient()

    results = client.predict(["A", "B", "C"])

    assert len(results) == 3
    assert client.predict_calls == [["A", "B", "C"]]


def test_batch_deduplicates_uncached_sentences_and_preserves_order():
    client = RecordingClient()

    results = client.predict(["A", "B", "A", "C", "B"])

    assert client.predict_calls == [["A", "B", "C"]]
    assert results[0] is results[2]
    assert results[1] is results[4]
    assert len(results) == 5


def test_batch_cardinality_mismatch_raises_before_caching():
    class ShortBatchClient(RecordingClient):
        def _predict(self, text):
            self.predict_calls.append(text)
            return ["only one"]

    client = ShortBatchClient()

    try:
        client.predict(["A", "B"])
    except RuntimeError as error:
        assert str(error) == "Batch prediction returned 1 outputs for 2 inputs"
    else:
        raise AssertionError("Expected a batch cardinality failure")

    assert client.cache_stats()["size"] == 0
