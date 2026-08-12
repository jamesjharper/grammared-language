"""Hermetic regression tests for gRPC batching and client lifecycle."""

import threading
from unittest.mock import Mock

import pytest

from api.src import grpc_server
from grammared_language.api.grpc_gen import ml_server_pb2
from grammared_language.language_tool.output_models import (
    LanguageToolRemoteResult,
    Match,
)


def result_for(text):
    return LanguageToolRemoteResult(
        language="English",
        languageCode="en-US",
        matches=[Match(offset=0, length=1, message=text, shortMessage=text)],
    )


class RecordingClient:
    def __init__(self, label=""):
        self.label = label
        self.calls = []
        self.closed = False

    def predict_with_merge(self, texts):
        self.calls.append(texts)
        if isinstance(texts, str):
            return result_for(self.label or texts)
        return [result_for(self.label or text) for text in texts]

    def close(self):
        self.closed = True


@pytest.fixture(autouse=True)
def reset_server_state():
    with grpc_server.client_condition:
        grpc_server.correction_multi_client = None
        grpc_server._client_inflight.clear()
        grpc_server._model_generation = 0
    grpc_server.analyze_cache_store.clear()
    grpc_server.process_cache_store.clear()
    grpc_server.match_cache_store.clear()
    grpc_server.match_anylized_cache_store.clear()
    yield


def process_request(*texts):
    return ml_server_pb2.ProcessRequest(
        sentences=[ml_server_pb2.AnalyzedSentence(text=text) for text in texts],
        options=ml_server_pb2.ProcessingOptions(language="en-US"),
    )


def descriptions(response):
    return [match.matchDescription for match in response.matches]


def test_process_batches_unique_misses_and_preserves_order():
    client = RecordingClient()
    grpc_server.activate_client(client)
    grpc_server.process_cache_store.add(
        grpc_server._model_cache_key(grpc_server._model_generation, "cached"),
        result_for("cached"),
    )

    response = grpc_server.ProcessingServerServicer().Process(
        process_request("cached", "first", "second", "cached", "first"), Mock()
    )

    assert client.calls == [["first", "second"]]
    assert descriptions(response) == ["cached", "first", "second", "cached", "first"]
    cache_key = grpc_server._model_cache_key(grpc_server._model_generation, "first")
    assert grpc_server.process_cache_store.contains(cache_key)


def test_process_uses_cached_entries_without_inference():
    client = RecordingClient()
    grpc_server.activate_client(client)
    key = grpc_server._model_cache_key(grpc_server._model_generation, "cached")
    grpc_server.process_cache_store.add(key, result_for("cached"))

    response = grpc_server.ProcessingServerServicer().Process(
        process_request("cached", "cached"), Mock()
    )

    assert client.calls == []
    assert descriptions(response) == ["cached", "cached"]


def test_concurrent_process_calls_are_not_serialized_by_client_lock():
    both_calls_started = threading.Barrier(2)

    class BlockingClient(RecordingClient):
        def predict_with_merge(self, texts):
            self.calls.append(texts)
            both_calls_started.wait(timeout=2)
            return [result_for(text) for text in texts]

    client = BlockingClient()
    grpc_server.activate_client(client)
    servicer = grpc_server.ProcessingServerServicer()
    failures = []

    def process(text):
        try:
            servicer.Process(process_request(text), Mock())
        except Exception as error:  # pragma: no cover - assertion below reports it
            failures.append(error)

    first = threading.Thread(target=process, args=("first",))
    second = threading.Thread(target=process, args=("second",))
    first.start()
    second.start()
    first.join(timeout=3)
    second.join(timeout=3)

    assert not first.is_alive()
    assert not second.is_alive()
    assert failures == []
    assert client.calls == [["first"], ["second"]]


def test_reload_swaps_client_and_retires_old_client_outside_lock():
    old_started = threading.Event()
    allow_old_finish = threading.Event()

    class BlockingOldClient(RecordingClient):
        def predict_with_merge(self, texts):
            self.calls.append(texts)
            old_started.set()
            assert allow_old_finish.wait(timeout=3)
            return [result_for(text) for text in texts]

    old_client = BlockingOldClient()
    new_client = RecordingClient("new")
    grpc_server.activate_client(old_client)
    servicer = grpc_server.ProcessingServerServicer()
    request_thread = threading.Thread(
        target=lambda: servicer.Process(process_request("old"), Mock())
    )
    request_thread.start()
    assert old_started.wait(timeout=1)

    reload_thread = threading.Thread(target=grpc_server.activate_client, args=(new_client,))
    reload_thread.start()

    # The old lease keeps retirement waiting, but the newly active client is
    # available immediately because the lifecycle lock is not held while waiting.
    response = servicer.Process(process_request("new request"), Mock())
    assert descriptions(response) == ["new"]
    assert new_client.calls == [["new request"]]
    assert not old_client.closed

    allow_old_finish.set()
    request_thread.join(timeout=3)
    reload_thread.join(timeout=3)
    assert not request_thread.is_alive()
    assert not reload_thread.is_alive()
    assert old_client.closed


def test_reload_invalidates_grammar_result_caches():
    old_client = RecordingClient("old")
    new_client = RecordingClient("new")
    servicer = grpc_server.ProcessingServerServicer()
    grpc_server.activate_client(old_client)

    assert descriptions(servicer.Process(process_request("same"), Mock())) == ["old"]
    assert old_client.calls == [["same"]]

    grpc_server.activate_client(new_client)
    response = servicer.Process(process_request("same"), Mock())

    assert descriptions(response) == ["new"]
    assert new_client.calls == [["same"]]
    assert old_client.closed


def test_simple_cache_store_clear():
    cache = grpc_server.SimpleCacheStore()
    cache.add("key", "value")
    cache.clear()
    assert not cache.contains("key")
