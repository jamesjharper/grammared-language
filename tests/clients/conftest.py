"""Shared fixtures for client unit tests."""

import pytest


@pytest.fixture(autouse=True)
def isolate_gector_unit_tests_from_verb_vocab_io(request, monkeypatch):
    """Keep GectorClient unit tests hermetic after verb-vocab provisioning moved into ensure_verb_dict.

    The older tests mock ``load_verb_dict`` directly. GectorClient now calls
    ``ensure_verb_dict`` first, which validates the path before reaching that
    lower-level loader. For the GectorClient unit-test class, replace the new
    provisioning boundary instead: no filesystem access and no network
    download should be required by these tests.
    """
    test_class = getattr(request.node, "cls", None)
    if test_class is None or test_class.__name__ != "TestGectorClientUnit":
        return

    module_name = getattr(request.node.module, "__name__", "")
    if not module_name.endswith("test_gector_client"):
        return

    from grammared_language.clients import gector_client

    monkeypatch.setattr(
        gector_client,
        "ensure_verb_dict",
        lambda *args, **kwargs: ({}, {}),
    )
