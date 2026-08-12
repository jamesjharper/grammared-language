import sys
from unittest.mock import MagicMock


def test_generation_config_reaches_pipeline(monkeypatch):
    # The Triton backend utilities are supplied by Triton at runtime, so stub
    # them before importing the model module in this unit test.
    monkeypatch.setitem(sys.modules, "triton_python_backend_utils", MagicMock())
    monkeypatch.setitem(sys.modules, "torch", MagicMock())
    monkeypatch.setitem(sys.modules, "transformers", MagicMock())
    from grammared_language.triton.triton_transformers_model import (
        TritonTransformersPythonModel,
    )
    from grammared_language.utils.config_parser import CoEditConfig

    model = TritonTransformersPythonModel()
    model.grammared_language_model_config = CoEditConfig(
        type="coedit",
        serving_config={},
        model_inference_config={
            "temperature": 0.8,
            "do_sample": True,
            "num_beams": 2,
            "max_length": 42,
            "unrelated_setting": "ignored",
        },
    )
    from grammared_language.triton.triton_transformers_model import generation_kwargs_from_config
    model.generation_kwargs = generation_kwargs_from_config(
        model.grammared_language_model_config.model_inference_config
    )
    model.pipeline = MagicMock(return_value=[{"generated_text": "fixed"}])

    # The mocked backend response classes keep this focused on the pipeline
    # invocation performed by generate_batch.
    model.generate_batch(["Fix: text"])

    model.pipeline.assert_called_once_with(
        ["Fix: text"], temperature=0.8, do_sample=True, num_beams=2, max_length=42
    )


def test_model_init_config_is_passed_to_pipeline(monkeypatch):
    monkeypatch.setitem(sys.modules, "triton_python_backend_utils", MagicMock())
    monkeypatch.setitem(sys.modules, "torch", MagicMock())
    monkeypatch.setitem(sys.modules, "transformers", MagicMock())
    from grammared_language.triton import triton_transformers_model as module
    from grammared_language.utils.config_parser import CoEditConfig

    pipeline = MagicMock()
    monkeypatch.setattr(module.transformers, "pipeline", pipeline)
    config = CoEditConfig(
        type="coedit",
        serving_config={"pretrained_model_name_or_path": "example/model"},
        model_init_config={"batch_size": 8},
    )

    module.load_pipeline_from_config(config, device="cpu")

    assert pipeline.call_args.kwargs["batch_size"] == 8
