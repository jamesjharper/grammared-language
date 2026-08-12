from grammared_language.triton.builder.repo_builder import TritonRepoBuilder
from grammared_language.utils.config_parser import ModelsConfig


def coedit_config(name="served_model", max_batch_size=8):
    return ModelsConfig.from_dict({
        "logical_model": {
            "type": "coedit",
            "serving_config": {
                "triton_model_name": name,
                "pretrained_model_name_or_path": "example/model",
                "max_batch_size": max_batch_size,
                "preferred_batch_sizes": [1, max_batch_size],
                "max_queue_delay_microseconds": 123,
            },
        },
    })


def test_reconciliation_updates_batching_name_removals_and_preserves_unchanged(tmp_path):
    builder = TritonRepoBuilder()

    builder.reconcile_model_repo(str(tmp_path), coedit_config(max_batch_size=8))
    original = tmp_path / "served_model" / "config.pbtxt"
    initial_contents = original.read_text()
    initial_mtime = original.stat().st_mtime_ns
    assert "max_batch_size: 8" in initial_contents

    builder.reconcile_model_repo(str(tmp_path), coedit_config(max_batch_size=16))
    assert "max_batch_size: 16" in original.read_text()

    builder.reconcile_model_repo(str(tmp_path), coedit_config(name="renamed_model", max_batch_size=16))
    assert not (tmp_path / "served_model").exists()
    renamed = tmp_path / "renamed_model" / "config.pbtxt"
    assert renamed.is_file()

    unchanged_mtime = renamed.stat().st_mtime_ns
    builder.reconcile_model_repo(str(tmp_path), coedit_config(name="renamed_model", max_batch_size=16))
    assert renamed.stat().st_mtime_ns == unchanged_mtime

    builder.reconcile_model_repo(str(tmp_path), ModelsConfig.from_dict({}))
    assert not (tmp_path / "renamed_model").exists()


def test_reconciliation_does_not_delete_unmanaged_directories(tmp_path):
    user_model = tmp_path / "user_model"
    user_model.mkdir()
    (user_model / "keep.txt").write_text("user content")

    TritonRepoBuilder().reconcile_model_repo(str(tmp_path), ModelsConfig.from_dict({}))

    assert (user_model / "keep.txt").read_text() == "user content"
