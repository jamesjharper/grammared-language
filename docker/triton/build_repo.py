from grammared_language.triton.builder.repo_builder import TritonRepoBuilder
from grammared_language.utils.config_parser import get_config, MODEL_REPO_FOLDER
import os
import logging
from pathlib import Path

logger = logging.getLogger(__name__)
logging.basicConfig(level=logging.DEBUG)


"""
1. check if MODEL_REPO_FOLDER exists
2. check MODEL_CONFIG_PATH
3. check environment variable
4. use DEFAULT_MODEL_CONFIG_PATH
"""


def build_triton_model_repo():
    model_repo_path = os.environ.get("GRAMMARED_LANGUAGE__MODEL_REPO_FOLDER", MODEL_REPO_FOLDER)
    config = get_config()
    Path(model_repo_path).mkdir(parents=True, exist_ok=True)
    builder = TritonRepoBuilder()
    builder.reconcile_model_repo(
        repo_folder=model_repo_path,
        config=config
    )


if __name__ == "__main__":
    build_triton_model_repo()
