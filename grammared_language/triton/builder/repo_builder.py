# from .triton.triton_templates import
import json
import shutil
import logging
from jinja2 import Environment, FileSystemLoader
from pathlib import Path
import yaml
from grammared_language.utils .config_parser import BaseModelConfig, ModelsConfig
from grammared_language.utils import config_parser

DEFAULT_TEMPLATE_FOLDER = str(Path(__file__).parent / "triton_templates")
GENERATED_MANIFEST_FILENAME = ".grammared-language-generated-models.json"
logger = logging.getLogger(__name__)

SUPPORTED_MODEL_TYPES = ("gector", "grammared_classifier", "coedit", "text2text")

TEMPLATE_FILES_BY_MODEL_TYPE = {
    "gector": {
        "config": "gector.config.pbtxt.jinja",
        "model": "gector.model.py.jinja"
    },
    "grammared_classifier": {
        "config": "grammared_classifier.config.pbtxt.jinja",
        "model": "grammared_classifier.model.py.jinja"
    },
    "coedit": {
        "config": "text2text.config.pbtxt.jinja",
        "model": "text2text.model.py.jinja"
    },
    "text2text": {
        "config": "text2text.config.pbtxt.jinja",
        "model": "text2text.model.py.jinja"
    }
}

# These are legacy generated-config defaults, kept outside the templates so all
# scheduling values supplied to a template have a single, explicit source.
DEFAULT_QUEUE_DELAY_MICROSECONDS_BY_MODEL_TYPE = {
    "gector": 350,
    "coedit": 750,
    "text2text": 750,
    "grammared_classifier": 100,
}

class TritonRepoBuilder:
    def __init__(self, template_folder: str|None=None):
        if template_folder is None:
            template_folder = DEFAULT_TEMPLATE_FOLDER
        self.template_folder = template_folder
        self.jina_loader = FileSystemLoader(searchpath=self.template_folder)
        self.jinja_env = Environment(loader=self.jina_loader)

    def build_model_repo(self, repo_folder: str, config: ModelsConfig|None=None, config_path:str|None=None):
        
        if not (config or config_path):
            raise ValueError("Either config or config_path must be provided.")
        
        if config is None:
            config = config_parser.load_config_from_file(config_path)

        for m in config.models:
            model_config = config.models[m]
            if model_config.type not in SUPPORTED_MODEL_TYPES:
                raise ValueError(f"Unsupported model type: {model_config.type}")

            name = m
            if model_config.serving_config.triton_model_name is not None:
                name = model_config.serving_config.triton_model_name

            self._build_model_repo(
                    name=name,
                    model_config=model_config,
                    repo_folder=repo_folder
            )

    def reconcile_model_repo(self, repo_folder: str, config: ModelsConfig):
        """Synchronize Grammared-owned model directories with ``config``.

        The manifest means a pre-existing user directory is never removed.
        Generated files are only rewritten when their rendered contents differ.
        """
        repo_path = Path(repo_folder)
        repo_path.mkdir(parents=True, exist_ok=True)
        manifest_path = repo_path / GENERATED_MANIFEST_FILENAME
        previous_names = set()
        if manifest_path.is_file():
            try:
                previous_names = set(json.loads(manifest_path.read_text()).get("models", []))
            except (OSError, json.JSONDecodeError):
                logger.warning("Ignoring invalid generated-model manifest: %s", manifest_path)

        desired = {}
        for logical_name, model_config in config.models.items():
            if model_config.type not in SUPPORTED_MODEL_TYPES:
                raise ValueError(f"Unsupported model type: {model_config.type}")
            name = model_config.serving_config.triton_model_name or logical_name
            if name in desired:
                raise ValueError(f"Duplicate Triton model name: {name}")
            desired[name] = model_config

        for stale_name in previous_names - set(desired):
            stale_path = repo_path / stale_name
            if stale_path.is_dir() and stale_path.parent.resolve() == repo_path.resolve():
                shutil.rmtree(stale_path)

        for name, model_config in desired.items():
            self._build_model_repo(name, model_config, str(repo_path))

        manifest_contents = json.dumps({"models": sorted(desired)}, indent=2) + "\n"
        self._write_if_changed(manifest_path, manifest_contents)
            
    def _build_model_repo(self, name:str, model_config:BaseModelConfig, repo_folder:str, model_version:int=1):
        model_type = model_config.type
        pretrained_model_name_or_path = model_config.serving_config.pretrained_model_name_or_path
        template_files = TEMPLATE_FILES_BY_MODEL_TYPE[model_type]
        output_folder = Path(repo_folder) / name
        model_folder = output_folder / str(model_version)
        serving_config = model_config.serving_config
        max_queue_delay_microseconds = serving_config.max_queue_delay_microseconds
        if max_queue_delay_microseconds is None:
            max_queue_delay_microseconds = DEFAULT_QUEUE_DELAY_MICROSECONDS_BY_MODEL_TYPE[
                model_type
            ]
        config_file = self.jinja_env.get_template(template_files["config"]).render({
            "model_name": name,
            "pretrained_model_name_or_path": pretrained_model_name_or_path,
            "json_model_config": json.dumps(model_config.model_dump_json())[1:-1],
            "max_batch_size": serving_config.max_batch_size,
            "preferred_batch_sizes": serving_config.preferred_batch_sizes,
            "max_queue_delay_microseconds": max_queue_delay_microseconds,
        })

        model_file = self.jinja_env.get_template(template_files["model"]).render()
        Path(output_folder).mkdir(parents=True, exist_ok=True)
        Path(model_folder).mkdir(parents=True, exist_ok=True)
        self._write_if_changed(output_folder / "config.pbtxt", config_file)
        self._write_if_changed(model_folder / "model.py", model_file)

    @staticmethod
    def _write_if_changed(path: Path, contents: str):
        if not path.is_file() or path.read_text() != contents:
            path.write_text(contents)
