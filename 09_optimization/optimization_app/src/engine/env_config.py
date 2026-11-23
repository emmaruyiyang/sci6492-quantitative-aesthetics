"""Machine-specific settings, read from env_config.json in the project root.

env_config.json is not in git (it holds a token and local paths): copy
env_config_example.json to env_config.json and fill it in.

  ROOT_MODEL_FOLDER     saved autoencoders / classifiers (custom_models)
  ROOT_DATASETS_FOLDER  image datasets, one folder each; new datasets are
                        written here
  ADDITIONAL_DATASET_FOLDERS  more folders of datasets, only read: listed as
                        dataset choices next to those of ROOT_DATASETS_FOLDER
                        (optional, default none)
  DEFAULT_RESULTS_FOLDER  where apps save their results (e.g. the optimization
                        app's snapshots); default: results/ in the project root
  HF_TOKEN              Hugging Face token, for gated or private models. The
                        HF_TOKEN environment variable is used when the file
                        has none.

optimization_app.py puts src/engine on the path, so anything can `import env_config`.
"""

import json
import os
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[2]
CONFIG_FILE = PROJECT_ROOT / "env_config.json"
EXAMPLE_FILE = PROJECT_ROOT / "env_config_example.json"

if not CONFIG_FILE.exists():
    raise FileNotFoundError(f"{CONFIG_FILE} is missing: copy {EXAMPLE_FILE.name} to {CONFIG_FILE.name} and fill it in")

_config = json.loads(CONFIG_FILE.read_text(encoding="utf-8"))

ROOT_MODEL_FOLDER = Path(_config["root_model_folder"])
ROOT_DATASETS_FOLDER = Path(_config["root_datasets_folder"])
ADDITIONAL_DATASET_FOLDERS = [Path(p) for p in _config.get("additional_dataset_folders") or []]
DEFAULT_RESULTS_FOLDER = Path(_config.get("default_results_folder") or PROJECT_ROOT / "results")
HF_TOKEN = _config.get("hf_token") or os.environ.get("HF_TOKEN") or None
