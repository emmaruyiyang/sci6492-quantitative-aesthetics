"""Machine-specific settings, read from env_config.json next to this file.

env_config.json is not in git (it may hold a token and local paths): copy
env_config_example.json to env_config.json and fill it in. Without it, the
defaults below are used (folders inside this repo).

  ROOT_MODEL_FOLDER     saved autoencoders / classifiers / RankNets
                        (default: models/ next to this file)
  ROOT_DATASETS_FOLDER  image datasets, one folder each; new datasets are
                        written here (default: src/datasets)
  ADDITIONAL_DATASET_FOLDERS  more folders of datasets, only read: listed as
                        dataset choices next to those of ROOT_DATASETS_FOLDER
                        (optional, default none)
  DEFAULT_RESULTS_FOLDER  where apps save their results; default: results/
                        next to this file
  HF_TOKEN              Hugging Face token, for gated or private models. The
                        HF_TOKEN environment variable is used when the file
                        has none.
"""

import json
import os
from pathlib import Path

HERE = Path(__file__).resolve().parent
CONFIG_FILE = HERE / "env_config.json"

_config = json.loads(CONFIG_FILE.read_text(encoding="utf-8")) if CONFIG_FILE.exists() else {}

ROOT_MODEL_FOLDER = Path(_config.get("root_model_folder") or HERE / "models")
ROOT_DATASETS_FOLDER = Path(_config.get("root_datasets_folder") or HERE.parent / "datasets")
ADDITIONAL_DATASET_FOLDERS = [Path(p) for p in _config.get("additional_dataset_folders") or []]
DEFAULT_RESULTS_FOLDER = Path(_config.get("default_results_folder") or HERE / "results")
HF_TOKEN = _config.get("hf_token") or os.environ.get("HF_TOKEN") or None

ROOT_MODEL_FOLDER.mkdir(parents=True, exist_ok=True)
