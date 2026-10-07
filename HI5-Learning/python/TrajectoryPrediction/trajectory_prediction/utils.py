# Copyright 2024, Theodor Westny. All rights reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

import os
import yaml
import importlib
from typing import Callable
from pathlib import Path
import torch


def load_config(config: str) -> dict:
    # check if file contains ".yml" extension
    if not config.endswith(".yml"):
        config += ".yml"

    # check if file exists in "configs":
    config_path = Path("trajectory_prediction/configs")

    # get all files in subdirectory
    files = [f for f in config_path.iterdir() if f.is_file()]

    # check if config is any of the files
    if not any([config in f.name for f in files]):
        raise FileNotFoundError(f"Config file {config} not found.")
    else:
        config = [f for f in files if config in f.name][0]

    with open(config) as f:
        conf = yaml.safe_load(f)

    return conf


def import_module(module_name: str) -> object:
    return importlib.import_module(module_name)


def import_from_module(module_name: str, class_name: str) -> Callable:
    module = import_module(module_name)
    return getattr(module, class_name)


def load_pretrained_model(modality: str = "UM", version: str = "V1", extra: str = "", device: str = "cpu") -> object:
    dataset = "sinD"
    add = ""
    if extra:
        add = f"_{extra}"

    conf = f"rnn_{dataset}" + add + ".yml"
    # conf = f"rnn_{dataset}_{modality}" + add + ".yml"
    config = load_config(conf)
    TorchModel = import_from_module("trajectory_prediction." + config["model"]["module"], config["model"]["class"])
    LitModel = import_from_module(
        "trajectory_prediction." + config["litmodule"]["module"], config["litmodule"]["class"]
    )

    net = TorchModel(config["model"])
    config["training"]["dataset"] = dataset
    model = LitModel(net, config["training"])

    mdl_name = f"TrajNet{add}.ckpt"
    # mdl_name = f"TrajNet-{version}-{modality}" + add + ".ckpt"
    mdl_path = os.path.join("trajectory_prediction/saved_models", dataset, mdl_name)
    ckpt = torch.load(mdl_path, map_location=device, weights_only=True)
    model.load_state_dict(ckpt["state_dict"], strict=False)

    return model
