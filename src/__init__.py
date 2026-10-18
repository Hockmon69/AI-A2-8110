"""
Musanze Harvest Intelligence Service (SWE 3513 - Assignment 2)
Author: Fajwan Chanjwok (25/28110)
Verification Code: AI-A2-8110
"""
import importlib.util
import os
import sys

__version__ = "1.0.0"
__author__ = "Fajwan Chanjwok"
__verification_code__ = "AI-A2-8110"

_src_dir = os.path.dirname(os.path.abspath(__file__))
_submodules = [
    "validation",
    "data_generator",
    "preprocessing",
    "evaluation",
    "models_regression",
    "models_classification",
    "models_clustering",
    "pipeline",
]

# Pass 1: create module objects and insert into sys.modules
_loaded_specs = []
for _name in _submodules:
    _full_name = f"src.{_name}"
    _file_path = os.path.join(_src_dir, f"{_name}.py")
    if os.path.exists(_file_path):
        _spec = importlib.util.spec_from_file_location(_full_name, _file_path)
        if _spec and _spec.loader:
            _mod = importlib.util.module_from_spec(_spec)
            sys.modules[_full_name] = _mod
            globals()[_name] = _mod
            _loaded_specs.append((_spec, _mod))

# Pass 2: execute module code in dependency order
for _spec, _mod in _loaded_specs:
    _spec.loader.exec_module(_mod)
