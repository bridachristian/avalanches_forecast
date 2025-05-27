# -*- coding: utf-8 -*-
"""
Created on Tue May 27 09:43:46 2025

@author: Christian
"""

import pkg_resources
import importlib
import sys

# Lista dei pacchetti che hai importato nel codice
used_packages = [
    "optuna", "scikit-learn", "numpy", "pandas", "matplotlib",
    "seaborn", "collections", "mlxtend", "imbalanced-learn", "pathlib", "shap",
    "IPython", "time", "scipy", "os", "glob", "logging", "joblib", 'json'
]

output_file = "librerie_utilizzate.txt"

with open(output_file, "w") as f:
    f.write("Librerie utilizzate nel progetto e relative versioni:\n\n")
    for pkg in used_packages:
        try:
            version = pkg_resources.get_distribution(pkg).version
            f.write(f"{pkg}=={version}\n")
        except pkg_resources.DistributionNotFound:
            f.write(f"{pkg} non trovato nel sistema.\n")

print(f"File salvato come {output_file}")
