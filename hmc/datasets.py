"""
Datasets and their loaders.
"""

import random

import pandas as pd

from hmc import ClassHierarchy
from hmc import DecisionTreeHierarchicalClassifier

seeds = [
    {"node":  "dark",
        "mu_1": 0, "mu_2": 10, "mu_3": 0,
        "sigma_1": 4, "sigma_2": 4, "sigma_3": 4},
    {"node": "black",
        "mu_1": 1, "mu_2": 9, "mu_3": 1,
        "sigma_1": 3, "sigma_2": 3, "sigma_3": 3},
    {"node": "gray",
        "mu_1": 2, "mu_2": 8, "mu_3": 2,
        "sigma_1": 2, "sigma_2": 2, "sigma_3": 2},
    {"node": "ash",
        "mu_1": 3, "mu_2": 7, "mu_3": 3,
        "sigma_1": 1, "sigma_2": 1, "sigma_3": 1},
    {"node": "slate",
        "mu_1": 4, "mu_2": 8, "mu_3": 2,
        "sigma_1": 1, "sigma_2": 1, "sigma_3": 1},
    {"node": "light",
        "mu_1": 10, "mu_2": 0, "mu_3": 10,
        "sigma_1": 4, "sigma_2": 4, "sigma_3": 4},
    {"node": "white",
        "mu_1": 9, "mu_2": 1, "mu_3": 9,
        "sigma_1": 3, "sigma_2": 3, "sigma_3": 3},
]


def load_shades_class_hierachy():
    ch = ClassHierarchy("colors")
    ch.add_node("light", "colors")
    ch.add_node("dark", "colors")
    ch.add_node("white", "light")
    ch.add_node("black", "dark")
    ch.add_node("gray", "dark")
    ch.add_node("slate", "gray")
    ch.add_node("ash", "gray")
    return ch


def load_shades_data(random_seed=1):
    random.seed(random_seed)
    data_rows = []
    label_rows = []
    for seed in seeds:
        for i in range(0, int(100 + 100 * random.random())):
            data_row = {}
            data_row["a"] = random.gauss(seed["mu_1"], seed["sigma_1"])
            data_row["b"] = random.gauss(seed["mu_2"], seed["sigma_2"])
            data_row["c"] = random.gauss(seed["mu_3"], seed["sigma_3"])
            data_rows.append(data_row)
            label_row = {}
            label_row["label"] = seed["node"]
            label_rows.append(label_row)
    return pd.DataFrame(data_rows), pd.DataFrame(label_rows)
