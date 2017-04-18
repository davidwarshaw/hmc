"""
Metrics for evaluating hierachical multi-classification performance.
"""

from __future__ import print_function
from __future__ import division

from sklearn import tree
from sklearn import metrics as skmetrics
from sklearn.utils import check_consistent_length
from sklearn.utils import column_or_1d
from sklearn.utils.multiclass import type_of_target

from itertools import chain

import numpy as np
import pandas as pd


def _check_targets_hmc(y_true, y_pred):
    check_consistent_length(y_true, y_pred)
    y_type = set([type_of_target(y_true), type_of_target(y_pred)])
    if y_type == set(["binary", "multiclass"]):
        y_type = set(["multiclass"])
    if y_type != set(["multiclass"]):
        raise ValueError("{0} is not supported".format(y_type))
    y_true = column_or_1d(y_true)
    y_pred = column_or_1d(y_pred)
    return y_true, y_pred


# General Scores
# Average accuracy
def accuracy_score(class_hierarchy, y_true, y_pred):
    y_true, y_pred = _check_targets_hmc(y_true, y_pred)
    return skmetrics.accuracy_score(y_true, y_pred)


# Hierarchy Precision / Recall
def _aggregate_class_sets(set_function, y_true, y_pred):
    intersection_sum = 0
    true_sum = 0
    predicted_sum = 0
    for true, pred in zip(y_true.tolist(), y_pred.tolist()):
        true_set = set([true] + set_function(true))
        pred_set = set([pred] + set_function(pred))
        intersection_sum += len(true_set.intersection(pred_set))
        true_sum += len(true_set)
        predicted_sum += len(pred_set)
    return (true_sum, predicted_sum, intersection_sum)


# Ancestors Scores (Super Class)
# Precision
def precision_score_ancestors(class_hierarchy, y_true, y_pred):
    y_true, y_pred = _check_targets_hmc(y_true, y_pred)
    true_sum, predicted_sum, intersection_sum = _aggregate_class_sets(
        class_hierarchy._get_ancestors, y_true, y_pred)
    return intersection_sum / predicted_sum


# Recall
def recall_score_ancestors(class_hierarchy, y_true, y_pred):
    y_true, y_pred = _check_targets_hmc(y_true, y_pred)
    true_sum, predicted_sum, intersection_sum = _aggregate_class_sets(
        class_hierarchy._get_ancestors, y_true, y_pred)
    return intersection_sum / true_sum


# Descendants Scores (Sub Class)
# Precision
def precision_score_descendants(class_hierarchy, y_true, y_pred):
    y_true, y_pred = _check_targets_hmc(y_true, y_pred)
    true_sum, predicted_sum, intersection_sum = _aggregate_class_sets(
        class_hierarchy._get_descendants, y_true, y_pred)
    return intersection_sum / predicted_sum


# Recall
def recall_score_descendants(class_hierarchy, y_true, y_pred):
    y_true, y_pred = _check_targets_hmc(y_true, y_pred)
    true_sum, predicted_sum, intersection_sum = _aggregate_class_sets(
        class_hierarchy._get_descendants, y_true, y_pred)
    return intersection_sum / true_sum


# Hierarchy Fscore
def _fbeta_score_class_sets(set_function, y_true, y_pred, beta=1):
    y_true, y_pred = _check_targets_hmc(y_true, y_pred)
    true_sum, predicted_sum, intersection_sum = _aggregate_class_sets(set_function, y_true, y_pred)
    precision = intersection_sum / predicted_sum
    recall = intersection_sum / true_sum
    return ((beta ** 2 + 1) * precision * recall) / ((beta ** 2 * precision) + recall)


def f1_score_ancestors(class_hierarchy, y_true, y_pred):
    y_true, y_pred = _check_targets_hmc(y_true, y_pred)
    return _fbeta_score_class_sets(class_hierarchy._get_ancestors, y_true, y_pred)


def f1_score_descendants(class_hierarchy, y_true, y_pred):
    y_true, y_pred = _check_targets_hmc(y_true, y_pred)
    return _fbeta_score_class_sets(class_hierarchy._get_descendants, y_true, y_pred)

# # Classification Report
# def classification_report(class_hierarchy, y_true, y_pred):
