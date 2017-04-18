"""
Tests for the hmc metrics module.
"""

import unittest

import pandas as pd

from sklearn import tree
from sklearn.cross_validation import train_test_split
from sklearn import metrics as skmetrics

import hmc
import hmc.metrics as metrics


class TestMetrics(unittest.TestCase):

    def setUp(self):
        self.ch = hmc.load_shades_class_hierachy()
        self.X, self.y = hmc.load_shades_data()
        self.X_train, self.X_test, self.y_train, self.y_test = train_test_split(
            self.X, self.y, test_size=0.50, random_state=0)
        self.dt = hmc.DecisionTreeHierarchicalClassifier(self.ch)
        self.dt_nonh = tree.DecisionTreeClassifier()
        self.dt = self.dt.fit(self.X_train, self.y_train)
        self.dt_nonh = self.dt_nonh.fit(self.X_train, self.y_train)
        self.y_pred = self.dt.predict(self.X_test)
        self.y_pred_nonh = self.dt_nonh.predict(self.X_test)

    # General Scores
    # Average accuracy
    def test_accuracy_score(self):
        accuracy = metrics.accuracy_score(self.ch, self.y_test, self.y_pred)
        accuracy_sk = skmetrics.accuracy_score(self.y_test, self.y_pred)
        # Hierachical classification should be at least as accurate as traditional classification
        self.assertTrue(accuracy >= accuracy_sk)

    # Hierarchy Precision / Recall
    # Ancestors Scores (Super Class)
    # Precision
    def test_precision_score_ancestors(self):
        precision_ancestors = metrics.precision_score_ancestors(self.ch, self.y_test, self.y_pred)
        precision_sk = skmetrics.precision_score(self.y_test, self.y_pred, average="macro")
        self.assertTrue(precision_ancestors >= precision_sk)

    # Recall
    def test_recall_score_ancestors(self):
        recall_ancestors = metrics.recall_score_ancestors(self.ch, self.y_test, self.y_pred)
        recall_sk = skmetrics.recall_score(self.y_test, self.y_pred, average="macro")
        self.assertTrue(recall_ancestors >= recall_sk)

    # Descendants Scores (Sub Class)
    # Precision
    def test_precision_score_descendants(self):
        precision_descendants = metrics.precision_score_descendants(
            self.ch, self.y_test, self.y_pred)
        precision_sk = skmetrics.precision_score(self.y_test, self.y_pred, average="macro")
        self.assertTrue(precision_descendants >= precision_sk)

    # Recall
    def test_recall_score_descendants(self):
        recall_descendants = metrics.recall_score_descendants(self.ch, self.y_test, self.y_pred)
        recall_sk = skmetrics.recall_score(self.y_test, self.y_pred, average="macro")
        self.assertTrue(recall_descendants >= recall_sk)

    # F1
    # Ancestors
    def test_f1_score_ancestors(self):
        f1_ancestors = metrics.f1_score_ancestors(self.ch, self.y_test, self.y_pred)
        f1_sk = skmetrics.f1_score(self.y_test, self.y_pred, average="macro")
        self.assertTrue(f1_ancestors >= f1_sk)

    # Descendants
    def test_f1_score_descendants(self):
        f1_descendants = metrics.f1_score_descendants(self.ch, self.y_test, self.y_pred)
        f1_sk = skmetrics.f1_score(self.y_test, self.y_pred, average="macro")
        self.assertTrue(f1_descendants >= f1_sk)


if __name__ == '__main__':
    unittest.main()
