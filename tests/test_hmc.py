"""
Tests for the hmc module.
"""

import unittest

import pandas as pd

from sklearn import tree
from sklearn.cross_validation import train_test_split

import hmc


class TestClassHierarchy(unittest.TestCase):

    def test_count_classes(self):
        ch = hmc.load_shades_class_hierachy()
        self.assertEqual(len(ch.classes_()), 8)

    def test_count_nodes(self):
        ch = hmc.load_shades_class_hierachy()
        self.assertEqual(len(ch.nodes_()), 7)

    def test_get_parent(self):
        ch = hmc.load_shades_class_hierachy()
        self.assertEqual(ch._get_parent('black'), 'dark')

    def test_get_children(self):
        ch = hmc.load_shades_class_hierachy()
        self.assertEqual(ch._get_children('dark'), ['black', 'gray'])

    def test_get_ancestors(self):
        ch = hmc.load_shades_class_hierachy()
        self.assertEqual(ch._get_ancestors('ash'), ['gray', 'dark'])
        self.assertEqual(len(ch._get_ancestors('colors')), 0)

    def test_get_descendants(self):
        ch = hmc.load_shades_class_hierachy()
        self.assertEqual(ch._get_descendants('dark'), ['black', 'gray', 'ash', 'slate'])
        self.assertEqual(len(ch._get_descendants('slate')), 0)

    def test_add_node(self):
        ch = hmc.load_shades_class_hierachy()
        old_number = len(ch.nodes_())
        ch.add_node('additional node', ch.root)
        new_number = len(ch.nodes_())
        # Adding a node should increase node count by 1
        self.assertEqual(old_number + 1, new_number)

    def test_add_redundant_node(self):
        ch = hmc.load_shades_class_hierachy()
        ch.add_node('redundant_node', ch.root)
        old_number = len(ch.nodes_())
        ch.add_node('redundant_node', ch.root)
        new_number = len(ch.nodes_())
        # Adding a redundant node should not increase node count
        self.assertEqual(old_number, new_number)

    def test_add_root_node(self):
        ch = hmc.load_shades_class_hierachy()
        # Adding the root as a child should throw an exception
        self.assertRaises(ValueError, ch.add_node, "colors", "light")

    def test_add_dag_node(self):
        ch = hmc.load_shades_class_hierachy()
        # Adding a child with a new parent should throw an exception
        self.assertRaises(ValueError, ch.add_node, "slate", "light")


class TestDecisionTreeHierarchicalClassifier(unittest.TestCase):

    def test_fit(self):
        ch = hmc.load_shades_class_hierachy()
        X, y = hmc.load_shades_data()
        dt = hmc.DecisionTreeHierarchicalClassifier(ch)
        dt = dt.fit(X, y)
        trees_fit = True
        for stage in dt.stages:
            if 'tree' not in stage.keys():
                trees_fit = False
        # After the fit each stage should have a tree
        self.assertEqual(trees_fit, True)

    def test_predict(self):
        ch = hmc.load_shades_class_hierachy()
        X, y = hmc.load_shades_data()
        dt = hmc.DecisionTreeHierarchicalClassifier(ch)
        dt = dt.fit(X, y)
        predictions = dt.predict(X)
        # One prediction for each observation
        self.assertEqual(len(predictions), len(X))

    def test_predict_stages(self):
        ch = hmc.load_shades_class_hierachy()
        X, y = hmc.load_shades_data()
        dt = hmc.DecisionTreeHierarchicalClassifier(ch)
        dt = dt.fit(X, y)

        def row_is_hierarchical(row):
            is_hierarchical = True
            for field in range(1, len(row)):
                if row[field - 1] not in [ch._get_children(row[field]), row[field - 1]]:
                    is_hierarchical = False
            return is_hierarchical

        stage_predictions = dt._predict_stages(X)
        stage_predictions['Hierarchical'] = stage_predictions.apply(
            lambda row: row_is_hierarchical(row), axis=1)
        # Each stage of classification should descend from the previous class
        self.assertEqual(len(stage_predictions[stage_predictions['Hierarchical'] != True]), 0)

    def test_score(self):
        ch = hmc.load_shades_class_hierachy()
        X, y = hmc.load_shades_data()
        X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.50, random_state=0)
        dt = hmc.DecisionTreeHierarchicalClassifier(ch)
        dt_nonh = tree.DecisionTreeClassifier()
        dt = dt.fit(X_train, y_train)
        dt_nonh = dt_nonh.fit(X_train, y_train)
        accuracy = dt.score(X_test, y_test)
        accuracy_nonh = dt_nonh.score(X_test, y_test)
        # Hierachical classification should be at least as accurate as traditional classification
        self.assertTrue(accuracy >= accuracy_nonh)

    def test_score_before_fit(self):
        ch = hmc.load_shades_class_hierachy()
        X, y = hmc.load_shades_data()
        dt = hmc.DecisionTreeHierarchicalClassifier(ch)
        # Scoring without fitting should raise exception
        self.assertRaises(ValueError, dt.score, X, y)


if __name__ == '__main__':
    unittest.main()
