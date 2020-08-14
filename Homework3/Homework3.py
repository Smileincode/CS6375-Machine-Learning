3# decision_tree.py
# ---------
# Licensing Information:  You are free to use or extend these projects for
# personal and educational purposes provided that (1) you do not distribute
# or publish solutions, (2) you retain this notice, and (3) you provide clear
# attribution to UT Dallas, including a link to http://cs.utdallas.edu.
#
# This file is part of Homework 3 for CS6375: Machine Learning.
# Gautam Kunapuli (gautam.kunapuli@utdallas.edu)
# Sriraam Natarajan (sriraam.natarajan@utdallas.edu),
#
#
# INSTRUCTIONS:
# ------------
# 1. This file contains a skeleton for implementing the ID3 algorithm for
# Decision Trees. Insert your code into the various functions that have the
# comment "INSERT YOUR CODE HERE".
#
# 2. Do NOT modify the classes or functions that have the comment "DO NOT
# MODIFY THIS FUNCTION".
#
# 3. Do not modify the function headers for ANY of the functions.
#
# 4. You may add any other helper functions you feel you may need to print,
# visualize, test, or save the data and results. However, you MAY NOT utilize
# the package scikit-learn OR ANY OTHER machine learning package in THIS file.

import numpy as np
import os
import graphviz
import math
import matplotlib.pyplot as plt
from sklearn.metrics import confusion_matrix
from sklearn.tree import DecisionTreeClassifier
from sklearn.model_selection import train_test_split

def partition(x):
    """
    Partition the column vector x into subsets indexed by its unique values (v1, ... vk)

    Returns a dictionary of the form
    { v1: indices of x == v1,
      v2: indices of x == v2,
      ...
      vk: indices of x == vk }, where [v1, ... vk] are all the unique values in the vector z.
    """
    subsets = dict()
    for i in range(len(x)):
        # print(type(x))
        # print(np.shape(x))
        if x[i, 0] not in subsets:
            # print(type(x[i]))
            subsets[x[i, 0]] = []
        subsets[x[i, 0]].append(i)
    return subsets
    # raise Exception('Function not yet implemented!')


def entropy(y):
    """
    Compute the entropy of a vector y by considering the counts of the unique values (v1, ... vk), in z

    Returns the entropy of z: H(z) = p(z=v1) log2(p(z=v1)) + ... + p(z=vk) log2(p(z=vk))
    """
    counts = dict()
    ans = 0.0
    for i in y:
        if i[0] not in counts:
            counts[i[0]] = 1
        else:
            counts[i[0]] += 1
    for j in counts:
        ans += -(1.0 * counts.get(j) / len(y)) * math.log(1.0 * counts.get(j) / len(y), 2)
    return ans
    # raise Exception('Function not yet implemented!')


def mutual_information(x, y):
    """
    Compute the mutual information between a data column (x) and the labels (y). The data column is a single attribute
    over all the examples (n x 1). Mutual information is the difference between the entropy BEFORE the split set, and
    the weighted-average entropy of EACH possible split.

    Returns the mutual information: I(x, y) = H(y) - H(y | x)
    """
    subsets = partition(x)
    count_y = dict()
    entropy_yx = 0.0
    for i in subsets:
        px = 1.0 * len(subsets[i]) / len(x)
        count_y.clear()
        for j in subsets[i]:
            if y[j, 0] not in count_y:
                count_y[y[j, 0]] = 1
            else:
                count_y[y[j, 0]] += 1
        pyx = 0.0
        for j in count_y:
            pyx += -(1.0 * count_y.get(j) / len(subsets[i])) * math.log(1.0 * count_y.get(j) / len(subsets[i]), 2)
        entropy_yx += px * pyx
    return entropy(y) - entropy_yx


def normalization(x, value):
    # print(type(x))
    # print(np.shape(x))
    for i in range(len(x)):
        if x[i, 0] != value:
            x[i, 0] = -value
    return x


def id3(x, y, attribute_value_pairs=None, depth=0, max_depth=5):
    """
    Implements the classical ID3 algorithm given training data (x), training labels (y) and an array of
    attribute-value pairs to consider. This is a recursive algorithm that depends on three termination conditions
        1. If the entire set of labels (y) is pure (all y = only 0 or only 1), then return that label
        2. If the set of attribute-value pairs is empty (there is nothing to split on), then return the most common
           value of y (majority label)
        3. If the max_depth is reached (pre-pruning bias), then return the most common value of y (majority label)
    Otherwise the algorithm selects the next best attribute-value pair using INFORMATION GAIN as the splitting criterion
    and partitions the data set based on the values of that attribute before the next recursive call to ID3.

    The tree we learn is a BINARY tree, which means that every node has only two branches. The splitting criterion has
    to be chosen from among all possible attribute-value pairs. That is, for a problem with two features/attributes x1
    (taking values a, b, c) and x2 (taking values d, e), the initial attribute value pair list is a list of all pairs of
    attributes with their corresponding values:
    [(x1, a),
     (x1, b),
     (x1, c),
     (x2, d),
     (x2, e)]
     If we select (x2, d) as the best attribute-value pair, then the new decision node becomes: [ (x2 == d)? ] and
     the attribute-value pair (x2, d) is removed from the list of attribute_value_pairs.

    The tree is stored as a nested dictionary, where each entry is of the form
                    (attribute_index, attribute_value, True/False): subtree
    * The (attribute_index, attribute_value) determines the splitting criterion of the current node. For example, (4, 2)
    indicates that we test if (x4 == 2) at the current node.
    * The subtree itself can be nested dictionary, or a single label (leaf node).
    * Leaf nodes are (majority) class labels

    Returns a decision tree represented as a nested dictionary, for example
    {(4, 1, False):
        {(0, 1, False):
            {(1, 1, False): 1,
             (1, 1, True): 0},
         (0, 1, True):
            {(1, 1, False): 0,
             (1, 1, True): 1}},
     (4, 1, True): 1}
    """
    if depth == 0:
        y = np.reshape(y, (len(y), 1))

    # print(type(y))
    # print(np.shape(y))
    count = dict()
    for i in y:
        if i[0] not in count:
            count[i[0]] =1
        else:
            count[i[0]] += 1
    keys = list(count.keys())
    # if len(keys) == 0:
    #     return 0
    if len(keys) == 1:
        return keys[0]
    # print(x)
    # print(y)
    # print(keys)
    label_one = keys[1]
    label_zero = keys[0]
    count_one = count.get(keys[1])
    count_zero = count.get(keys[0])
    # label_y = return_label(y, label_one, count_one, label_zero, count_zero)
    if depth == max_depth or (depth > 0 and attribute_value_pairs is None):
        if count_one > count_zero:
            return label_one
        else:
            return label_zero

    root = dict()
    # print(type(x))
    # print(np.shape(x))
    if attribute_value_pairs is None:
        attribute_value_pairs = []
        subsets = dict()
        for i in range(1, len(x[0]) + 1):
            subsets.clear()
            subsets = partition(np.reshape(x[:, i - 1], (len(x), 1)))
            for j in subsets:
                attribute_value_pairs.append((i, j))

    best_attribute = 0
    attribute_value = 0
    best_mutual_info = -1
    best_normalization = []
    for pair in attribute_value_pairs:
        # print(type(pair))
        # print(type(pair[0]))
        # print(type(np.reshape(x[:, pair[0] - 1], (len(x), 1))))
        # print(type(x))
        # print(x)
        # print("depth=", depth)
        # print(type(np.reshape(x[:, 1], (len(x), 1))))
        # print(pair)
        # print(np.shape(np.reshape(x[:, pair[0] - 1], (len(x), 1))))
        slice_x = (x[:, pair[0] - 1]).copy()
        column_x = np.reshape(slice_x, (len(x), 1))
        curr_normalization = normalization(column_x, pair[1])
        curr_mutual_info = mutual_information(curr_normalization, y)
        if curr_mutual_info > best_mutual_info:
            best_attribute = pair[0]
            attribute_value = pair[1]
            best_mutual_info = curr_mutual_info
            best_normalization = curr_normalization

    # print(type(x))
    # print(type(x[0]))
    # print(best_attribute)
    # print(attribute_value)
    if best_attribute != 0:
        attribute_value_pairs.remove((best_attribute, attribute_value))

    # print(type(best_normalization))
    #
    # print(np.shape(best_normalization))

    false_example_x = []
    false_example_y = []
    true_example_x = []
    true_example_y = []
    for i in range(len(best_normalization)):
        if best_normalization[i][0] != attribute_value:
            # print(np.shape(x[i, :]))
            # print(np.shape(y[i, :]))
            false_example_x.append(x[i, :])
            false_example_y.append(y[i, :])
        else:
            true_example_x.append(x[i, :])
            true_example_y.append(y[i, :])

    if len(false_example_x) == 0 or len(true_example_x) == 0:
        if count_one > count_zero:
            return label_one
        else:
            return label_zero

    root[(best_attribute, attribute_value, True)] = id3(np.asarray(true_example_x), np.asarray(true_example_y),
                                                        attribute_value_pairs, depth + 1, max_depth)
    root[(best_attribute, attribute_value, False)] = id3(np.asarray(false_example_x), np.asarray(false_example_y),
                                                         attribute_value_pairs, depth + 1, max_depth)
    # root[(best_attribute, attribute_value, True)] = id3(np.asarray(true_example_x), np.asarray(true_example_y),
    #                                                      attribute_value_pairs, depth + 1, max_depth)

    return root


def predict_example(x, tree):
    """
    Predicts the classification label for a single example x using tree by recursively descending the tree until
    a label/leaf node is reached.

    Returns the predicted label of x according to tree
    """
    if type(tree) is dict:
        for i in tree:
            if x[i[0] - 1] == i[1]:
                if i[2] is True:
                    return predict_example(x, tree[i])
            else:
                if i[2] is False:
                    return predict_example(x, tree[i])
    else:
        return tree


def compute_error(y_true, y_pred):
    """
    Computes the average error between the true labels (y_true) and the predicted labels (y_pred)

    Returns the error = (1/n) * sum(y_true != y_pred)
    """
    n = len(y_true)
    y_true = np.reshape(y_true, (n, 1))
    y_pred = np.reshape(y_pred, (n, 1))
    sum = 0.0
    for i in range(n):
        if y_true[i, 0] != y_pred[i, 0]:
            sum += 1
    return sum / n


def pretty_print(tree, depth=0):
    """
    Pretty prints the decision tree to the console. Use print(tree) to print the raw nested dictionary representation
    DO NOT MODIFY THIS FUNCTION!
    """
    if depth == 0:
        print('TREE')

    for index, split_criterion in enumerate(tree):
        sub_trees = tree[split_criterion]

        # Print the current node: split criterion
        print('|\t' * depth, end='')
        print('+-- [SPLIT: x{0} = {1} {2}]'.format(split_criterion[0], split_criterion[1], split_criterion[2]))

        # Print the children
        if type(sub_trees) is dict:
            pretty_print(sub_trees, depth + 1)
        else:
            print('|\t' * (depth + 1), end='')
            print('+-- [LABEL = {0}]'.format(sub_trees))


def render_dot_file(dot_string, save_file, image_format='png'):
    """
    Uses GraphViz to render a dot file. The dot file can be generated using
        * sklearn.tree.export_graphviz()' for decision trees produced by scikit-learn
        * to_graphviz() (function is in this file) for decision trees produced by  your code.
    DO NOT MODIFY THIS FUNCTION!
    """
    if type(dot_string).__name__ != 'str':
        raise TypeError('visualize() requires a string representation of a decision tree.\nUse tree.export_graphviz()'
                        'for decision trees produced by scikit-learn and to_graphviz() for decision trees produced by'
                        'your code.\n')

    # Set path to your GraphViz executable here
    os.environ["PATH"] += os.pathsep + 'C:/Program Files (x86)/Graphviz2.38/bin/'
    graph = graphviz.Source(dot_string)
    graph.format = image_format
    graph.render(save_file, view=True)


def to_graphviz(tree, dot_string='', uid=-1, depth=0):
    """
    Converts a tree to DOT format for use with visualize/GraphViz
    DO NOT MODIFY THIS FUNCTION!
    """

    uid += 1       # Running index of node ids across recursion
    node_id = uid  # Node id of this node

    if depth == 0:
        dot_string += 'digraph TREE {\n'

    for split_criterion in tree:
        sub_trees = tree[split_criterion]
        attribute_index = split_criterion[0]
        attribute_value = split_criterion[1]
        split_decision = split_criterion[2]

        if not split_decision:
            # Alphabetically, False comes first
            dot_string += '    node{0} [label="x{1} = {2}?"];\n'.format(node_id, attribute_index, attribute_value)

        if type(sub_trees) is dict:
            if not split_decision:
                dot_string, right_child, uid = to_graphviz(sub_trees, dot_string=dot_string, uid=uid, depth=depth + 1)
                dot_string += '    node{0} -> node{1} [label="False"];\n'.format(node_id, right_child)
            else:
                dot_string, left_child, uid = to_graphviz(sub_trees, dot_string=dot_string, uid=uid, depth=depth + 1)
                dot_string += '    node{0} -> node{1} [label="True"];\n'.format(node_id, left_child)

        else:
            uid += 1
            dot_string += '    node{0} [label="y = {1}"];\n'.format(uid, sub_trees)
            if not split_decision:
                dot_string += '    node{0} -> node{1} [label="False"];\n'.format(node_id, uid)
            else:
                dot_string += '    node{0} -> node{1} [label="True"];\n'.format(node_id, uid)

    if depth == 0:
        dot_string += '}\n'
        return dot_string
    else:
        return dot_string, node_id, uid

def learning_curves():
    trnErr1 = dict()
    tstErr1 = dict()
    trnErr2 = dict()
    tstErr2 = dict()
    trnErr3 = dict()
    tstErr3 = dict()

    M = np.genfromtxt('./monks-1.train', missing_values=0, skip_header=0, delimiter=',', dtype=int)
    ytrn1 = M[:, 0]
    Xtrn1 = M[:, 1:]
    M = np.genfromtxt('./monks-1.test', missing_values=0, skip_header=0, delimiter=',', dtype=int)
    ytst1 = M[:, 0]
    Xtst1 = M[:, 1:]

    M = np.genfromtxt('./monks-2.train', missing_values=0, skip_header=0, delimiter=',', dtype=int)
    ytrn2 = M[:, 0]
    Xtrn2 = M[:, 1:]
    M = np.genfromtxt('./monks-2.test', missing_values=0, skip_header=0, delimiter=',', dtype=int)
    ytst2 = M[:, 0]
    Xtst2 = M[:, 1:]
    M = np.genfromtxt('./monks-3.train', missing_values=0, skip_header=0, delimiter=',', dtype=int)
    ytrn3 = M[:, 0]
    Xtrn3 = M[:, 1:]
    M = np.genfromtxt('./monks-3.test', missing_values=0, skip_header=0, delimiter=',', dtype=int)
    ytst3 = M[:, 0]
    Xtst3 = M[:, 1:]
    for maxdepth in range(1,11):

        decision_tree1 = id3(Xtrn1, ytrn1, max_depth=maxdepth)
        y_pred_trn1 = [predict_example(x, decision_tree1) for x in Xtrn1]
        y_pred_tst1 = [predict_example(x, decision_tree1) for x in Xtst1]
        trnErr1[maxdepth] = compute_error(ytrn1, y_pred_trn1)
        tstErr1[maxdepth] = compute_error(ytst1, y_pred_tst1)

        decision_tree2 = id3(Xtrn2, ytrn2, max_depth=maxdepth)
        y_pred_trn2 = [predict_example(x, decision_tree2) for x in Xtrn2]
        y_pred_tst2 = [predict_example(x, decision_tree2) for x in Xtst2]
        trnErr2[maxdepth] = compute_error(ytrn2, y_pred_trn2)
        tstErr2[maxdepth] = compute_error(ytst2, y_pred_tst2)


        decision_tree3 = id3(Xtrn3, ytrn3, max_depth=maxdepth)
        y_pred_trn3 = [predict_example(x, decision_tree3) for x in Xtrn3]
        y_pred_tst3 = [predict_example(x, decision_tree3) for x in Xtst3]
        trnErr3[maxdepth] = compute_error(ytrn3, y_pred_trn3)
        tstErr3[maxdepth] = compute_error(ytst3, y_pred_tst3)

    plt.figure()
    plt.plot(tstErr1.keys(), tstErr1.values(), marker='o', linewidth=3, markersize=12)
    plt.plot(trnErr1.keys(), trnErr1.values(), marker='s', linewidth=3, markersize=12)
    plt.plot(tstErr2.keys(), tstErr2.values(), marker='o', linewidth=3, markersize=12)
    plt.plot(trnErr2.keys(), trnErr2.values(), marker='s', linewidth=3, markersize=12)
    plt.plot(tstErr2.keys(), tstErr3.values(), marker='o', linewidth=3, markersize=12)
    plt.plot(trnErr2.keys(), trnErr3.values(), marker='s', linewidth=3, markersize=12)
    plt.xlabel('Tree Depth', fontsize=16)
    plt.ylabel('Average Training/Test errors', fontsize=16)
    plt.xticks(list(tstErr1.keys()), fontsize=12)
    plt.legend(['Test Error 1', 'Training Error 1', 'Test Error 2', 'Training Error 2', 'Test Error 3', 'Training Error 3'], fontsize=16)
    plt.axis([0, 11, 0, 1])
    plt.show()

def weaklearners():
    M = np.genfromtxt('./monks-1.train', missing_values=0, skip_header=0, delimiter=',', dtype=int)
    ytrn11 = M[:, 0]
    Xtrn11 = M[:, 1:]
    M = np.genfromtxt('./monks-1.test', missing_values=0, skip_header=0, delimiter=',', dtype=int)
    ytst11 = M[:, 0]
    Xtst11 = M[:, 1:]
    for maxdepth in range(1, 6, 2):
        decision_tree11 = id3(Xtrn11, ytrn11, max_depth=maxdepth)
        # y_pred_trn1 = [predict_example(x, decision_tree1) for x in Xtrn1]
        y_pred_tst11 = [predict_example(x, decision_tree11) for x in Xtst11]
        # trnErr[maxdepth] = compute_error(ytrn1, y_pred_trn1) / 3
        # tstErr[maxdepth] = compute_error(ytst1, y_pred_tst1) / 3
        #
        print("Confusion matrix on the test set for depth=", maxdepth)
        print("Test Error = {0:4.2f}%.".format(compute_error(ytst11, y_pred_tst11) * 100))
        print(confusion_matrix(ytst11, y_pred_tst11))

def scikit_learn():
    print("\nscikit-learn")
    M = np.genfromtxt('./monks-1.train', missing_values=0, skip_header=0, delimiter=',', dtype=int)
    ytrn111 = M[:, 0]
    Xtrn111 = M[:, 1:]
    M = np.genfromtxt('./monks-1.test', missing_values=0, skip_header=0, delimiter=',', dtype=int)
    ytst111 = M[:, 0]
    Xtst111 = M[:, 1:]
    for maxdepth in range(1, 6, 2):
        clf = DecisionTreeClassifier(criterion='entropy', max_depth=maxdepth)
        clf.fit(Xtrn111, ytrn111)
        y_pred_tst111 = clf.predict(Xtst111)
        print("Confusion matrix on the test set for depth=", maxdepth)
        print("Test Error = {0:4.2f}%.".format(compute_error(ytst111, y_pred_tst111) * 100))
        print(confusion_matrix(ytst111, y_pred_tst111))


def other_data_sets():
    print("\nOther data set from UCI repository about Breast Cancer Coimbra")
    print("The dataset has been pre-processed into binary features using the mean due to continuous features.")
    M = np.genfromtxt('./otherdatasets.csv', missing_values=0, skip_header=1, delimiter=',', dtype=int)
    ally = M[:, 9]
    allx = M[:, 0:9]
    Xtrn12, Xtst12, ytrn12, ytst12 = train_test_split(allx, ally, test_size=0.35)
    print("\nMy own decisiontree's confusion matrix")
    for maxdepth in range(1, 6, 2):
        decision_tree12 = id3(Xtrn12, ytrn12, max_depth=maxdepth)
        # y_pred_trn1 = [predict_example(x, decision_tree1) for x in Xtrn1]
        y_pred_tst12 = [predict_example(x, decision_tree12) for x in Xtst12]
        # trnErr[maxdepth] = compute_error(ytrn1, y_pred_trn1) / 3
        # tstErr[maxdepth] = compute_error(ytst1, y_pred_tst1) / 3
        #
        print("Confusion matrix on the test set for depth=", maxdepth)
        print("Test Error = {0:4.2f}%.".format(compute_error(ytst12, y_pred_tst12) * 100))
        print(confusion_matrix(ytst12, y_pred_tst12))
    print("\nDecisionTreeClassifier's confusion matrix")
    for maxdepth in range(1, 6, 2):
        clf = DecisionTreeClassifier(criterion='entropy',max_depth=maxdepth)
        clf.fit(Xtrn12, ytrn12)
        y_pred_tst122 = clf.predict(Xtst12)
        print("Confusion matrix on the test set for depth=", maxdepth)
        print("Test Error = {0:4.2f}%.".format(compute_error(ytst12, y_pred_tst122) * 100))
        print(confusion_matrix(ytst12, y_pred_tst122))

if __name__ == '__main__':
    # Load the training data
    M = np.genfromtxt('./monks-1.train', missing_values=0, skip_header=0, delimiter=',', dtype=int)
    ytrn = M[:, 0]
    Xtrn = M[:, 1:]

    # print(type(Xtrn[0]))
    # print(np.shape(Xtrn))
    # print(type(ytrn))
    # print(np.shape(ytrn))

    # Load the test data
    M = np.genfromtxt('./monks-1.test', missing_values=0, skip_header=0, delimiter=',', dtype=int)
    ytst = M[:, 0]
    Xtst = M[:, 1:]

    # Learn a decision tree of depth 3
    decision_tree = id3(Xtrn, ytrn, max_depth=3)

    # Pretty print it to console
    pretty_print(decision_tree)

    # Visualize the tree and save it as a PNG image
    dot_str = to_graphviz(decision_tree)
    render_dot_file(dot_str, './my_learned_tree')

    # Compute the test error
    y_pred = [predict_example(x, decision_tree) for x in Xtst]
    tst_err = compute_error(ytst, y_pred)

    print('Test Error = {0:4.2f}%.'.format(tst_err * 100))

learning_curves()
weaklearners()
scikit_learn()
other_data_sets()