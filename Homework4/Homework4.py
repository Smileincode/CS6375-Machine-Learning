
a
import os
import graphviz
import math
import matplotlib.pyplot as plt
from sklearn.metrics import confusion_matrix
from sklearn.tree import DecisionTreeClassifier
from sklearn.model_selection import train_test_split
import random
from sklearn.ensemble import AdaBoostClassifier
from sklearn.ensemble import BaggingClassifier

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


def entropy(y, w=None):
    """
    Compute the entropy of a vector y by considering the counts of the unique values (v1, ... vk), in z

    Returns the entropy of z: H(z) = p(z=v1) log2(p(z=v1)) + ... + p(z=vk) log2(p(z=vk))
    """
    if w is not None:
        counts = dict()
        ans = 0.0
        sum = 0.0
        for i in w:
            sum += i
        for i in y:
            if i[0] not in counts:
                counts[i[0]] = w[i]
            else:
                counts[i[0]] += w[i]
        for j in counts:
            ans += -(1.0 * counts.get(j) / sum) * math.log(1.0 * counts.get(j) / sum, 2)
        return ans
    else:
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


def mutual_information(x, y, w=None):
    """
    Compute the mutual information between a data column (x) and the labels (y). The data column is a single attribute
    over all the examples (n x 1). Mutual information is the difference between the entropy BEFORE the split set, and
    the weighted-average entropy of EACH possible split.

    Returns the mutual information: I(x, y) = H(y) - H(y | x)
    """
    # print("x=", len(x))
    # print("y=", len(y))
    # print("w=", len(w))
    if w is not None:
        subsets = partition(x)
        sumweight = dict()
        count_y = dict()
        entropy_yx = 0.0
        sum = 0.0
        #print(len(w))
        for i in subsets:
            sumweight[i] = 0
            for j in subsets[i]:
                # print("w=", len(w))
                # print("j=",j)
                sumweight[i] += w[j]
        for i in w:
            sum += i
        for i in subsets:
            px = 1.0 * sumweight[i] / sum
            count_y.clear()
            for j in subsets[i]:
                if y[j, 0] not in count_y:
                    count_y[y[j, 0]] = w[j]
                else:
                    count_y[y[j, 0]] += w[j]
            pyx = 0.0
            for j in count_y:
                pyx += -(1.0 * count_y.get(j) / sumweight[i]) * math.log(1.0 * count_y.get(j) / sumweight[i], 2)
            entropy_yx += px * pyx
        return entropy(y) - entropy_yx
    else:
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


def id3(x, y, attribute_value_pairs=None, depth=0, max_depth=5, w=None):
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
        curr_mutual_info = mutual_information(curr_normalization, y, w)
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
    false_example_w = []
    true_example_x = []
    true_example_y = []
    true_example_w = []
    for i in range(len(best_normalization)):
        if best_normalization[i][0] != attribute_value:
            # print(np.shape(x[i, :]))
            # print(np.shape(y[i, :]))
            false_example_x.append(x[i, :])
            false_example_y.append(y[i, :])
            if w is not None:
                false_example_w.append(w[i])
        else:
            true_example_x.append(x[i, :])
            true_example_y.append(y[i, :])
            if w is not None:
                true_example_w.append(w[i])

    if len(false_example_x) == 0 or len(true_example_x) == 0:
        if count_one > count_zero:
            return label_one
        else:
            return label_zero
    if w is not None:
        root[(best_attribute, attribute_value, True)] = id3(np.asarray(true_example_x), np.asarray(true_example_y),
                                                            attribute_value_pairs, depth + 1, max_depth, true_example_w)
        root[(best_attribute, attribute_value, False)] = id3(np.asarray(false_example_x), np.asarray(false_example_y),
                                                             attribute_value_pairs, depth + 1, max_depth, false_example_w)
    else:
        root[(best_attribute, attribute_value, True)] = id3(np.asarray(true_example_x), np.asarray(true_example_y),
                                                            attribute_value_pairs, depth + 1, max_depth)
        root[(best_attribute, attribute_value, False)] = id3(np.asarray(false_example_x), np.asarray(false_example_y),
                                                             attribute_value_pairs, depth + 1, max_depth)
    # root[(best_attribute, attribute_value, True)] = id3(np.asarray(true_example_x), np.asarray(true_example_y),
    #                                                      attribute_value_pairs, depth + 1, max_depth)

    return root


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



def bagging(x, y, max_depth, num_trees):
    bag = []
    y = np.reshape(y, (len(y), 1))
    for i in range(num_trees):
        randomX = []
        randomY = []
        # randomX = np.empty([len(x), len(x[0])], dtype=int)
        # randomY = np.empty([len(y), 1], dtype=int)
        for j in range(len(x)):
            row = random.randint(0, len(x) - 1)
            randomX.append(x[row])
            randomY.append(y[row])

            # np.append(randomX, [x[row]], axis=0)
            # np.append(randomY, [y[row]], axis=0)
        # print(type(randomX))
        # print(type(randomY))
        # print(np.shape(randomX))
        # print(np.shape(randomY))
        bag.append((i, id3(np.asarray(randomX), np.asarray(randomY), max_depth=max_depth)))
        #bag.append((i, id3(randomX, randomY, max_depth=max_depth)))
    return bag

def boosting(x, y, max_depth, num_stumps):
    ans = []
    weights = []
    y = np.reshape(y, (len(y), 1))
    for i in range(len(x)):
        weights.append(1 / len(x))
    for t in range(num_stumps):
        sumdt = 0.0
        sumdt1 = 0.0
        for i in weights:
            sumdt += i
        root = id3(x, y, max_depth=max_depth, w=weights)
        # print(type(y))
        # print(np.shape(y))
        for i in range(len(x)):
            if predict_example(x[i], root) != y[i][0]:
                sumdt1 += weights[i]
        et = sumdt1 / sumdt
        at = 1.0 / 2 * np.log((1 - et) / et)
        zt = 0.0
        for i in range(len(x)):
            if predict_example(x[i], root) != y[i][0]:
                weights[i] = weights[i] * np.exp(at)
                zt += weights[i] * np.exp(at)
            else:
                weights[i] = weights[i] * np.exp(-at)
                zt += weights[i] * np.exp(-at)
        for i in range(len(x)):
            weights[i] = weights[i] / zt
        ans.append((at, root))
    return ans

def predict_example(x, h_ens):
    if type(h_ens) is dict:
        for i in h_ens:
            if x[i[0] - 1] == i[1]:
                if i[2] is True:
                    return predict_example(x, h_ens[i])
            else:
                if i[2] is False:
                    return predict_example(x, h_ens[i])

    elif isinstance(h_ens, list):
        predict = 0
        #print(type(h_ens))
        for i in h_ens:
            predict_lable = predict_example(x, i[1])
            if predict_lable == 0:
                predict_lable = -1
            predict += i[0] * predict_lable
        if predict >= 0.5:
            return 1
        else:
            return 0
    #elif isinstance(h_ens, int):
    else:
        return h_ens

def predict_bagging(x, bag):
    one = 0
    zero = 0
    for i in bag:
        if predict_example(x, i[1]) == 1:
            one += 1
        else:
            zero += 1
    if one >= zero:
        return 1
    else:
        return 0


if __name__ == '__main__':
    # Load the training data
    M = np.genfromtxt('./mushroom.train', missing_values=0, skip_header=0, delimiter=',', dtype=int)
    ytrn = M[:, 0]
    Xtrn = M[:, 1:]
    # Load the test data
    M = np.genfromtxt('./mushroom.test', missing_values=0, skip_header=0, delimiter=',', dtype=int)
    ytst = M[:, 0]
    Xtst = M[:, 1:]
    # print(type(ytrn))
    #
    # print(np.shape(ytrn))
    bag = []
    for d in [3, 5]:
        for k in [5, 10]:
            bag = bagging(Xtrn, ytrn, d, k)
            y_pred_tst = [predict_example(x, bag) for x in Xtst]
            tn, fp, fn, tp = confusion_matrix(ytst, y_pred_tst).ravel()
            print("Use my bagging learner for maximum depth = %d and bag size = %d"%(d,k))
            print("Test Error = {0:4.2f}%.".format(compute_error(ytst, y_pred_tst) * 100),
                  " Presion = {0:4.2f}%.".format(tp / (tp + fp) * 100),
                  " Recall = {0:4.2f}%.".format(tp / (tp + fn) * 100))
            print(confusion_matrix(ytst, y_pred_tst))
            bagClassifier = BaggingClassifier(DecisionTreeClassifier(criterion='entropy', max_depth=d),
                                                 n_estimators=k).fit(Xtrn, ytrn)
            y_pred_tst = (bagClassifier.predict(Xtst))
            tn, fp, fn, tp = confusion_matrix(ytst, y_pred_tst).ravel()
            print("Use scikit-learn's bagging learner for maximum depth = %d and bag size = %d"%(d,k))
            print("Test Error = {0:4.2f}%.".format(compute_error(ytst, y_pred_tst) * 100),
                  " Presion = {0:4.2f}%.".format(tp / (tp + fp) * 100),
                  " Recall = {0:4.2f}%.".format(tp / (tp + fn) * 100))
            print(confusion_matrix(ytst, y_pred_tst))

    boost = []
    for d in [1, 2]:
        for k in [5, 10]:
            boost = boosting(Xtrn, ytrn, d, k)
            y_pred_tst = [predict_example(x, boost) for x in Xtst]
            tn, fp, fn, tp = confusion_matrix(ytst, y_pred_tst).ravel()
            print("Use my AdaBoost learner for maximum depth = %d and ensemble size = %d"%(d,k))
            print("Test Error = {0:4.2f}%.".format(compute_error(ytst, y_pred_tst) * 100),
                  " Presion = {0:4.2f}%.".format(tp / (tp + fp) * 100),
                  " Recall = {0:4.2f}%.".format(tp / (tp + fn) * 100))
            print(confusion_matrix(ytst, y_pred_tst))
            boostClassifier = AdaBoostClassifier(DecisionTreeClassifier(criterion='entropy', max_depth=d),
                                                 n_estimators=k).fit(Xtrn, ytrn)
            y_pred_tst = (boostClassifier.predict(Xtst))
            tn, fp, fn, tp = confusion_matrix(ytst, y_pred_tst).ravel()
            print("Use scikit-learn's AdaBoost learner for maximum depth = %d and ensemble size = %d"%(d,k))
            print("Test Error = {0:4.2f}%.".format(compute_error(ytst, y_pred_tst) * 100),
                  " Presion = {0:4.2f}%.".format(tp / (tp + fp) * 100),
                  " Recall = {0:4.2f}%.".format(tp / (tp + fn) * 100))
            print(confusion_matrix(ytst, y_pred_tst))
