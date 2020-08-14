import numpy as np
from sklearn.preprocessing import scale
from sklearn.linear_model import SGDClassifier
import matplotlib.pyplot as plt
from prettytable import PrettyTable

def compute_error(y_pred, y_true):
    countT = 0
    for i in range(len(y_pred)):
        if y_pred[i] != y_true[i]:
            countT += 1
    return countT / len(y_pred)

if __name__ == '__main__':
    # a. Perform PCA on the training data and visualize the top 16 eigendigits.
    # Load the training data
    M = np.genfromtxt('./usps.train', missing_values=0, skip_header=0, delimiter=',', dtype=float)
    ytrn = M[:, 0].astype(int)              # n * 1
    Xtrn = M[:, 1:]                         # n * d

    # Pre-process and center the training data
    Xtrn = scale(Xtrn, with_std=False)

    # Compute the eigenvalues and eigenvectors of the covariance matrix
    cov = np.cov(Xtrn.T)
    eigenvalues, eigenvectors = np.linalg.eig(cov)

    # Sort the eigenvalues
    indices_of_eigenvalues_in_ascending_order = np.argsort(eigenvalues)
    indices_of_eigenvalues_in_descending_order = indices_of_eigenvalues_in_ascending_order[::-1]
    top_16_eigenvectors = eigenvectors[:, indices_of_eigenvalues_in_descending_order[:16]]

    # Visualize the top 16 eigendigits
    top_16_eigenvectors_t = top_16_eigenvectors.T
    plt.figure()
    for i in range(16):
        plt.subplot(4, 4, i+1)
        plt.imshow(top_16_eigenvectors_t[i, :].reshape(16, 16), cmap='gray')
        plt.title(i, size=12)
        plt.xticks(())
        plt.yticks(())
    plt.show()

    # b.Plot the cumulative explained variance ratio vs.number of components.
    # Sort the eigenvalues
    eigenvalues_in_descending_order = np.sort(eigenvalues)[::-1]

    # Calculate the cumulative explained variance ratio
    cumsum = np.cumsum(eigenvalues_in_descending_order) / sum(eigenvalues_in_descending_order)

    # Save the dimensionalities k70, k80 and k90
    k70 = 0
    k80 = 0
    k90 = 0
    for i in range(len(cumsum) - 1, -1, -1):
        if cumsum[i] >= 0.9:
            k90 = i
        elif cumsum[i] >= 0.8:
            k80 = i
        elif cumsum[i] >= 0.7:
            k70 = i
        else:
            break

    # Plot
    ratio = [0.7, 0.8, 0.9]
    plt.figure()
    number_of_components = range(1, 257)
    plt.plot(number_of_components, cumsum)
    plt.xlabel('Number of components', fontsize=16)
    plt.ylabel('Cumulative explained variance ratio', fontsize=16)
    plt.yticks(np.arange(0, 1.1, step=0.1), fontsize=12)
    plt.xticks(np.arange(0, 257, step=32), fontsize=12)

    plt.annotate("(%d, 70%%)" % k70, xy=(k70, 0.7))
    plt.annotate("(%d, 80%%)" % k80, xy=(k80, 0.8))
    plt.annotate("(%d, 90%%)" % k90, xy=(k90, 0.9))
    plt.show()

    # c. Use sklearn.linear model.SGDClassifier
    # (a) Compute the projection Xf
    Xf = {}
    for i in (k70, k80, k90, 256):
        eigenvectors_in_descending_order = eigenvectors[:, indices_of_eigenvalues_in_descending_order[:i]]
        Xf[i] = Xtrn @ eigenvectors_in_descending_order

    # (b) Learn different multi_class SVM classifiers
    k_range = [k70, k80, k90, 256]
    alpha_range = (0.0001, 0.001, 0.01, 0.1)
    models = {}
    valerrs = {}

    tb = PrettyTable()
    tb.field_names = ["k", "a=0.0001", "a=0.001", "a=0.01", "a=0.1"]

    M = np.genfromtxt('usps.valid', missing_values=0, skip_header=0, delimiter=',', dtype=float)
    yval = M[:, 0].astype(int)              # n * 1
    Xval = M[:, 1:]                         # n * d
    Xval = scale(Xval, with_std=False)
    for k in k_range:
        eigenvectors_in_descending_order = eigenvectors[:, indices_of_eigenvalues_in_descending_order[:k]]
        for a in alpha_range:
            clf = SGDClassifier(loss="hinge", penalty="l2", random_state=10, alpha=a, max_iter=1000, tol=1e-3)
            clf.fit(Xf[k], ytrn)
            models[k, a] = clf

    # (c) Evaluate the learned SVM model on the validation set.
            y_pred_val = clf.predict(Xval @ eigenvectors_in_descending_order)
            valerrs[k, a] = compute_error(y_pred_val, yval)
        tb.add_row([k, "{0:4.4f}".format(valerrs[k, 0.0001]), "{0:4.4f}".format(valerrs[k, 0.001]),
                    "{0:4.4f}".format(valerrs[k, 0.01]), "{0:4.4f}".format(valerrs[k, 0.1])])
    print(tb)

    # d. Report the error of the best (k, a) pair on the test data
    M = np.genfromtxt('usps.test', missing_values=0, skip_header=0, delimiter=',', dtype=float)
    ytst = M[:, 0].astype(int)              # n * 1
    Xtst = M[:, 1:]                         # n * d
    Xtst = scale(Xtst, with_std=False)

    best_model = models[52, 0.1]
    eigenvectors_in_descending_order = eigenvectors[:, indices_of_eigenvalues_in_descending_order[:52]]
    tsterr = compute_error(best_model.predict(Xtst @ eigenvectors_in_descending_order), ytst)
    print("Test error of the best pair (52, 0.1) with features selection: {:.4f}".format(tsterr))
    best_model = models[256, 0.01]
    eigenvectors_in_descending_order = eigenvectors[:, indices_of_eigenvalues_in_descending_order[:256]]
    tsterr = compute_error(best_model.predict(Xtst @ eigenvectors_in_descending_order), ytst)
    print("Test error of the best pair (256, 0.01) without features selection: {:.4f}".format(tsterr))
