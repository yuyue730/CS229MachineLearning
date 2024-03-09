import numpy as np
import util
import sys

sys.path.append('../linearclass')

### NOTE : You need to complete p01b_logreg implementation first!

from logreg import LogisticRegression

# Character to replace with sub-problem letter in plot_path/save_path
WILDCARD = 'X'


def main(train_path, valid_path, test_path, save_path):
    """Problem 2: Logistic regression for incomplete, positive-only labels.

    Run under the following conditions:
        1. on t-labels,
        2. on y-labels,
        3. on y-labels with correction factor alpha.

    Args:
        train_path: Path to CSV file containing training set.
        valid_path: Path to CSV file containing validation set.
        test_path: Path to CSV file containing test set.
        save_path: Path to save predictions.
    """
    output_path_true = save_path.replace(WILDCARD, 'true')
    output_path_naive = save_path.replace(WILDCARD, 'naive')
    output_path_adjusted = save_path.replace(WILDCARD, 'adjusted')

    # *** START CODE HERE ***
    # Part (a): Train and test on true labels
    # Train a logistic regression classifier with {x, t}
    x_train, t_train = util.load_dataset(train_path, label_col='t', add_intercept=True)
    log_reg_a = LogisticRegression()
    log_reg_a.fit(x_train, t_train)

    # Inference the test set and save the results
    x_test, t_test = util.load_dataset(test_path, label_col='t', add_intercept=True)
    p_test = log_reg_a.predict(x_test)
    np.savetxt(output_path_true, p_test)

    # Output prediction accuracy
    accuracy = np.mean((p_test == 1) == (t_test == 1))
    print("Accuracy = {0}".format(accuracy))

    # Plot x_test, y_test and the decision boundary
    plot_path_true = output_path_true.replace(".txt", ".png")
    util.plot(x_test, t_test, log_reg_a.theta, plot_path_true)

    # Part (b): Train on y-labels and test on true labels
    # Train a logistic regression classifier with {x, y}
    x_train, y_train = util.load_dataset(train_path, label_col='y', add_intercept=True)
    log_reg_b = LogisticRegression()
    log_reg_b.fit(x_train, y_train)

    # Make sure to save predicted probabilities to output_path_naive using np.savetxt()
    # Inference the test set and save the results
    x_test, y_test = util.load_dataset(test_path, label_col='y', add_intercept=True)
    p_test_b = log_reg_b.predict(x_test)
    np.savetxt(output_path_naive, p_test_b)

    # Plot x_test, t_test and the decision boundary
    plot_path_naive = output_path_naive.replace(".txt", ".png")
    util.plot(x_test, t_test, log_reg_b.theta, plot_path_naive)

    # Part (f): Apply correction factor using validation set and test on true labels
    x_valid, y_valid = util.load_dataset(valid_path, label_col='y', add_intercept=True)
    valid_sample_size = x_valid.shape[0]
    positive_valid_sample_size = np.sum(y_valid == 1)
    hx_sum = 0
    for i in range(valid_sample_size):
        if y_valid[i] == 1:
            hx_sum += log_reg_b.predict(x_valid[i])
    alpha = hx_sum / positive_valid_sample_size

    # Make sure to save predicted probabilities to output_path_naive using np.savetxt()
    # Inference the test set and save the results
    p_test_f = log_reg_b.predict(x_test) / alpha
    np.savetxt(output_path_adjusted, p_test_f)

    # Plot x_test, t_test and the decision boundary
    plot_path_naive = output_path_adjusted.replace(".txt", ".png")
    util.plot(x_test, t_test, log_reg_b.theta, plot_path_naive, correction=alpha)

    # Plot and use np.savetxt to save outputs to output_path_adjusted
    # *** END CODER HERE

if __name__ == '__main__':
    main(train_path='train.csv',
        valid_path='valid.csv',
        test_path='test.csv',
        save_path='posonly_X_pred.txt')
