import numpy as np
import util

def main(train_path, valid_path, save_path):
    """Problem: Gaussian discriminant analysis (GDA)

    Args:
        train_path: Path to CSV file containing dataset for training.
        valid_path: Path to CSV file containing dataset for validation.
        save_path: Path to save predicted probabilities using np.savetxt().
    """
    # Load dataset
    x_train, y_train = util.load_dataset(train_path, add_intercept=False)

    # *** START CODE HERE ***
    # Train a GDA classifier
    gda = GDA()
    gda.fit(x_train, y_train)

    # Plot decision boundary on validation set
    x_eval, y_eval = util.load_dataset(valid_path, add_intercept=False)
    plot_path = save_path.replace(".txt", ".png")
    util.plot(x_eval, y_eval, gda.theta, plot_path)

    # Use np.savetxt to save predictions on eval set to save_path
    p_eval = gda.predict(x_eval)
    y_hat = p_eval > 0.5
    accuracy = np.mean((y_hat == 1) == (y_eval == 1))
    print("Accuracy = {0}".format(accuracy))
    np.savetxt(save_path, p_eval)
    # *** END CODE HERE ***


class GDA:
    """Gaussian Discriminant Analysis.

    Example usage:
        > clf = GDA()
        > clf.fit(x_train, y_train)
        > clf.predict(x_eval)
    """
    def __init__(self, step_size=0.01, max_iter=10000, eps=1e-5,
                 theta_0=None, verbose=True):
        """
        Args:
            step_size: Step size for iterative solvers only.
            max_iter: Maximum number of iterations for the solver.
            eps: Threshold for determining convergence.
            theta_0: Initial guess for theta. If None, use the zero vector.
            verbose: Print loss values during training.
        """
        self.theta = theta_0
        self.step_size = step_size
        self.max_iter = max_iter
        self.eps = eps
        self.verbose = verbose

    def fit(self, x, y):
        """Fit a GDA model to training set given by x and y by updating
        self.theta.

        Args:
            x: Training example inputs. Shape (n_examples, dim).
            y: Training example labels. Shape (n_examples,).
        """
        # *** START CODE HERE ***
        (sample_size, dimension) = x.shape

        # phi = sum(1 if y(i)=1) for all i / len(y)
        # np.count_nonzero -- Count the occurrence of an element in a NumPy array.
        phi = np.count_nonzero(y == 1) / y.size

        # mu_0 = sum(x if y(i)==0) for all i / occurrence of y = 0
        # mu_1 = sum(x if y(i)==1) for all i / occurrence of y = 1
        # numpy.dot(A, b) -- If A is an N*d array and b is a 1-d array, A.dot(b) is a the sum-product
        # over the last axis of A and b. The result is a 1-d array.
        mu_0 = (y == 0).dot(x) / np.count_nonzero(y == 0)
        mu_1 = (y == 1).dot(x) / np.count_nonzero(y == 1)

        mu_yi = np.zeros((sample_size, 2))
        for i in range(y.size):
            if y[i] == 0:
                mu_yi[i] = mu_0
            elif y[i] == 1:
                mu_yi[i] = mu_1

        # x-mu_yi is a 800*2 matrix -> 1 / sample_size * (x-mu_yi).T.dot(x-mu_yi) is a 2*2 matrix
        sigma = 1 / sample_size * (x-mu_yi).T.dot(x-mu_yi)
        sigma_inverse = np.linalg.inv(sigma)

        self.theta = sigma_inverse.dot(mu_1 - mu_0)
        mu_diff = mu_0.T.dot(sigma_inverse).dot(mu_0) - mu_1.T.dot(sigma_inverse).dot(mu_1)

        self.theta = np.zeros(dimension + 1)
        self.theta[0] = 0.5 * mu_diff - np.log((1 - phi) / phi)
        self.theta[1:] = sigma_inverse.dot(mu_1 - mu_0)

        print("theta = {0}".format(self.theta))
        # Find phi, mu_0, mu_1, and sigma
        # Write theta in terms of the parameters
        # *** END CODE HERE ***

    def predict(self, x):
        """Make a prediction given new inputs x.

        Args:
            x: Inputs of shape (n_examples, dim).

        Returns:
            Outputs of shape (n_examples,).
        """
        # *** START CODE HERE ***
        return 1 / (1 + np.exp(-(self.theta[0] + x.dot(self.theta[1:]))))
        # *** END CODE HERE

if __name__ == '__main__':
    main(train_path='ds1_train.csv',
         valid_path='ds1_valid.csv',
         save_path='gda_pred_1.txt')

    main(train_path='ds2_train.csv',
         valid_path='ds2_valid.csv',
         save_path='gda_pred_2.txt')
