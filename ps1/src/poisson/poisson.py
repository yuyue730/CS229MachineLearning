import numpy as np
import util
import matplotlib.pyplot as plt

def main(lr, train_path, eval_path, save_path):
    """Problem: Poisson regression with gradient ascent.

    Args:
        lr: Learning rate for gradient ascent.
        train_path: Path to CSV file containing dataset for training.
        eval_path: Path to CSV file containing dataset for evaluation.
        save_path: Path to save predictions.
    """
    # Load training set
    x_train, y_train = util.load_dataset(train_path, add_intercept=True)

    # *** START CODE HERE ***
    # Fit a Poisson Regression model
    poisson = PoissonRegression(step_size=lr)
    poisson.fit(x_train, y_train)

    # Run on the validation set, and use np.savetxt to save outputs to save_path
    x_eval, y_eval = util.load_dataset(eval_path, add_intercept=True)
    p_eval = poisson.predict(x_eval)
    np.savetxt(save_path, p_eval)

    # Plot the result
    plt.figure()
    plt.scatter(y_eval, p_eval, alpha=0.4, c='red', label='True values vs. Predict values')
    plt.xlabel('True values')
    plt.ylabel('Predict values')
    plt.legend()
    plt.savefig('poisson_valid.png')
    # *** END CODE HERE ***

class PoissonRegression:
    """Poisson Regression.

    Example usage:
        > clf = PoissonRegression(step_size=lr)
        > clf.fit(x_train, y_train)
        > clf.predict(x_eval)
    """

    def __init__(self, step_size=1e-5, max_iter=10000, eps=1e-5,
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
        """Run gradient ascent to maximize likelihood for Poisson regression.

        Args:
            x: Training example inputs. Shape (n_examples, dim).
            y: Training example labels. Shape (n_examples,).
        """
        # *** START CODE HERE ***
        # Get the sample size and dimension from input x and initialize theta
        n_examples, dim = x.shape
        self.theta = np.zeros(dim, dtype=np.float32)

        step_cnt = 0
        while step_cnt < self.max_iter:
            # Make a copy of the previous theta as a local variable, instead of referencing
            # the original `self.theta` class member
            prev_theta = np.copy(self.theta)
            if prev_theta is None:
                break
            self.theta = self._step(x, y, prev_theta)
            step_cnt += 1
            if step_cnt % 5 == 0:
                print("Step = {0}, theta = {1}".format(
                    step_cnt, np.array_str(self.theta, precision=5, suppress_small=True)))
            if np.sum(np.abs(self.theta - prev_theta)) <= self.eps:
                # Break the while loop if sum of theta absolute change does not is no larger than `self.eps`
                break

        print("Total step = {0}, final theta = {1}".format(
            step_cnt, np.array_str(self.theta, precision=5, suppress_small=True)))
        # *** END CODE HERE ***

    def predict(self, x):
        """Make a prediction given inputs x.

        Args:
            x: Inputs of shape (n_examples, dim).

        Returns:
            Floating-point prediction for each input, shape (n_examples,).
        """
        # *** START CODE HERE ***
        return np.exp(x.dot(self.theta))
        # *** END CODE HERE ***

    def _step(self, x, y, prev_theta):
        subtraction = y - np.exp(x.dot(prev_theta))
        return prev_theta + self.step_size * x.T.dot(subtraction)

if __name__ == '__main__':
    main(lr=1e-5,
        train_path='train.csv',
        eval_path='valid.csv',
        save_path='poisson_pred.txt')
