import numpy as np
import util

def main(train_path, valid_path, save_path):
    """Problem: Logistic regression with Newton's Method.

    Args:
        train_path: Path to CSV file containing dataset for training.
        valid_path: Path to CSV file containing dataset for validation.
        save_path: Path to save predicted probabilities using np.savetxt().
    """
    x_train, y_train = util.load_dataset(train_path, add_intercept=True)

    # *** START CODE HERE ***
    # Train a logistic regression classifier
    log_reg = LogisticRegression()
    log_reg.fit(x_train, y_train)

    # Plot decision boundary on top of validation set set
    x_eval, y_eval = util.load_dataset(valid_path, add_intercept=True)
    plot_path = save_path.replace(".txt", ".png")
    util.plot(x_eval, y_eval, log_reg.theta, plot_path)

    # Use np.savetxt to save predictions on eval set to save_path
    p_eval = log_reg.predict(x_eval)
    y_hat = p_eval > 0.5
    accuracy = np.mean((y_hat == 1) == (y_eval == 1))
    print("Accuracy = {0}".format(accuracy))
    np.savetxt(save_path, p_eval)
    # *** END CODE HERE ***

class LogisticRegression:
    """Logistic regression with Newton's Method as the solver.

    Example usage:
        > clf = LogisticRegression()
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
        """Run Newton's Method to minimize J(theta) for logistic regression.

        Args:
            x: Training example inputs. Shape (n_examples, dim).
            y: Training example labels. Shape (n_examples,).
        """
        # *** START CODE HERE ***
        # Get number of features and number of samples
        (n_examples, dim) = x.shape
        assert n_examples == y.shape[0]

        # Initialize theta as a zero vector if no default values are input
        if self.theta == None:
            self.theta = np.zeros(dim, dtype=np.float64)

        for i in range(self.max_iter):  # range(self.max_iter):
            # For each step of iteration, we use Newton's Method. The formula is
            # theta := theta - (Heissan^-1) * (GD of loss_function(theta))
            # Heissan's dimension is dim * dim
            hessian = self._hessian(x)  # 3*3
            gradient = self._gradient(x, y)  # 800*3
            prev_theta = np.copy(self.theta)
            self.theta -= self.step_size * np.linalg.inv(hessian).dot(gradient)
            # np.linalg.inv(hessian): 3*3
            # gradient: 3*1
            # self.theta: 3*1

            current_loss = self._loss(x, y)
            print("Iteration = {0}, loss = {1}".format(i, current_loss))

            if np.sum(np.abs(self.theta - prev_theta)) < self.eps:
                break

        print("Final theta = {0}".format(self.theta))
        # *** END CODE HERE ***

    def predict(self, x):
        """Make a prediction given new inputs x.

        Args:
            x: Inputs of shape (n_examples, dim).

        Returns:
            Outputs of shape (n_examples,).
        """
        # *** START CODE HERE ***
        return self._sigmoid(x.dot(self.theta))
        # x_eval: 100*3
        # theta: 3*1
        # x_eval * theta: 100*1
        # *** END CODE HERE ***

    def _hessian(self, x):
        # Hessian is a dim*dim matrix where dim is number of feature dimensions.
        # H_pq = sum(x_p, x_q) for all samples * g(z) * (1-g(z))
        #      = (x^T * x) * (g(z) * (1-g(z)))
        # g(.) is the sigmoid function g(z)=1/(1+exp(-z))
        # z=<theta,x>

        n_examples = x.shape[0]
        dim = x.shape[1]

        # compute H_pq = x^T * diag(g(z)*(1-g(z))) * x

        # g = g(z) * (1-g(z))
        sigmoid = self._sigmoid(np.dot(x, self.theta))  # 800*1
        g = sigmoid * (1.0 - sigmoid)  # 800*1

        # dot(diag(g)) -> 800*800
        # x -> 800*3
        # x.T (3*800) * diag(g) (800*800) * x (800*3) -> 3*3
        return 1 / n_examples * x.T.dot(np.diag(g)).dot(x)

    def _gradient(self, x, y):
        # Gradient decent = -sum(y-sigmoid(x)) for all samples / n_samples
        # x: 800 * 3
        n_samples = x.shape[0]
        diff = y - self._sigmoid(np.dot(x, self.theta))  # 800-d vector
        return -1.0 / n_samples * np.dot(diff, x)
        # diff: 800-d vector, x: 800*3 -> 3-d vector
        # np.dot(diff, x): 3-d vector

    def _loss(self, x, y):
        n_samples = x.shape[0]
        # log(x + eps) -> if eps does not exist, it is possible log(x) = nan.
        loss1 = np.dot(y, np.log(self._sigmoid(np.dot(x, self.theta)) + self.eps))
        loss2 = np.dot(1-y, np.log(1-self._sigmoid(np.dot(x, self.theta)) + self.eps))
        return -(loss1 + loss2) / n_samples

    def _sigmoid(self, x):
        return 1.0 / (1.0 + np.exp(-x))

if __name__ == '__main__':
    main(train_path='ds1_train.csv',
         valid_path='ds1_valid.csv',
         save_path='logreg_pred_1.txt')

    main(train_path='ds2_train.csv',
         valid_path='ds2_valid.csv',
         save_path='logreg_pred_2.txt')
