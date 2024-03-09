import util
import numpy as np
import matplotlib.pyplot as plt

np.seterr(all='raise')

factor = 2.0

class LinearModel(object):
    """Base class for linear models."""

    def __init__(self, theta=None):
        """
        Args:
            theta: Weights vector for the model.
        """
        self.theta = theta

    def fit(self, X, y):
        """Run solver to fit linear model. You have to update the value of
        self.theta using the normal equations.

        Args:
            X: Training example inputs. Shape (n_examples, dim).
            y: Training example labels. Shape (n_examples,).
        """
        # *** START CODE HERE ***
        # Calculate the pseudoinverse and solve the linear regression problem
        # See https://en.wikipedia.org/wiki/Moore%E2%80%93Penrose_inverse for definition
        # of Pseudo Inverse
        pinv_X = np.linalg.pinv(X)
        self.theta = pinv_X @ y

        # Solution uses the implementation below. When k gets larger, the two implementation
        # methods will divert. In my opinion, this is due to numpy library implementation
        # difference between these two APIs. Both should be considered correct.
        # y_new = np.expand_dims(y, axis=1)
        # theta = np.linalg.solve(np.dot(X.T, X), np.dot(X.T, y_new))
        # *** END CODE HERE ***

    def create_poly(self, k, X):
        """
        Generates a polynomial feature map using the data x.
        The polynomial map should have powers from 0 to k
        Output should be a numpy array whose shape is (n_examples, k+1)

        Args:
            X: Training example inputs. Shape (n_examples, 2).
        """
        # *** START CODE HERE ***
        X_c1 = X[:,1]
        for next_power in range(2, k + 1):
            next_col = np.power(X_c1, next_power)
            # Concatenate the new column using hstack
            X = np.hstack((X, next_col[:, None]))
        return X
        # *** END CODE HERE ***

    def create_sin(self, k, X):
        """
        Generates a sin with polynomial featuremap to the data x.
        Output should be a numpy array whose shape is (n_examples, k+2)

        Args:
            X: Training example inputs. Shape (n_examples, 2).
        """
        # *** START CODE HERE ***
        X = self.create_poly(k, X)
        X_c1 = X[:, 1]
        sin_X = np.sin(X_c1)
        return np.hstack((X, sin_X[:, None]))
        # *** END CODE HERE ***

    def predict(self, X):
        """
        Make a prediction given new inputs x.
        Returns the numpy array of the predictions.

        Args:
            X: Inputs of shape (n_examples, dim).

        Returns:
            Outputs of shape (n_examples,).
        """
        # *** START CODE HERE ***
        return np.dot(X, self.theta)
        # *** END CODE HERE ***


def run_exp(train_path, sine=False, ks=[1, 2, 3, 5, 10, 20], filename='plot.png'):
    train_x, train_y = util.load_dataset(train_path,add_intercept=True)
    plot_x = np.ones([1000, 2])
    plot_x[:, 1] = np.linspace(-factor*np.pi, factor*np.pi, 1000)
    plt.figure()
    plt.scatter(train_x[:, 1], train_y)

    for k in ks:
        '''
        Our objective is to train models and perform predictions on plot_x data
        '''
        # *** START CODE HERE ***
        linearModel = LinearModel()
        if sine:
            train_phi = linearModel.create_sin(k, train_x)  # training feature map
            plot_phi = linearModel.create_sin(k, plot_x)  # plot feature map
        else:
            train_phi = linearModel.create_poly(k, train_x)  # training feature map
            plot_phi = linearModel.create_poly(k, plot_x)  # plot feature map

        linearModel.fit(train_phi, train_y)
        plot_y = linearModel.predict(plot_phi)
        # *** END CODE HERE ***
        '''
        Here plot_y are the predictions of the linear model on the plot_x data
        '''
        plt.ylim(-2, 2)
        plt.plot(plot_x[:, 1], plot_y, label='k=%d' % k)

    plt.legend()
    plt.savefig(filename)
    plt.clf()


def main(train_path, small_path, eval_path):
    '''
    Run all expetriments
    '''
    # *** START CODE HERE ***
    # Part b. Degree-3 polynomial regression
    run_exp(train_path, sine=False, ks=[3], filename="large-poly3")
    # Part c. Degree-k polynomial regression
    run_exp(train_path, sine=False, ks=[1, 2, 3, 5, 10, 20], filename="large-multipoly")
    # Part d. Degree-k polynomial with sin(x) regression
    run_exp(train_path, sine=True, ks=[1, 2, 3, 5, 10, 20], filename="large-sin")
    # Part e. A smaller dataset
    run_exp(small_path, sine=False, ks=[1, 2, 3, 5, 10, 20], filename="small-multipoly")
    run_exp(small_path, sine=True, ks=[1, 2, 3, 5, 10, 20], filename="small-sin")
    # *** END CODE HERE ***

if __name__ == '__main__':
    main(train_path='train.csv',
        small_path='small.csv',
        eval_path='test.csv')
