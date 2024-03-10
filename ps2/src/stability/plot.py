import util
import matplotlib.pyplot as plt

def main():
    print('==== Training model on data set A ====')
    Xa, Ya = util.load_csv('ds1_a.csv', add_intercept=True)
    plot(Xa, Ya, "data_a_plot")

    print('\n==== Training model on data set B ====')
    Xb, Yb = util.load_csv('ds1_b.csv', add_intercept=True)
    plot(Xb, Yb, "data_b_plot")

def plot(X, y, save_path):
    # Plot points on a figure
    plt.figure()
    plt.plot(X[y == 1, -2], X[y == 1, -1], 'bx', linewidth=2)
    plt.plot(X[y == 0, -2], X[y == 0, -1], 'go', linewidth=2)

    # Add labels and save to disk
    plt.xlabel('x1')
    plt.ylabel('x2')
    plt.savefig(save_path)

if __name__ == '__main__':
    main()
