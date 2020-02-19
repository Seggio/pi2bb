import numpy as np


class LWR:

    """
    Class implementing a Locally Weighted Regression model

    Attributes
       ----------
       x_dataset: np.array
       Array containing the x dataset, shape(num_examples,num_features)

       y_dataset: np.array
       Array containing the y dataset, shape (num_examples,)

       tau=0.1:float
       Variance for the Gaussian Kernels

    Methods
       -------
       predict(query_pt):
       Trains a local model and return the prediction a given query point
    """

    def __init__(self, x_dataset, y_dataset, tau=0.1):
        self.x_dataset = x_dataset
        self.y_dataset = y_dataset
        self.tau = tau

    def predict(self, query_pt):

        if self.x_dataset.ndim == 1:
            return self._predict_univariate(query_pt)
        else:
            return self._predict_multivariate(query_pt)

    def _predict_univariate(self, query_pt):

        w = np.exp(- (self.x_dataset - query_pt)**2 / (2 * self.tau))

        den = np.dot(self.x_dataset * w, self.x_dataset)
        num = np.dot(self.x_dataset * w, self.y_dataset)

        return (num / den) * query_pt

    def _predict_multivariate(self, query_pt):

        w = np.diag(np.exp(- np.linalg.norm(self.x_dataset - query_pt, axis=1) / (2 * (self.tau ** 2))))

        first_term = np.linalg.pinv(np.matmul(np.matmul(self.x_dataset.T, w), self.x_dataset))
        second_term = np.matmul(np.matmul(self.x_dataset.T, w), self.y_dataset)
        params = np.matmul(first_term, second_term)

        return np.dot(params, query_pt)


def test():
    """Regressing a sin"""
    import matplotlib.pyplot as plt

    x = np.linspace(1, np.pi *2, 50)
    x_even = x[::2]
    x_odd = x[::2]
    y = np.sin(x_even)
    query = x_odd
    pred = []

    regressor = LWR(x_even,y)
    for q in query:
        pred.append(regressor.predict(q))

    plt.plot(x_even, y, "g^")
    for i in range(len(query)):
        plt.plot(query[i],pred[i],"r*")
    plt.title("Sin interpolation")
    plt.legend(["Training points", "Query points"])
    plt.xlabel("x")
    plt.ylabel("sin(x)")
    plt.show()


def test_multi():
    """Regressing a sin"""
    import matplotlib.pyplot as plt
    from mpl_toolkits.mplot3d import Axes3D

    x1 = np.linspace(1, 10, 100)
    x2 = np.linspace(1, 10, 100)
    x1_even = np.array(x1[::2])
    x1_odd = np.array(x1[1::2])
    x2_even = np.array(x2[::2])
    x2_odd = np.array(x2[1::2])
    y_train = np.sin(x1_even+x2_even) * x1_even**2 + x2_even
    pred = []

    x_train = np.vstack((x1_even,x2_even)).T
    regressor = LWR(x_train, y_train)

    for query in np.vstack((x1_odd,x2_odd)).T:
        pred.append(regressor.predict(query))

    fig = plt.figure()
    ax = fig.add_subplot(111, projection='3d')
    ax.plot(x1_even,x2_even,y_train, "g*")
    ax.plot(x1_odd, x2_odd, pred, "ro")
    ax.set_xlabel("x")
    ax.set_ylabel("y")
    ax.set_zlabel("sin(x+y)")
    ax.legend(["x_training", "x_test"])

    plt.show()


if __name__ == "__main__":
    test_multi()







