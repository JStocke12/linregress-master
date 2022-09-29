import numpy as np
import matplotlib.pyplot as plt


class LinearRegression:
    def __init__(self, xs, ys):

        if xs is None:
            xs = np.zeros(10)

        if ys is None:
            ys = np.zeros(10)

        self.xs = xs
        self.ys = ys
        self.slope = 0

    def loss(self, slope=None):
        if slope is None:
            slope = self.slope
        return np.sum((slope*self.xs-self.ys)**2)

    def fit(self, slope=None, delta=(2**-10), learning=(2**-20)):
        if slope is None:
            slope = self.slope
        for i in range(100):
            gradient = (self.loss(slope+delta)-self.loss(slope-delta))/(2*delta)
            slope -= gradient*learning
            print(gradient)
        return slope

    def predict(self, xs):
        return self.slope*xs


def main():
    adelie_bill_len_mm = np.loadtxt("adelie.csv", delimiter=',', skiprows=1, usecols=0)
    adelie_flipper_len_mm = np.loadtxt("adelie.csv", delimiter=',', skiprows=1, usecols=1)

    regression = LinearRegression(adelie_bill_len_mm,adelie_flipper_len_mm)
    loss_xs = np.linspace(0,10,100)
    loss_ys = np.array([regression.loss(i) for i in loss_xs])

    plt.plot(loss_xs,loss_ys)
    plt.show()

    print(regression.fit(slope=4.9))

    """plt.plot(adelie_bill_len_mm, adelie_flipper_len_mm, '.')
    plt.show()"""

if __name__ == '__main__':
    main()