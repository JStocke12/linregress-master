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
        self.intercept = 0

    def loss(self, slope=None, intercept=None):
        if slope is None:
            slope = self.slope
        if intercept is None:
            intercept = self.intercept
        return np.sum((slope*self.xs+intercept-self.ys)**2)

    def fit(self, slope=None, intercept=None, delta=(2**-10), learning=1):
        if slope is None:
            slope = self.slope
        if intercept is None:
            intercept = self.intercept
        for i in range(10000):
            slope_gradient = (self.loss(slope+delta, intercept)-self.loss(slope-delta, intercept))/(2*delta)
            intercept_gradient = (self.loss(slope, intercept + delta) - self.loss(slope, intercept - delta)) / (2 * delta)
            if self.loss(slope, intercept) < self.loss(slope-slope_gradient*learning, intercept-intercept_gradient*learning)-delta**2:
                learning /= 2
            else:
                slope -= slope_gradient*learning
                intercept -= intercept_gradient*learning
            if slope_gradient**2+intercept_gradient**2 < 0.01:
                break
        self.slope = slope
        self.intercept = intercept
        return (slope, intercept)

    def predict(self, xs):
        return self.slope*xs

    def data_plot(self):
        plt.scatter(self.xs,self.ys)
        plt.show()


def main():
    adelie_bill_len_mm = np.loadtxt("adelie.csv", delimiter=',', skiprows=1, usecols=0)
    adelie_flipper_len_mm = np.loadtxt("adelie.csv", delimiter=',', skiprows=1, usecols=1)

    regression = LinearRegression(adelie_bill_len_mm,adelie_flipper_len_mm)
    """loss_xs = np.linspace(0,10,100)
    loss_ys = np.array([regression.loss(i) for i in loss_xs])

    plt.plot(loss_xs,loss_ys)
    plt.show()"""

    print(regression.fit())

    regression.data_plot()

    """plt.plot(adelie_bill_len_mm, adelie_flipper_len_mm, '.')
    plt.show()"""

if __name__ == '__main__':
    main()