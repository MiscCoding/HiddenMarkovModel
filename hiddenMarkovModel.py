import numpy as np
import matplotlib.pyplot as plt

def random_normalized(d1, d2):
    x = np.random.random((d1, d2))
    return x/x.sum(axis=1, keepdims=True)

class HMM:
    def __init__(self, M):
        self.M = M

    def fit(self, X, max_iter=30):
        np.random.seed(123)

        V = max(max(x) for x in X) + 1
        N = len(X)

        self.pi = np.ones(self.M)/self.M
        self.A = random_normalized(self.M, self.M)
        self.B = random_normalized(self.M, V)

        costs = []
        for it in range(max_iter):
            if it % 10 == 0:
                print("it: ", it)

            alphas = []
            betas = []
            P = np.zeros(N)
            for n in range(N):
                x = X[n]
                T = len(x)
                alpha = np.zeros({T, self.M})
                alphas[0] = self.pi * self.B[:, x[0]]
                for t in range(1, T):
                    alphas[t] = alpha[t-1].dot(self.A) * self.B[:, x[t]]
                    for t in range(1, T):
                        alphas[t] = alpha[t-1].dot(self.A) * self.B(t, x[t])

                    P[n] = alpha[-1].sum()
                    alphas.append(alpha)

