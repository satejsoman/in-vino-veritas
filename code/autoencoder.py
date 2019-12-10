import logging
from itertools import product
from os import path
from types import SimpleNamespace as ns

import numpy as np
import pandas as pd
from tqdm import tqdm


# (f, f') tuple 
class Activation():
    def __init__(self, f, df):
        self.f = f
        self.df = df 

    def __call__(self, *args, **kwargs): 
        return self.f(*args, **kwargs)

# index into the derivative of the activation function
def d(activation):
    return activation.df

# activation functions
_sigmoid = lambda z: 1 / (1 + np.exp(-z))
activations = ns(
    sigmoid = Activation(_sigmoid, lambda z: _sigmoid(z) * (1 - _sigmoid(z))),
    relu    = Activation(lambda z: z*(z>0), lambda z: (z > 0) + 1e-12)
)

class Autoencoder(): 
    def __init__(self, layers):
        logging.info("Setting up autoencoder")
        self.layers = layers
        self.params = []

    def train(self, x, tau=0.01, epochs=100, seed=112358):
        np.random.seed(seed)
        p, n = x.shape 

        # wire up layers 
        L = len(self.layers)
        # inputs, outputs, weights, biases, and activation functions
        z, a, w, b, f = {}, {}, {}, {}, {}
        logging.info("Wiring up layers")
        for (l, (activation, shape)) in enumerate(tqdm(self.layers), start=1):
            (n_o, n_i) = shape
            f[l] = activation
            # Xavier initialization: 
            w[l] = np.random.randn(*shape)/(n_i**0.5)
            b[l] = np.zeros((n_o, 1))
    
        # start training
        logging.info("Starting training")
        for t in tqdm(range(epochs)):
            # choose random point
            i = np.random.randint(0, p)  

            # feed forward 
            a[0] = np.array(x.iloc[i]).reshape((n, 1))
            for l in range(1, L+1):
                zl = w[l].dot(a[l-1]) + b[l]
                z[l], a[l] = zl, f[l](zl)

            # calculate node delta for output by averaging input/output difference
            delta = {L: 0.5 * (a[L] - a[0])}
            
            # backpropagation
            for l in reversed(range(L - 1, 0)):
                delta[l] = (w[l+1] @ delta[l+1]) * d(f[l])(z[l])
                w[l] = w[l] - tau * np.outer(delta[l], a[l-1])
                b[l] = b[l] - tau * delta[l]
        self.params = [z, a, w, b, f]
        logging.info("Finished")

    def compress(self, x):
        z, a, w, b, f = self.params
        L = len(self.layers)
        n = x.shape[0] 
        apred = {0: np.array(x).reshape((n, 1))}
        for l in range(1, L):
            apred[l] = f[l](w[l] @ apred[l-1] + b[l])
        return apred[L-1]

    def compress_all(self, xs, out_path):
        logging.info("Compressing input of size %s", xs.shape)
        out = np.vstack([self.compress(row).T for (_, row) in tqdm(xs.iterrows())])
        logging.info("Writing out compressed output")
        np.savetxt(out_path, out, delimiter=",")
        

if __name__ == "__main__":
    logging.basicConfig(format="%(asctime)s|%(levelname)s| %(message)s", datefmt='%Y-%m-%d %H:%M:%S')
    logging.getLogger().setLevel("INFO")

    logging.info("Reading in data")
    features = pd.read_csv("data/features.csv", header=None)

    deep = lambda fn: [
        (fn, (10000, 26023)),
        (fn, (5000, 10000)),
        (fn, (1000, 5000)),
        (fn, (100, 1000)),
        (fn, (50, 100)),
        (fn, (26023, 50))
    ]

    medium = lambda fn: [
        (fn, (10000, 26023)),
        (fn, (1000, 10000)),
        (fn, (50, 1000)),
        (fn, (26023, 50))
    ]

    shallow = lambda fn: [(fn, (50, 26023)), (fn, (26023, 50))]

    # test out some architecutures and training cycles
    epochs = [10, 100, 1000, 10000]
    architectures = {"shallow": shallow, "medium": medium, "deep": deep}
    activation_functions = {"sigmoid": activations.sigmoid, "relu": activations.relu}
    for ((fn_name, activation_fn), (arch_name, arch_fn)) in product(activation_functions.items(), architectures.items()):
        for e in epochs:
            filename = "data/compressed_{}_{}_{}.csv".format(fn_name, arch_name, e)
            if path.exists(filename):
                logging.info("target %s exists, skipping", filename)
            else: 
                logging.info("%s %s %s -> %s", fn_name, arch_name, e, filename)
                nn = Autoencoder(arch_fn(activation_fn))
                nn.train(features, epochs=e)
                nn.compress_all(features, filename)
