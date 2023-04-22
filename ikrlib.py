from glob import glob

import librosa
import matplotlib.pyplot as plt
import numpy as np
import scipy.fftpack
from numpy import ravel, diag, newaxis, array, vstack, pi
from numpy.linalg import eigh, inv
from numpy.random import rand, randn
from scipy.spatial.distance import cdist
from scipy.special import logsumexp


def k_nearest_neighbours(test_data, class1, class2, k):
    euclidean = cdist(np.r_[class1, class2], test_data)
    i = np.argsort(euclidean.T)
    return np.sum(i[:, :k] >= len(class1), axis=1) / float(k)


# Plotting function

def gellipse(mu, cov, n=100, *args, **kwargs):
    """
    Contour plot of 2D Multivariate Gaussian distribution.

    gellipse(mu, cov, n) plots ellipse given by mean vector MU and
    covariance matrix COV. Ellipse is plotted using N (default is 100)
    points. Additional parameters can specify various line types and
    properties. See description of matplotlib.pyplot.plot for more details.
    """
    mu = mu.copy()
    if mu.shape == (2,):
        mu.shape = (2, 1)
    if mu.shape != (2, 1) or cov.shape != (2, 2):
        raise RuntimeError('mu must be a two element vector and cov must be 2 x 2 matrix')

    d, v = eigh(4 * cov)
    d = diag(d)
    t = np.linspace(0, 2 * pi, n)
    x = v.dot(np.sign(d)).dot(np.sqrt(np.abs(d))).dot(array([np.cos(t), np.sin(t)])) + mu
    return plt.plot(x[0], x[1], *args, **kwargs)


def plot2dfun(f, limits, resolution, ax=None):
    if ax is None:
        ax = plt
    xmin, xmax, ymin, ymax = limits
    xlim = np.arange(ymin, ymax, (ymax - ymin) / float(resolution))
    ylim = np.arange(xmin, xmax, (xmax - xmin) / float(resolution))
    (b, a) = np.meshgrid(xlim, ylim)
    return ax.imshow(f(vstack([ravel(a), ravel(b)]).T).reshape(b.shape).T[::-1, :],
                     cmap='gray', aspect='auto', extent=(xmin, xmax, ymin, ymax))


# Gaussian distributions related functions

def logpdf_gauss(x, mu, cov):
    assert (mu.ndim == 1 and len(mu) == len(cov) and (cov.ndim == 1 or cov.shape[0] == cov.shape[1]))
    x = np.atleast_2d(x) - mu
    if cov.ndim == 1:
        return -0.5 * (len(mu) * np.log(2 * pi) + np.sum(np.log(cov)) + np.sum((x ** 2) / cov, axis=1))
    else:
        return -0.5 * (len(mu) * np.log(2 * pi) + np.linalg.slogdet(cov)[1] + np.sum(x.dot(inv(cov)) * x, axis=1))


def train_gauss(x):
    """
    Estimates gaussian distribution from data.
    (MU, COV) = TRAIN_GAUSS(X) return Maximum Likelihood estimates of mean
    vector and covariance matrix estimated using training data X
    """
    return np.mean(x, axis=0), np.cov(x.T, bias=True)


def rand_gauss(n, mu, cov):
    if cov.ndim == 1:
        cov = np.diag(cov)
    assert (mu.ndim == 1 and len(mu) == len(cov) and cov.ndim == 2 and cov.shape[0] == cov.shape[1])
    d, v = eigh(cov)
    return (randn(n, len(mu)) * np.sqrt(d)).dot(v) + mu


# GMM distributions related functions

def logpdf_gmm(x, ws, mus, covs):
    return logsumexp([np.log(w) + logpdf_gauss(x, m, c) for w, m, c in zip(ws, mus, covs)], axis=0)


def train_gmm(x, ws, mus, covs):
    """
    TRAIN_GMM Single iteration of EM algorithm for training Gaussian Mixture Model
    [Ws_new,MUs_new, COVs_new, TLL]= TRAIN_GMM(X,Ws,NUs,COVs) performs single
    iteration of EM algorithm (Maximum Likelihood estimation of GMM parameters)
    using training data X and current model parameters Ws, MUs, COVs and returns
    updated model parameters Ws_new, MUs_new, COVs_new and total log likelihood
    TLL evaluated using the current (old) model parameters. The model
    parameters are mixture component mean vectors given by columns of M-by-D
    matrix MUs, covariance matrices given by M-by-D-by-D matrix COVs and vector
    of weights Ws.
    """
    gamma = np.vstack([np.log(w) + logpdf_gauss(x, m, c) for w, m, c in zip(ws, mus, covs)])
    logevidence = logsumexp(gamma, axis=0)
    gamma = np.exp(gamma - logevidence)
    tll = logevidence.sum()
    gammasum = gamma.sum(axis=1)
    ws = gammasum / len(x)
    mus = gamma.dot(x) / gammasum[:, np.newaxis]

    if covs[0].ndim == 1:  # diagonal covariance matrices
        covs = gamma.dot(x ** 2) / gammasum[:, np.newaxis] - mus ** 2
    else:
        covs = np.array(
            [(gamma[i] * x.T).dot(x) / gammasum[i] - mus[i][:, newaxis].dot(mus[[i]]) for i in range(len(ws))])
    return ws, mus, covs, tll


def rand_gmm(n, ws, mus, covs):
    """
    RAND_GAUSS  Gaussian mixture distributed random numbers.
    X = RAND_GMM(N, Ws, MUs, COVs) returns matrix with N rows, where each 
    column is a vector chosen from a distribution represnted by a Gaussian
    Mixture Model. The GMM parameters are mixture component mean vectors given
    by rows of M-by-D matrix MUs, covariance matrices given by M-by-D-by-D
    matrix COVs and vector of weights Ws.
    """
    ws = np.random.multinomial(n, ws);
    x = np.vstack([rand_gauss(w, m, c) for w, m, c in zip(ws, mus, covs)])
    np.random.shuffle(x)
    return x


# Linear classifiers
def train_generative_linear_classifier(x, class_id):
    true_data = x[class_id == 1]
    false_data = x[class_id != 1]

    (true_mean, true_cov) = train_gauss(true_data)
    (false_mean, false_cov) = train_gauss(false_data)

    data_cov = (true_cov * len(true_data) + false_cov * len(false_data)) / len(x)
    inv_cov = np.linalg.inv(data_cov)
    w0 = -0.5 * true_mean.dot(inv_cov).dot(true_mean) + 0.5 * false_mean.dot(inv_cov).dot(false_mean)
    w = inv_cov.dot(true_mean - false_mean)
    return w, w0, data_cov


def train_linear_logistic_regression(x, class_ids, wold, w0old):
    x = np.c_[x, np.ones(len(x))]
    wold = np.r_[wold, w0old]
    posteriors = logistic_sigmoid(x.dot(wold))
    r = posteriors * (1 - posteriors)
    wnew = wold - (posteriors - class_ids).dot(x).dot(inv((x.T * r).dot(x)))
    # <--------- Gradient ---------> <------- Hessian ------->
    return wnew[:-1], wnew[-1]


def train_linear_logistic_regression_GD(x, class_ids, wold, w0old, learning_rate=None):
    if learning_rate is None:
        learning_rate = 0.003 / len(x)

    x = np.c_[x, np.ones(len(x))]
    wold = np.r_[wold, w0old]
    posteriors = logistic_sigmoid(x.dot(wold))
    wnew = wold - learning_rate * (posteriors - class_ids).dot(x)
    return wnew[:-1], wnew[-1]


#  Neural network binary classifier

def logistic_sigmoid(a):
    return 1 / (1 + np.exp(-a))


def eval_nnet(x, w1, w2):
    """
    Propagate data through the neural network binary classifier

    - X are the input features, datapoints are stored column-wise
    - W1 weights of 1st layer (first column contain bias)
    - W2 weights of 2nd layer (first column contain bias)
    
    Returns the network outputs (single dimensional)
    """
    h = logistic_sigmoid(np.c_[np.ones(len(x)), x].dot(w1))
    return logistic_sigmoid(np.c_[np.ones(len(h)), h].dot(w2))


def train_nnet(X, T, w1, w2, epsilon):
    mixer = np.random.permutation(len(X))
    X = X[mixer]
    T = T[mixer]
    ed = 0
    for x, t in zip(X, T):
        h = logistic_sigmoid(np.r_[1, x].dot(w1))
        y = logistic_sigmoid(np.r_[1, h].dot(w2))

        de_da2 = y - t
        de_dh = w2[1:].T * de_da2

        de_da1 = de_dh * h * (1 - h)

        w1 -= epsilon * np.r_[1, x][:, np.newaxis].dot(de_da1)
        w2 -= epsilon * np.r_[1, h][:, np.newaxis] * de_da2
        ed -= t * np.log(y) + (1 - t) * np.log(1 - y)
    return (w1, w2, ed)


def framing(a, window, shift=1):
    shape = ((a.shape[0] - window) / shift + 1, window) + a.shape[1:]
    strides = (a.strides[0] * shift, a.strides[0]) + a.strides[1:]
    return np.lib.stride_tricks.as_strided(a, shape=shape, strides=strides)


def spectrogram(x, window, noverlap=None, nfft=None):
    if np.isscalar(window): window = np.hamming(window)
    if noverlap is None:    noverlap = window.size / 2
    if nfft is None:    nfft = window.size
    x = framing(x, window.size, window.size - noverlap)
    x = scipy.fftpack.fft(x * window, nfft)
    return x[:, :x.shape[1] / 2 + 1]


def wav16khz2mfcc(dir_name):
    """
    Loads all *.wav files from directory dir_name (must be 16kHz), converts them into MFCC 
    features (13 coefficients) and stores them into a dictionary. Keys are the file names
    and values and 2D numpy arrays of MFCC features.
    """
    features = {}
    for f in glob(dir_name + '/*.wav'):
        print('Processing file: ', f)
        s, rate = librosa.load(f, sr=None)
        assert (rate == 16000)
        features[f] = librosa.feature.mfcc(y=s, sr=16000, n_mfcc=13)
        # SMAZAT MFCC old
    return features


def png2fea(dir_name):
    """
    Loads all *.png images from directory dir_name into a dictionary. Keys are the file names
    and values and 2D numpy arrays with corresponding grayscale images
    """
    features = {}
    for f in glob(dir_name + '/*.png'):
        print('Processing file: ', f)
        features[f] = plt.imread(f, True).astype(np.float64)
    return features
