import matplotlib.pyplot as plt
import numpy as np
from keras.layers import Dense, Activation
from keras.optimizers import Adam
from keras.models import Sequential
from keras import activations
from keras.callbacks import Callback
from keras import backend as k
import gc


def plot_learning(scores, avg_scores, title, filename):
    size = len(scores)
    x = [i+1 for i in range(size)]

    plt.ylabel('Score')
    plt.xlabel('Episode')
    plt.plot(x, scores, color='blue', label='score (orginal)')
    plt.plot(x, avg_scores, color='orange', label='score (100 moving averages)')

    plt.title(title)
    plt.legend()

    plt.savefig(filename)
