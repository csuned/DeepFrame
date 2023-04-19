import numpy as np
import os
import matplotlib.pyplot as plt

res_dir = os.path.expanduser("~/PycharmProjects/DeepFrame/results")

def plot_series(series_list, title_list):
    fig = plt.figure()
    for series, title in zip(series_list, title_list):
        plt.plot(series, label=title)
    plt.legend()
    plt.savefig(f'{res_dir}/test.png')
    return