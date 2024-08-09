import os
import logging
import sys
from sklearn.manifold import TSNE
from sklearn.manifold import MDS
import seaborn as sns
import matplotlib.pyplot as plt
import numpy as np

def normalize(series):
    xmin = min(series)
    xran = max(series) - min(series)
    if xran==0:
        print('????normalize')
    result = (np.array(series) - xmin) / xran
    return result

def mask_simi_series_old(simi_series_list, motif_loc):
    first_simi = simi_series_list[0]
    threshold = 0.2
    first_period_len = len(first_simi)
    motif_percent = motif_loc / first_period_len
    left_percent = max(0, motif_percent-threshold)
    right_percent = min(1, motif_percent+threshold)
    # left_percent = 0
    # right_percent = 1

    new_simi_list = []
    for simi in simi_series_list:
        simi_len = len(simi)
        left_idx = int(left_percent * simi_len)
        right_idx = int(right_percent * simi_len)
        tmp_simi = -simi[left_idx:right_idx]
        new_simi = normalize(tmp_simi)
        result = np.zeros(simi_len)
        result[left_idx:right_idx] = new_simi
        # todo test, 特定位置=1
        # result[left_idx:right_idx] = 1
        # todo test, <0.5=0
        # result[result<0.5]=0
        new_simi_list.append(result)

    return new_simi_list

def mask_simi_series(simi_series_list, motif_loc):
    '''
    simi_series_list: similarity series of every periods generated from one motif
    motif_loc: location of the motif at the first period
    Algorithm: find the highest peak of each similarity series, and then extend +/- 0.1 percent.
    use this range only, replace other values as 0.
    '''

    threshold = 0.1
    new_simi_list = []
    for simi in simi_series_list:
        simi_len = len(simi)
        motif_loc = np.where(simi==min(simi))[0][0]  # 目前是min
        motif_percent = motif_loc / simi_len
        left_percent = max(0, motif_percent - threshold)
        right_percent = min(1, motif_percent+threshold)
        left_idx = int(left_percent * simi_len)
        right_idx = int(right_percent * simi_len)

        result = np.zeros(simi_len)
        # tmp_simi = normalize(-simi)
        # result[left_idx:right_idx] = tmp_simi[left_idx:right_idx]

        tmp_simi = normalize(-simi[left_idx:right_idx])
        result[left_idx:right_idx] = tmp_simi
        new_simi_list.append(result)

    return new_simi_list

def filter_simi_by_threshold(simi_series_list):
    threshold = 0.8
    new_simi_list = []
    for simi in simi_series_list:
        norm_simi = normalize(-simi)
        norm_simi[norm_simi<threshold] = 0
        new_simi_list.append(norm_simi)
    return new_simi_list

def _logger(logger_name, level=logging.DEBUG):
    """
    Method to return a custom logger with the given name and level
    """
    logger = logging.getLogger(logger_name)
    logger.setLevel(level)
    format_string = "%(message)s"
    log_format = logging.Formatter(format_string)
    # Creating and adding the console handler
    console_handler = logging.StreamHandler(sys.stdout)
    console_handler.setFormatter(log_format)
    logger.addHandler(console_handler)
    # Creating and adding the file handler
    file_handler = logging.FileHandler(logger_name, mode='a')
    file_handler.setFormatter(log_format)
    logger.addHandler(file_handler)
    return logger

def tsne(latent, y_ground_truth, save_dir):
    """
        Plot t-SNE embeddings of the features
    """
    latent = latent.cpu().detach().numpy()
    y_ground_truth = y_ground_truth.cpu().detach().numpy()
    tsne = TSNE(n_components=2, verbose=1, perplexity=40, n_iter=300)
    tsne_results = tsne.fit_transform(latent)
    plt.figure(figsize=(16,10))
    set_y = set(y_ground_truth)
    # set_y = set(y_ground_truth)
    num_labels = len(set_y)
    sns_plot = sns.scatterplot(
        x=tsne_results[:,0], y=tsne_results[:,1],
        hue=y_ground_truth,
        palette=sns.color_palette("hls", num_labels),
        legend="full",
        alpha = 0.5
        )

    handles, labels = sns_plot.get_legend_handles_labels()
    sns_plot.legend(handles, ['picking', 'replace_label', 'assemble_box', 'pack_in_box',
                              'close_box', 'attach_label', 'read_label', 'recording'])
    sns_plot.get_figure().savefig(save_dir, dpi=600, bbox_inches='tight', pad_inches=0.0)


def mds(latent, y_ground_truth, save_dir):
    """
        Plot MDS embeddings of the features
    """
    latent = latent.cpu().detach().numpy()
    mds = MDS(n_components=2)
    mds_results = mds.fit_transform(latent)
    plt.figure(figsize=(16,10))
    set_y = set(y_ground_truth)
    num_labels = len(set_y)
    sns_plot = sns.scatterplot(
        x=mds_results[:,0], y=mds_results[:,1],
        hue=y_ground_truth,
        # palette=sns.color_palette("hls", num_labels),
        # data=df_subset,
        legend="full",
        alpha=0.5
        )

    sns_plot.get_figure().savefig(save_dir)