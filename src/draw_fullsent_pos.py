import os
import matplotlib.pyplot as plt
import numpy as np
import seaborn as sns

from utils import load_features


def draw_stats(data_list, names=['train', 'test'], filename_app='DUC', cut_thres=15):
    # Plot
    sns.set(style="whitegrid", context="talk")
    fig, axes = plt.subplots(1, 2, figsize=(15, 7), sharey=True, sharex=False, dpi=300)

    for i, data in enumerate(data_list):
        ax = axes[i]
        data = np.array(data)
        num_data = len(data)

        x = np.arange(1, cut_thres + 1)
        data_percent = []
        for j in range(cut_thres):
            count = np.sum(data == j)
            data_percent.append(count / num_data * 100.)

        sns.barplot(x=x, y=data_percent, palette=sns.color_palette("GnBu_r", cut_thres + 10), ax=axes[i])

        ax.spines["top"].set_visible(False)
        ax.spines["right"].set_visible(False)
        ax.get_xaxis().tick_bottom()
        ax.get_yaxis().tick_left()

        ax.set_xlim(-.5, cut_thres - .5)

        ax.set_title('{}'.format(names[i]), fontsize=17, weight='bold')
        if i == 0:
            ax.set_ylabel("Percentage of Sentence (%)", fontsize=17, weight='bold')
            ax.set_xlabel("", fontsize=7)
        else:
            ax.set_ylabel("")
            ax.set_xlabel("", fontsize=7)

    plt.tight_layout()

    fig.savefig('full_sent_pos_{}.png'.format(filename_app))


if __name__ == '__main__':

    splits = ['train', 'test']
    datasets = ['DUC', 'TAC']

    for split in splits:
        full_sent_pos_list = []
        for dataset in datasets:
            output_path = os.path.join('../data', dataset, 'xlnet')
            filename = os.path.join(output_path, split + '_stats.pkl')
            full_sent_pos, _, _, _ = load_features(filename)
            full_sent_pos_list.append(full_sent_pos)
            print('data is loaded from {}'.format(filename))

        draw_stats(full_sent_pos_list, datasets, split)
