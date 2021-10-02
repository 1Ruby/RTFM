import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import os
import argparse

parser = argparse.ArgumentParser(description='Make statistic images.', formatter_class=argparse.ArgumentDefaultsHelpFormatter)

parser.add_argument('--logs_dir', type=str, help='The folder of your logs.csv')
parser.add_argument('--save_dir', type=str, help='The folder where images are stored')
parser.add_argument('--fig_id', type=str, default='', help='Suffix of images')
parser.add_argument('--data_batch', type=int, default=100)
parser.add_argument('--subtitle', type=str, default='', help='Subtitle of images')

flag = parser.parse_args()
if not os.path.exists(flag.save_dir):
    os.makedirs(flag.save_dir)
    
fig_id = '_' + flag.fig_id if flag.fig_id != '' else ''

logs_path = flag.logs_dir + '/logs.csv'
return_fig_path = flag.save_dir + '/mean_return' + fig_id + '.png'
win_fig_path = flag.save_dir + '/win_rate' + fig_id + '.png'
len_fig_path = flag.save_dir + '/mean_len' + fig_id + '.png'
regret_fig_path = flag.save_dir + '/regret' + fig_id + '.png'

logs = pd.read_csv(logs_path)
length = len(logs.index)
batch = flag.data_batch
frames = []

mean_return = []
mean_win_rate = []
mean_len = []
mean_regret = []
mean_return_test = []
mean_win_rate_test = []
mean_len_test = []
mean_regret_test = []


for i in range(length // batch):
    frames.append(logs['frames'][(i + 1) * batch])
    mean_return.append(logs['mean_episode_return'][i * batch: (i + 1) * batch].mean())
    mean_len.append(logs['mean_episode_len'][i * batch: (i + 1) * batch].mean())
    mean_win_rate.append(logs['mean_win_rate'][i * batch: (i + 1) * batch].mean())
    mean_regret.append(logs['mean_episode_regret'][i * batch: (i + 1) * batch].mean())
    
    mean_return_test.append(logs['episode_return_test'][i * batch: (i + 1) * batch].mean())
    mean_len_test.append(logs['eplen_test'][i * batch: (i + 1) * batch].mean())
    mean_win_rate_test.append(logs['win_rate_test'][i * batch: (i + 1) * batch].mean())
    mean_regret_test.append(logs['episode_regret_test'][i * batch: (i + 1) * batch].mean())
    
plt.figure(1)
plt.plot(frames, mean_return, label='train')
plt.plot(frames, mean_return_test, label='test')
plt.legend()
plt.suptitle('Mean episode return')
plt.title(flag.subtitle)
plt.savefig(return_fig_path)

plt.figure(2)
plt.plot(frames, mean_len, label='train')
plt.plot(frames, mean_len_test, label='test')
plt.legend()
plt.suptitle('Mean episode length')
plt.title(flag.subtitle)
plt.savefig(len_fig_path)

plt.figure(3)
plt.plot(frames, mean_win_rate, label='train')
plt.plot(frames, mean_win_rate_test, label='test')
plt.legend()
plt.suptitle('Mean win rate')
plt.title(flag.subtitle)
plt.savefig(win_fig_path)

plt.figure(4)
plt.plot(frames, mean_regret, label='train')
plt.plot(frames, mean_regret_test, label='test')
plt.legend()
plt.suptitle('Mean regret')
plt.title(flag.subtitle)
plt.savefig(regret_fig_path)