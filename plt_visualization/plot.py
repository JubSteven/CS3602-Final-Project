import pandas as pd
import matplotlib.pyplot as plt


def smooth_curve(points, factor=0.6):
    """ Smooths the curve using exponential moving average """
    smoothed_points = []
    for point in points:
        if smoothed_points:
            previous = smoothed_points[-1]
            smoothed_points.append(previous * factor + point * (1 - factor))
        else:
            smoothed_points.append(point)
    return smoothed_points

plt.figure(figsize=(15, 15))

# Tracking the index for subplot, excluding Best Accuracy
plot_index = 1

file_paths_titles = {
    "plt_visualization/assets/run-LA_LSTM_GRU_2023-12-20-16-59-48_max_epoch=100_bs=32_lr=0.0001_decay=[50]_gamma=0.1_encoder=bert-base-chinese_decoder=GRU_dropout=0.2-tag-dev_Best_Acc.csv": "Best Accuracy",
    "plt_visualization/assets/run-LA_LSTM_GRU_2023-12-20-16-59-48_max_epoch=100_bs=32_lr=0.0001_decay=[50]_gamma=0.1_encoder=bert-base-chinese_decoder=GRU_dropout=0.2-tag-dev_Dev_Acc.csv": "Development Accuracy",
    "plt_visualization/assets/run-LA_LSTM_GRU_2023-12-20-16-59-48_max_epoch=100_bs=32_lr=0.0001_decay=[50]_gamma=0.1_encoder=bert-base-chinese_decoder=GRU_dropout=0.2-tag-dev_Dev_F.csv": "F-Measure",
    "plt_visualization/assets/run-LA_LSTM_GRU_2023-12-20-16-59-48_max_epoch=100_bs=32_lr=0.0001_decay=[50]_gamma=0.1_encoder=bert-base-chinese_decoder=GRU_dropout=0.2-tag-dev_Dev_P.csv": "Precision",
    "plt_visualization/assets/run-LA_LSTM_GRU_2023-12-20-16-59-48_max_epoch=100_bs=32_lr=0.0001_decay=[50]_gamma=0.1_encoder=bert-base-chinese_decoder=GRU_dropout=0.2-tag-dev_Dev_R.csv": "Recall"
}
for file_path, title in file_paths_titles.items():
    # Read data
    data = pd.read_csv(file_path)

    # sort data by step
    data = data.sort_values(by=['Step'])

    # Get values and steps
    values = data['Value'].tolist()
    steps = data['Step'].tolist()



    # Smooth values
    smoothed_values = smooth_curve(values, factor=0.6)

    # Find max value and its position in the smoothed data
    max_value = max(smoothed_values)
    max_step = steps[smoothed_values.index(max_value)]

    # Special handling for Best Accuracy
    if title == "Best Accuracy":
        ax = plt.subplot(3, 1, 3)
    else:
        ax = plt.subplot(3, 2, plot_index)
        plot_index += 1

    # Plotting
    ax.plot(steps, smoothed_values, label=title.replace('_', ' '))
    ax.scatter(max_step, max_value, color='red')
    ax.text(max_step, max_value, f'Max: {max_value:.2f} at {max_step}', fontsize=9, verticalalignment='bottom')
    ax.set_xlabel('Step')
    ax.set_ylabel('Value')
    ax.set_title(title)
    ax.legend()

plt.tight_layout()
plt.savefig('plt_visualization/assets/plot.png')