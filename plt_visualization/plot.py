# Adjusting the plot layout to place Best Accuracy at the bottom spanning two grids

plt.figure(figsize=(15, 15))

# Tracking the index for subplot, excluding Best Accuracy
plot_index = 1

for file_path, title in file_paths_titles.items():
    # Read data
    data = pd.read_csv(file_path)

    # Extract values and steps
    values = data['Value']
    steps = data['Step']

    # Find max value and its position
    max_value = values.max()
    max_step = steps[values.idxmax()]

    # Special handling for Best Accuracy
    if title == "Best Accuracy":
        ax = plt.subplot(3, 1, 3)
    else:
        ax = plt.subplot(3, 2, plot_index)
        plot_index += 1

    # Plotting
    ax.plot(steps, values, label=title.replace('_', ' '))
    ax.scatter(max_step, max_value, color='red')
    ax.text(max_step, max_value, f'Max: {max_value:.2f} at {max_step}', fontsize=9, verticalalignment='bottom')
    ax.set_xlabel('Step')
    ax.set_ylabel('Value')
    ax.set_title(title)
    ax.legend()

plt.tight_layout()
plt.show()
