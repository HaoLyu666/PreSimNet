import matplotlib.pyplot as plt
import numpy as np

def create_bar_chart():
    """Creates and saves a custom bar chart."""
    # Data from the image
    labels = ['AV-AV', 'AV-HV', 'HV-HV', 'HV-AV']
    values = [0.25, 0.75, 0.0, 0.0]
    colors = ['#CFEAF1', '#A9E9E4', '#F3BAC3', '#A5B1DA']

    # Create figure and axis
    fig, ax = plt.subplots(figsize=(8, 3.5))

    # Create bars
    bars = ax.bar(labels, values, color=colors, edgecolor='black', linewidth=3)

    # Add labels on top of each bar
    for bar in bars:
        yval = bar.get_height()
        ax.text(bar.get_x() + bar.get_width()/2.0, yval + 0.01, f'{yval}', ha='center', va='bottom', fontsize=20, weight='bold')

    # Remove y-axis, and spines
    ax.yaxis.set_visible(False)
    ax.spines['top'].set_visible(False)
    ax.spines['right'].set_visible(False)
    ax.spines['left'].set_visible(False)
    ax.spines['bottom'].set_color('black')

    # Set x-axis labels
    ax.set_xticks(np.arange(len(labels)))
    ax.set_xticklabels(labels, fontsize=20, weight='bold')

    # Set transparent background
    fig.patch.set_alpha(0.0)
    ax.patch.set_alpha(0.0)

    # Adjust layout and save the figure
    plt.tight_layout()
    output_filename = 'fig/vis/mapping/custom_bar_chart.png'
    plt.savefig(output_filename, dpi=300, transparent=True)
    plt.close(fig)
    print(f'Saved bar chart to {output_filename}')

if __name__ == '__main__':
    create_bar_chart()