import matplotlib.pyplot as plt

def line_plot(x, series_dict, title, xlabel, ylabel, save_path=None):
    """series_dict: {label: [y-values matching x]}"""
    fig, ax = plt.subplots(figsize=(6, 4))
    for label, y in series_dict.items():
        ax.plot(x, y, marker="o", label=label)
    ax.set_title(title); ax.set_xlabel(xlabel); ax.set_ylabel(ylabel)
    ax.legend()
    if save_path:
        plt.savefig(save_path, dpi=150, bbox_inches="tight")
    plt.show()
