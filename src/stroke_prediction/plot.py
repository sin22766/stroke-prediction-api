import matplotlib.pyplot as plt
import pandas as pd
import seaborn as sns


def plot_feature_importance(feature_importances: pd.Series) -> plt.Figure:
    sorted = feature_importances.sort_values(ascending=False)

    fig, ax = plt.subplots(figsize=(10, 6))
    sns.barplot(
        sorted,
        ax=ax,
        orient='h'
    )
    ax.set_xlabel("Feature Importance")
    ax.set_title("Feature Importance")
    plt.show()
    plt.tight_layout()

    return fig