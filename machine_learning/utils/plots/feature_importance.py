import matplotlib.pyplot as plt
import seaborn as sns
import pandas as pd

def plot_feature_importance(importance_df: pd.DataFrame) -> None:
    """Display feature importance barplot using seaborn.

    Args:
        importance_df (pd.DataFrame): DataFrame с двумя столбцами:
            'feature' — названия признаков,
            'importance' — значения важности.
    """
    importance_df = importance_df.sort_values(by='importance', ascending=False)
    
    plt.figure(figsize=(10, 6))
    sns.barplot(
        data=importance_df,
        x='importance',
        y='feature',
        hue='feature',
        palette='viridis'
    )
    plt.title('Feature Importance (Gain)')
    plt.xlabel('Importance')
    plt.ylabel('Feature')
    plt.tight_layout()
    plt.show()
