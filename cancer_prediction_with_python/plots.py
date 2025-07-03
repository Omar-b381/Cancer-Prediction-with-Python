import matplotlib.pyplot as plt
import seaborn as sns

def plot_class_distribution(df):
    sns.countplot(x=df['diagnosis'])
    plt.title("Class Distribution")
    plt.show()

def plot_feature_distributions(df):
    for col in df.select_dtypes('float'):
        plt.figure(figsize=(6, 4))
        sns.boxplot(x=df[col])
        plt.title(f"Distribution of {col}")
        plt.show()
