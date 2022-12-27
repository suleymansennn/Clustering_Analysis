import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
import numpy as np
from sklearn.cluster import KMeans

colors = ['#FFB6B9', '#FAE3D9', '#BBDED6', '#61C0BF', "#CCA8E9", "#F67280"]


def check_df(dataframe, head=5, tail=5):
    print("*" * 70)
    print(" Shape ".center(70, "*"))
    print("*" * 70)
    print(dataframe.shape)

    print("*" * 70)
    print(" Types ".center(70, "*"))
    print("*" * 70)
    print(dataframe.dtypes)

    print("*" * 70)
    print(" Head ".center(70, "*"))
    print("*" * 70)
    print(dataframe.head(head))

    print("*" * 70)
    print(" Tail ".center(70, "*"))
    print("*" * 70)
    print(dataframe.tail(tail))

    print("*" * 70)
    print(" NA ".center(70, "*"))
    print("*" * 70)
    print(dataframe.isnull().sum())

    print("*" * 70)
    print(" Quantiles ".center(70, "*"))
    print("*" * 70)
    print(dataframe.describe([.01, .05, .1, .5, .9, .95, .99]).T)

    print("*" * 70)
    print(" Uniques ".center(70, "*"))
    print("*" * 70)
    print(dataframe.nunique())


def cat_plots(dataframe, cat_col):
    print("".center(100, "#"))
    print(dataframe[cat_col].value_counts())

    plt.figure(figsize=(15, 10))
    plt.suptitle(cat_col.capitalize(), size=16)
    plt.subplot(1, 2, 1)
    plt.title("Percentages")
    plt.pie(dataframe[cat_col].value_counts().values.tolist(),
            labels=dataframe[cat_col].value_counts().keys().tolist(),
            labeldistance=1.1,
            wedgeprops={'linewidth': 3, 'edgecolor': 'white'},
            colors=colors,
            autopct='%1.0f%%')

    plt.subplot(1, 2, 2)
    plt.title("Countplot")
    sns.countplot(data=dataframe, x=cat_col, palette=colors)
    plt.tight_layout(pad=3)
    plt.show(block=True)


def num_summary(dataframe, col_name, target):
    quantiles = [.01, .05, .1, .5, .9, .95, .99]
    print(dataframe.groupby(target)[col_name].describe(percentiles=quantiles))
    xlim = dataframe[col_name].describe(quantiles).T["99%"]

    plt.figure(figsize=(15, 10))
    plt.suptitle(col_name.capitalize(), size=16)
    plt.subplot(1, 3, 1)
    plt.title("Histogram")
    sns.histplot(dataframe[col_name], color="#FFB6B9")
    plt.xlim(0, xlim)

    plt.subplot(1, 3, 2)
    plt.title("Box Plot")
    sns.boxplot(data=dataframe, y=col_name, color="#F67280")
    plt.ylim(0, xlim)

    plt.subplot(1, 3, 3)
    sns.barplot(data=dataframe, x=col_name, y=target, palette=colors, estimator=np.mean)
    plt.title(f"Sum of {col_name.capitalize()} by {target.capitalize()}")
    plt.tight_layout(pad=1.5)
    plt.show(block=True)


def plot_kmeans(data):
    kmeans = KMeans()
    ssd = []
    K = range(1, 30)

    for k in K:
        kmeans = KMeans(n_clusters=k).fit(data)
        ssd.append(kmeans.inertia_)

    plt.plot(K, ssd, "bx-")
    plt.xlabel("SSE/SSR/SSD vs. Different K Values")
    plt.title("Elbow Method for Optimum Number of Clusters")
    # plt.savefig("optimum_cluster")
    plt.show(block=True)


def analysis_report(dataframe):
    l = [["Recency", "Frequency"], ["Recency", "Monetary"], ["Frequency", "Monetary"]]
    for i in l:
        plt.subplot(1, 2, 1)
        plt.title("Cluster")
        sns.scatterplot(data=dataframe, x=i[0], y=i[1], hue="Cluster")

        plt.subplot(1, 2, 2)
        plt.title("RFM")
        sns.scatterplot(data=dataframe, x=i[0], y=i[1], hue="Segment")

        plt.tight_layout()
        # plt.savefig(str(i[0])+"_"+str(i[1]))
        plt.show(block=True)
    c_list = dataframe["Cluster"].unique().tolist()
    total_turnover = dataframe["Monetary"].sum()
    total_sales = dataframe["Frequency"].sum()
    for i in c_list:
        ratio = dataframe.loc[dataframe["Cluster"] == i, "Frequency"].sum() / total_sales
        exp = dataframe.loc[dataframe["Cluster"] == i, "Monetary"].sum() / total_turnover
        print(f" Cluster {i} represent {round(ratio * 100, 2)}% of sales, and {round((exp) * 100, 2)}% of turnover")

    c_list2 = dataframe["Segment"].unique().tolist()
    for i in c_list2:
        ratio = dataframe.loc[dataframe["Segment"] == i, "Frequency"].sum() / total_sales
        exp = dataframe.loc[dataframe["Segment"] == i, "Monetary"].sum() / total_turnover
        print(
            f" {i} Segment represent {round(ratio * 100, 2)}% of sales, and {round((exp) * 100, 2)}% of turnover")
