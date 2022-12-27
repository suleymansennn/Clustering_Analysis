from rfm_analysis import *
from helpers import *
import pandas as pd
import time
import matplotlib.pyplot as plt
from sklearn.cluster import KMeans
from sklearn.preprocessing import MinMaxScaler
from yellowbrick.cluster import KElbowVisualizer


def kmeans(dataframe, manuel=False):
    scaler = MinMaxScaler()
    kmeans = KMeans()
    df = pd.DataFrame(scaler.fit_transform(dataframe[["Recency", "Frequency", "Monetary"]]),
                      columns=["Recency", "Frequency", "Monetary"])
    if manuel:
        plot_kmeans(df)
        k_cluster = int(input("Enter K Cluster:"))
        opt_kmeans = KMeans(n_clusters=k_cluster).fit(df)
        df["cluster"] = opt_kmeans.labels_
        return df["cluster"]
    else:
        elbow = KElbowVisualizer(kmeans, k=(2, 20))
        elbow.fit(df)
        elbow.show()
        opt_kmeans = KMeans(n_clusters=elbow.elbow_value_).fit(df)
        df["cluster"] = opt_kmeans.labels_
        print(f"{df['cluster'].nunique()} Clusters Selected")
        return df["cluster"]


def main(manuel=False):
    print("RFM Analysis Started...")
    start_time = time.perf_counter()
    df = preprocess("flo_data_20k.csv")
    rfm_ = rfm_table(df)
    rfm = rfm_segment(rfm_)
    end_time = time.perf_counter()
    elapsed_time = end_time - start_time
    print("RFM Analysis Finished...")
    print(f"Process Time: {elapsed_time:.6f} seconds")

    print("Clustering Analysis Started...")
    start_time = time.perf_counter()
    if manuel:
        cluster = kmeans(rfm, True)
    else:
        cluster = kmeans(rfm, False)

    end_time = time.perf_counter()
    elapsed_time = end_time - start_time
    print("Clustering Analysis Finished...")
    print(f"Process Time: {elapsed_time:.6f} seconds")
    rfm["Cluster"] = cluster

    print(rfm.groupby(["Cluster", "Segment"]).agg({"Recency": ["mean", "median", "count"],
                                                   "Frequency": ["mean", "median", "count"],
                                                   "Monetary": ["mean", "median", "count"]}), end="\n\n")

    analysis_report(rfm)


if __name__ == "__main__":
    main(True)
