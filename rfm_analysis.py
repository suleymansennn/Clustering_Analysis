import pandas as pd
import datetime as dt


def preprocess(path):
    df_ = pd.read_csv(path)
    df = df_.copy()
    df["new_total_expenditure"] = df["customer_value_total_ever_offline"] + df["customer_value_total_ever_online"]
    df["new_total_purchases"] = df["order_num_total_ever_online"] + df["order_num_total_ever_offline"]
    date_vars = df.columns[df.columns.str.contains("date")]
    for col in date_vars:
        df[col] = pd.to_datetime(df[col])
    return df


def rfm_table(dataframe):
    max_date = (dataframe["last_order_date"].max() + dt.timedelta(days=2))
    rfm = pd.DataFrame({
        "Recency": (max_date - dataframe["last_order_date"]),
        "Frequency": dataframe["new_total_purchases"],
        "Monetary": dataframe["new_total_expenditure"]
    })
    rfm["Recency"] = rfm["Recency"].apply(lambda x: x.days)
    return rfm



def rfm_segment(rfm):
    rfm["Recency_Score"] = pd.qcut(rfm["Recency"], q=5, labels=[5, 4, 3, 2, 1])
    rfm["Frequency_Score"] = pd.qcut(rfm["Frequency"].rank(method="first"), q=5, labels=[1, 2, 3, 4, 5])
    rfm["Monetary_Score"] = pd.qcut(rfm["Monetary"], q=5, labels=[1, 2, 3, 4, 5])
    rfm["RF_Score"] = rfm["Recency_Score"].astype(str) + rfm["Frequency_Score"].astype(str)

    seg_map = {
        r"[1-2][1-2]": "hibernating",
        r"[1-2][3-4]": "at_Risk",
        r"[1-2]5": "cant_loose",
        r"3[1-2]": "about_to_sleep",
        r"33": "need_attention",
        r"[3-4][4-5]": "loyal_customers",
        r"41": "promising",
        r"51": "new_customers",
        r"[4-5][2-3]": "potential_loyalists",
        r"5[4-5]": "champions"
    }

    rfm["Segment"] = rfm["RF_Score"].replace(seg_map, regex=True)

    return rfm








