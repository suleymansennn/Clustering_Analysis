import numpy as np
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
import matplotlib
from helpers import check_df, cat_plots, num_summary
import datetime as dt

matplotlib.use("Qt5Agg")

colors = ['#FFB6B9', '#FAE3D9', '#BBDED6', '#61C0BF', "#CCA8E9", "#F67280"]
pd.set_option("display.max_columns", None)
pd.set_option("display.width", 500)

df_ = pd.read_csv("clustering/flo_data_20k.csv")
df = df_.copy()
df.head()

check_df(df)

df["new_total_expenditure"] = df["customer_value_total_ever_offline"] + df["customer_value_total_ever_online"]
df["new_total_purchases"] = df["order_num_total_ever_online"] + df["order_num_total_ever_offline"]

print(
    f'{df["new_total_purchases"].sum()} invoices were carried out from {df["first_order_date"].min()} to'
    f' {df["last_order_date"].max()}')

print(
    f'{df["master_id"].nunique()} customer were served from {df["first_order_date"].min()} to '
    f'{df["last_order_date"].max()}')

date_vars = df.columns[df.columns.str.contains("date")]

for col in date_vars:
    df[col] = pd.to_datetime(df[col])

cat_plots(df, "order_channel")
cat_plots(df, "last_order_channel")

for col in df.columns[df.columns.str.contains("new")]:
    num_summary(df, col, "order_channel")

df[["master_id", "new_total_expenditure", "new_total_purchases"]].sort_values("new_total_expenditure",
                                                                              ascending=False).head(10)

df[["master_id", "new_total_expenditure", "new_total_purchases"]].sort_values("new_total_purchases",
                                                                              ascending=False).head(10)

df["first_order_year"] = df["first_order_date"].dt.year
df["first_order_month"] = df["first_order_date"].dt.month_name()
df["first_order_day"] = df["first_order_date"].dt.day_name()

df["last_order_year"] = df["last_order_date"].dt.year
df["last_order_month"] = df["last_order_date"].dt.month_name()
df["last_order_day"] = df["last_order_date"].dt.day_name()

cat_plots(df, "first_order_year")
cat_plots(df, "first_order_month")
cat_plots(df, "first_order_day")
cat_plots(df, "last_order_year")
cat_plots(df, "last_order_month")
cat_plots(df, "last_order_day")
