# ruff: noqa: F841

import numpy as np

import pandas as pd


N = 5_000_000
N_USERS = 50_000
N_CATEGORIES = 200
REGIONS = ["SHA", "SGP", "FRA", "IAD", "NRT", "SYD"]
EVENT_TYPES = [
    "purchase",
    "refund",
    "click",
    "view",
    "search",
    "add_cart",
    "checkout",
    "share",
]


def generate_dataframe(rng):
    df = pd.DataFrame(
        {
            "user_id": rng.integers(0, N_USERS, size=N),
            "timestamp": pd.date_range("2025-01-01", periods=N, freq="s"),
            "amount": rng.uniform(1.0, 500.0, size=N),
            "region": rng.choice(REGIONS, size=N),
            "event_type": rng.choice(EVENT_TYPES, size=N),
            "category_id": rng.integers(0, N_CATEGORIES, size=N),
            "rating": rng.uniform(1.0, 5.0, size=N),
        }
    )
    mask = rng.random(N) < 0.15
    df.loc[mask, "rating"] = np.nan
    return df


class _Base:
    def setup(self):
        rng = np.random.default_rng(seed=42)
        self.df = generate_dataframe(rng)


class DataFrameGeneration:
    def setup(self):
        self.rng = np.random.default_rng(seed=42)

    def time_generate_dataframe(self):
        df = generate_dataframe(self.rng)


class GroupByAggregation(_Base):
    def time_groupby_agg(self):
        result = self.df.groupby("user_id").agg(
            total_amount=("amount", "sum"),
            avg_amount=("amount", "mean"),
            txn_count=("amount", "count"),
            unique_categories=("category_id", "nunique"),
            avg_rating=("rating", "mean"),
            region_count=("region", "nunique"),
        )


class _MergeBase(_Base):
    def setup(self):
        super().setup()
        self.user_dim = (
            self.df.groupby("user_id")
            .agg(first_event=("timestamp", "min"), last_event=("timestamp", "max"))
            .reset_index()
        )
        self.user_dim["tenure_days"] = (
            self.user_dim["last_event"] - self.user_dim["first_event"]
        ).dt.total_seconds() / 86400.0

        self.cat_dim = pd.DataFrame(
            {
                "category_id": self.df["category_id"].unique(),
                "category_name": [
                    f"cat_{cid}" for cid in self.df["category_id"].unique()
                ],
            }
        )


class MergeUser(_MergeBase):
    def time_merge_user(self):
        merged = self.df.merge(
            self.user_dim[["user_id", "tenure_days"]],
            on="user_id",
            how="left",
        )


class MergeCategory(_MergeBase):
    def setup(self):
        super().setup()
        self.merged = self.df.merge(
            self.user_dim[["user_id", "tenure_days"]],
            on="user_id",
            how="left",
        )

    def time_merge_category(self):
        merged = self.merged.merge(self.cat_dim, on="category_id", how="left")


class TimeSeriesResampleRolling(_Base):
    def setup(self):
        super().setup()
        self.hourly = self.df.set_index("timestamp").resample("h")["amount"].sum()

    def time_resample(self):
        hourly = self.df.set_index("timestamp").resample("h")["amount"].sum()

    def time_rolling_24h(self):
        rolling_mean = self.hourly.rolling(window=24, min_periods=1).mean()
        rolling_std = self.hourly.rolling(window=24, min_periods=1).std()


class StringCategorical(_Base):
    def setup(self):
        super().setup()
        self.df_cat = self.df.copy()
        self.df_cat["region"] = self.df_cat["region"].astype("category")
        self.df_cat["event_type"] = self.df_cat["event_type"].astype("category")

    def time_to_categorical(self):
        df_cat = self.df.copy()
        df_cat["region"] = df_cat["region"].astype("category")
        df_cat["event_type"] = df_cat["event_type"].astype("category")

    def time_string_ops(self):
        upper_region = self.df["region"].str.upper()
        contains_a = self.df["region"].str.contains("A", regex=False)
        str_len = self.df["event_type"].str.len()

    def time_value_counts(self):
        region_counts = self.df_cat["region"].value_counts()
        event_counts = self.df_cat["event_type"].value_counts()


class PivotCrosstab(_Base):
    def time_pivot_table(self):
        pivot = self.df.pivot_table(
            values="amount",
            index="region",
            columns="event_type",
            aggfunc="mean",
        )

    def time_crosstab(self):
        xtab = pd.crosstab(self.df["region"], self.df["event_type"])


class _ApplyBase(_Base):
    def setup(self):
        super().setup()
        sample_size = min(len(self.df), 500_000)
        self.df_sample = self.df.head(sample_size).copy()


class ApplyRowwise(_ApplyBase):
    def setup(self):
        super().setup()

        def _score_row(row):
            base = row["amount"] * 0.8
            if row["event_type"] == "purchase":
                base *= 1.5
            elif row["event_type"] == "refund":
                base *= 0.3
            if pd.notna(row["rating"]):
                base += row["rating"] * 10
            return base

        self.score_row = _score_row

    def time_apply_rowwise(self):
        df_sample = self.df_sample
        df_sample["score_apply"] = df_sample.apply(self.score_row, axis=1)


class ApplyVectorized(_ApplyBase):
    def time_vectorized(self):
        df_sample = self.df_sample
        multiplier = pd.Series(1.0, index=df_sample.index)
        multiplier = multiplier.where(df_sample["event_type"] != "purchase", 1.5)
        multiplier = multiplier.where(df_sample["event_type"] != "refund", 0.3)
        rating_bonus = df_sample["rating"].fillna(0.0) * 10
        df_sample["score_vec"] = df_sample["amount"] * 0.8 * multiplier + rating_bonus
