from pathlib import Path
import pandas as pd


data_dir = Path("data/")


def create_holdout(dataset, size=0.1):
    """Simple wrapper function used to split a hold out set from a dataframe.
    :param dataset: Original dataset (pandas dataframe)
    :param size: Size of hold-out set as fraction of original data. Default is
    0.1
    :returns: remainder and hold out datasets, both as DataFrame
    """
    holdout = dataset.sample(frac=size)
    remainder = dataset.drop(holdout.index)
    return holdout, remainder


all_data = pd.read_csv(data_dir / "housing-data.csv", index_col="Order")

hold_out, remainder = create_holdout(all_data)

remainder.to_csv(data_dir / "housing-data.csv")
hold_out.to_csv(data_dir / "hold-out.csv")
