import pandas as pd
from sklearn.model_selection import train_test_split


class DataArray:
    def __init__(self, file_path, test_ratio):
        self.file_path : str = file_path
        self.test_ratio: float =  test_ratio

    def __call__(self):
        df = pd.read_csv(self.file_path)
        train_df, test_df = train_test_split(df, test_size=self.test_ratio, random_state=42)
        return train_df, test_df




