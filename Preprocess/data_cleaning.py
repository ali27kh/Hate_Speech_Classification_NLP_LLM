import pandas as pd

class DataCleaning:
    @staticmethod
    def clean_class_column(df: pd.DataFrame, column_name: str) -> pd.DataFrame:
        """
        Cleans the class column by assigning:
        - 1 if value is between 0.6 and 1
        - 0 if value is between 0 and 0.4
        - not considered if value is between 0.39 and 0.59 
        """
        df['class'] = df[column_name].apply(lambda x: 1 if 0.6 <= x <= 1 else (0 if 0 <= x <= 0.4 else 'not considered'))
        return df