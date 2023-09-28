from sklearn.preprocessing import LabelEncoder
import pandas as pd
import mlflow


class Bining():

    def __init__(self, columns):
        self.columns = columns

    def fit(self, df, y=None):
        df_t = df.copy()
        if self.columns is not None:
            self.bucket_dict = {}

            for col in self.columns:
                bins = list(df_t[df_t[col] > 0][col].quantile([0.2, 0.4, 0.6, 0.8, 0.9]).values)
                bins = [0] + bins + [99999999]
                self.bucket_dict[col] = bins

        return self

    def fit_transform(self, df, y=None):
        df_t = df.copy()
        if self.columns is not None:
            self.bucket_dict = {}

            for col in self.columns:
                bins = list(df_t[df_t[col] > 0][col].quantile([0.2, 0.4, 0.6, 0.8, 0.9]).values)
                bins = [0] + bins + [99999999]
                self.bucket_dict[col] = bins
                df_t[col] = pd.cut(df_t[col], bins, include_lowest=True)

        return df_t

    def transform(self, df):
        df_t = df.copy()
        if self.columns is not None:
            for col, bins in self.bucket_dict.items():
                df_t[col] = pd.cut(df_t[col], bins, include_lowest=True)

        return df_t

    def get_feature_names_out(self, columns):
        return columns


class MultiColumnLabelEncoder(LabelEncoder):
    """
    Wraps sklearn LabelEncoder functionality for use on multiple columns of a
    pandas dataframe.

    """

    def __init__(self, columns=None):
        self.columns = columns

    def fit(self, dframe, y=None):

        if self.columns is not None:
            self.encoding_dict = {}

            for col in self.columns:
                le = LabelEncoder()
                le.fit(dframe[col])

                # append this column's encoder
                self.encoding_dict[col] = le

        return self

    def fit_transform(self, df, y=None):
        df_t = df.copy()
        if self.columns is not None:
            self.encoding_dict = {}

            for col in self.columns:
                le = LabelEncoder()
                df_t[col] = le.fit_transform(df_t[col])
                self.encoding_dict[col] = le

        return df_t

    def transform(self, df):
        """
        Transform labels to normalized encoding.
        """
        df_t = df.copy()
        if self.columns is not None:
            for col in self.columns:
                le = self.encoding_dict[col]
                df_t[col] = le.transform(df_t[col])

        return df_t

    def get_feature_names_out(self, columns):
        return columns


class SklearnModelWrapper(mlflow.pyfunc.PythonModel):
    def __init__(self, model):
        self.model = model

    def predict(self, model_input):
        return self.model.predict_proba(model_input)[:, 1]
