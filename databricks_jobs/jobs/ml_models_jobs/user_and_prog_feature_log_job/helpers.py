from typing import NamedTuple, List

from pyspark.sql import functions as F


class PivotFormatter(NamedTuple):

    agg_column: str
    col_name: str
    pivot_cols: List[str]
    prefix: str = None

    def rename_column(self):
        return F.concat(F.lit(self.full_prefix), F.col(self.col_name).cast("string"))

    @property
    def full_prefix(self):
        if self.prefix is None:
            return "_".join([self.agg_column, self.col_name]) + "="
        else:
            return "_".join([self.prefix, self.agg_column, self.col_name]) + "="

    @property
    def renamed_pivot_columns(self):
        return [self.full_prefix + str(c) for c in self.pivot_cols]

    def pivot_and_sum(self, df, group_by_col):
        """
        pivot_cols[column_name]
        """
        df = df. \
            withColumn(self.col_name, self.rename_column())
        return df. \
            groupBy(group_by_col). \
            pivot(self.col_name, values=self.renamed_pivot_columns). \
            agg(F.sum(self.agg_column))


class DictFormatter(NamedTuple):
    agg_column: str
    dict_col: str
    dict_col_values: List
    prefix: str = None

    @property
    def final_col_name(self):
        if self.prefix is None:
            return self.agg_column + "_" + self.dict_col
        else:
            return self.prefix + "_" + self.agg_column + "_" + self.dict_col

    def build_sum_dict(self, df, group_by_col):
        """
        pivot_cols[column_name]

        """
        return df. \
            groupBy(group_by_col, self.dict_col). \
            agg(F.sum(self.agg_column).alias(self.agg_column)). \
            where(F.col(self.dict_col).isin(*self.dict_col_values)). \
            groupBy(group_by_col). \
            agg(F.map_from_entries(F.collect_list(F.struct(self.dict_col, self.agg_column))).alias(self.final_col_name))


class SocioDemoBuilder(NamedTuple):
    age_low: int
    age_high: int
    gender: str

    def age_gender_filter(self):
        return F.when((self.age_low < F.col('AGE')) & (F.col('AGE') < self.age_high) &
                      (F.col('GENDER') == self.gender), 1).otherwise(0)

    @property
    def segment_name(self):
        elems = [self.gender, self.age_low + 1]
        if self.age_high != 100:
            elems.append(self.age_high)
        return "_".join(map(str, elems))


class SocioDemoAggregator(NamedTuple):
    socio_demo: SocioDemoBuilder
    col_name: str
    prefix: str = "channel"

    @property
    def formatted_col_name(self):
        if self.prefix is None or self.prefix == "":
            return self.col_name + "_socio=" + self.socio_demo.segment_name
        else:
            return self.prefix + "_" + self.col_name + "_socio=" + self.socio_demo.segment_name

    def op_fn(self):
        return (self.socio_demo.age_gender_filter() * F.col(self.col_name)).alias(self.formatted_col_name)
