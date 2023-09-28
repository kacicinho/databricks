import pyspark.sql.functions as F


def typed_udf(return_type):
    def _typed_udf_wrapper(func):
        return F.udf(func, return_type)
    return _typed_udf_wrapper


def file_exists(path, dbutils):
    try:
        dbutils.fs.ls(path)
        return True
    except Exception as e:
        if 'java.io.FileNotFoundException' in str(e):
            return False
        else:
            raise


def create_empty_df(spark, schema):
    emptyRDD = spark.sparkContext.emptyRDD()
    return spark.createDataFrame(emptyRDD, schema)
