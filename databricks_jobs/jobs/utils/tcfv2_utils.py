from pyspark.sql.types import BooleanType
import json

from databricks_jobs.jobs.utils.spark_utils import typed_udf


def is_consent(purposes, special_features, custom_purposes):
    @typed_udf(BooleanType())
    def tcfv2_udf(purposes, special_features, custom_purposes):
        purposes = json.loads(purposes)
        special_features = json.loads(special_features)
        custom_purposes = json.loads(custom_purposes)
        if sorted(purposes["enabled"]) == [1, 2, 3, 4, 5, 6, 7, 8, 9, 10] \
            and sorted(special_features["enabled"]) == [1, 2] \
                and sorted(custom_purposes["enabled"]) == [1, 2]:
            return True
        else:
            return False
    return tcfv2_udf(purposes, special_features, custom_purposes) == True
