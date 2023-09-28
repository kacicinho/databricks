from typing import NamedTuple
import boto3
from typing import List

class MockBoto3():

    def client(name, aws_access_key_id, aws_secret_access_key, region_name):
        return MockClient()

class MockClient(NamedTuple):

    def put_metric_data(self, Namespace: str, MetricData: List[dict]):
         return [{**{"Namespace": Namespace}, **metric} for metric in MetricData]

