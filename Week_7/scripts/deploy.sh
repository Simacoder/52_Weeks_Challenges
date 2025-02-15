import sagemaker
from sagemaker.sklearn import SKLearnModel

model_uri = 's3://my-bucket/model/model.joblib'

model = SKLearnModel(model_data=model_uri,
                     role='arn:aws:iam::123456789012:role/SageMakerRole',
                     entry_point='src/inference.py')

predictor = model.deploy(instance_type='ml.m5.large', initial_instance_count=1)