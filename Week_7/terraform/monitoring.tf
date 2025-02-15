from sagemaker.model_monitor import DataCaptureConfig

data_capture_config = DataCaptureConfig(
    enable_capture=True,
    sampling_percentage=100,
    destination_s3_uri='s3://my-bucket/data-capture/'
)

predictor.update_data_capture_config(data_capture_config=data_capture_config)