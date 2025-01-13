DISK_TYPE = "pd-ssd"  # [ pd-ssd, pd-standard]
DISK_SIZE = 100  # GB

disk_spec = {"boot_disk_type": DISK_TYPE, "boot_disk_size_gb": DISK_SIZE}
# Set path to save model
MODEL_DIR = "gs://tymestack-artifacts/aiplatform-custom-job"

# Set the worker pool specs
worker_pool_spec = [
    {
        "replica_count": 1,
        "machine_spec": {"machine_type": "n1-standard-4", "accelerator_count": 0},
        "disk_spec": disk_spec,
        "python_package_spec": {
            "executor_image_uri": "us-docker.pkg.dev/vertex-ai/training/xgboost-cpu.1-6:latest",
            "package_uris": ["gs://tymestack-artifacts" + "/trainer-0.1.tar.gz"],
            "python_module": "trainer.task",
            
        },
    }
]
from google.cloud import aiplatform
from google.cloud.aiplatform import hyperparameter_tuning as hpt
from google.oauth2 import service_account
import requests
cred_json = requests.get("https://storage.googleapis.com/tymestack-artifacts/tymestack.json").json()
credentials = service_account.Credentials.from_service_account_info(cred_json)
aiplatform.init(project="tymestack-443211", location="us-central1", staging_bucket="gs://tymestack-artifacts",credentials=credentials)
job = aiplatform.CustomJob(
    display_name="xgb-housing-01",
    worker_pool_specs=worker_pool_spec,
    base_output_dir=MODEL_DIR,
)


hpt_job = aiplatform.HyperparameterTuningJob(
    display_name="xgb-housing-01",
    custom_job=job,
    metric_spec={
        "mean_squared_error": "minimize",
    },
    parameter_spec={
        "max-depth": hpt.IntegerParameterSpec(min=10, max=100,scale='linear'),
        "n-estimators": hpt.IntegerParameterSpec(min=50, max=500,scale='linear'),
        "subsample":hpt.DoubleParameterSpec(min=0.5,max=1.0,scale='linear'),
        "learning-rate":hpt.DoubleParameterSpec(min=0.01,max=0.3,scale='linear')
    },
    search_algorithm=None,# Bayesian Search
    max_trial_count=1,
    parallel_trial_count=1,
)
hpt_job.run()
print(hpt_job.trials)
# Initialize a tuple to identify the best configuration
best = (None, None, None, 0.0)
# Iterate through the trails and update the best configuration
for trial in hpt_job.trials:
    # Keep track of the best outcome
    if float(trial.final_measurement.metrics[0].value) > best[3]:
        try:
            best = (
                trial.id,
                float(trial.parameters[0].value),
                float(trial.parameters[1].value),
                float(trial.final_measurement.metrics[0].value),
            )
        except:
            best = (
                trial.id,
                float(trial.parameters[0].value),
                None,
                float(trial.final_measurement.metrics[0].value),
            )

# print details of the best configuration
print(best)
# the first index is the best model to download