import kfp
from kfp.dsl import component

@component()
def model_train() -> str:
    from google.cloud import aiplatform
    from google.cloud.aiplatform import hyperparameter_tuning as hpt
    from google.oauth2 import service_account
    import requests
    from datetime import datetime
    # Constants declaration
    config = requests.get("https://storage.googleapis.com/tymestack-artifacts/other.json").json()
    cred_json = requests.get("https://storage.googleapis.com/tymestack-artifacts/tymestack.json").json()
    credentials = service_account.Credentials.from_service_account_info(cred_json)
    #Init the project
    aiplatform.init(project=config["PROJECT_ID"], location=config["LOCATION"], staging_bucket=config["STAGING_BUCKET"],credentials=credentials)
    job_suffix = str(int(datetime.now().timestamp()))
    display_name = config["JOB_PREFIX"]+"-"+job_suffix
    job = aiplatform.CustomJob(
        display_name=display_name,
        worker_pool_specs=config["WORKER_POOL_SPEC"],
        base_output_dir=config["MODEL_DIR"],
    )


    hpt_job = aiplatform.HyperparameterTuningJob(
        display_name=display_name,
        custom_job=job,
        metric_spec={
            "mean_squared_error": "minimize",
        },
        parameter_spec={
            "max-depth": hpt.IntegerParameterSpec(min=10, max=100,scale="linear"),
            "n-estimators": hpt.IntegerParameterSpec(min=50, max=500,scale="linear"),
            "subsample":hpt.DoubleParameterSpec(min=0.5,max=1.0,scale="linear"),
            "learning-rate":hpt.DoubleParameterSpec(min=0.01,max=0.3,scale="linear")
        },
        search_algorithm=None,# Bayesian Search
        max_trial_count=config["MAX_TRIAL_COUNT"],
        parallel_trial_count=config["PARALLEL_TRIAL_COUNT"],
        
    )
    hpt_job.run()
    trials = hpt_job.trials

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

    best_model = best[0]
    return config['MODEL_DIR']+"/"+best_model+"/model/"


# Component to Deploy the model
@component
def model_deploy(model_gcs_path:str,flag:str) ->str:
    from google.cloud import aiplatform
    from google.oauth2 import service_account
    from datetime import datetime
    import requests
    # Constants declaration
    config = requests.get("https://storage.googleapis.com/tymestack-artifacts/other.json").json()
    cred_json = requests.get("https://storage.googleapis.com/tymestack-artifacts/tymestack.json").json()
    credentials = service_account.Credentials.from_service_account_info(cred_json)
    #Init the project
    aiplatform.init(project=config["PROJECT_ID"], location=config["LOCATION"], staging_bucket=config["STAGING_BUCKET"],credentials=credentials)
    # Upload the model to the Model registry
    model = aiplatform.Model.upload(
        display_name="housing-price-model-artifact-"+str(int(datetime.now().timestamp())),
        artifact_uri=model_gcs_path,
        serving_container_image_uri=config["PREDICTION_IMAGE_URI"],
    )
    def deploy_model(end_point_id):
        # Fetch the endpoint
        endpoint = aiplatform.Endpoint(endpoint_name=end_point_id)
        
        # Deploy the model to the endpoint
        deployed_model = model.deploy(
            endpoint=endpoint,
            deployed_model_display_name="xgb-housing-model",
            machine_type=config["PREDICTION_MACHINE_TYPE"],
            traffic_percentage=100
        )
    # Deploy the model to the endpoint
    if flag == "staging-true":
        end_point_id = config["PRODUCTION_END_POINT_ID"]
        deploy_model(end_point_id=end_point_id)
        return "Best Model is being deployed in production end point"

    elif flag == "staging-false":
        end_point_id = config["STAGING_END_POINT_ID"]
        deploy_model(end_point_id=end_point_id)
        return "Model is deploying in staging endpoint for testing purpose"
    else:
        return "The current model is production is alreay the best model. exiting..."

    


# Component for Model testing
@component
def model_test(stg:str,msg:str) -> float:
    from google.cloud import aiplatform
    import pandas as pd
    from sklearn.metrics import mean_squared_error
    from google.oauth2 import service_account
    import requests
    # Constants declaration
    config = requests.get("https://storage.googleapis.com/tymestack-artifacts/other.json").json()
    cred_json = requests.get("https://storage.googleapis.com/tymestack-artifacts/tymestack.json").json()
    credentials = service_account.Credentials.from_service_account_info(cred_json)
    #Init the project
    aiplatform.init(project=config["PROJECT_ID"], location=config["LOCATION"], staging_bucket=config["STAGING_BUCKET"],credentials=credentials)

    # Init the endpoint
    if stg == "staging":
        end_point_id = config["STAGING_END_POINT_ID"]
    elif stg == "production":
        end_point_id = config["PRODUCTION_END_POINT_ID"]
    # Fetch the endpoint
    endpoint = aiplatform.Endpoint(endpoint_name=end_point_id)

    # Load the test dataset
    test_df = pd.read_csv(config["TEST_DATASET_URL"])
    y_test = test_df['medv'].values
    test_df.drop(columns=['medv'],inplace=True)
    X_test = test_df.values

    # Infer the predictions
    instances = X_test.tolist()
    predictions = endpoint.predict(instances=instances)

    # Calculate the RMSE
    rmse = round(float(mean_squared_error(y_true=y_test,y_pred=predictions.predictions)),2)
    return rmse


# Model Comparison
@component
def model_compare(stg_rmse:float,prd_rmse:float) -> str:
    if prd_rmse < stg_rmse:
        return "production-false"
    else:
        return "staging-true"
