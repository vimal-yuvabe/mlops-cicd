from kfp.dsl import pipeline,If
from components import model_train, model_deploy, model_test, model_compare
from kfp import local
local.init(local.SubprocessRunner(use_venv=False))

@pipeline(name="xgboost-model-training",description="Pipeline for e2e training and deployment of a xgboost regression model")
def xgboost_model_training():
    train_model = model_train()
    stg_deploy = model_deploy(model_gcs_path=train_model.output,flag="staging-true")
    rmse_staging = model_test(stg="staging",msg=stg_deploy.output)
    rmse_production = model_test(stg="production",msg="")
    best_model = model_compare(stg_rmse=rmse_staging.output,prd_rmse=rmse_production.output)
    redeploy_best_model = model_deploy(model_gcs_path=train_model.output,flag=best_model.output)
    

if __name__ == "__main__":
    xgboost_model_training()