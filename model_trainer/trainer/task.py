import os
import pandas as pd
import xgboost as xgb
import hypertune
import argparse
import logging
from sklearn.metrics import mean_squared_error

parser = argparse.ArgumentParser()
parser.add_argument("--model-dir",dest='model_dir',default=os.getenv('AIP_MODEL_DIR'),type=str, help="Model directory")
parser.add_argument("--max-depth", dest="max_depth",default=6, type=int, help="Maximum depth of the tree. Increasing this value can make the model more complex and prone to overfitting.")
parser.add_argument("--n-estimators", dest='n_estimators',default=100, type=int, help="Number of boosting rounds (trees). A higher value allows more complex models but increases training time.")
parser.add_argument("--subsample",dest='subsample',type=float, default=0.3, help="Fraction of the training data used for growing each tree. Lower values prevent overfitting.")
parser.add_argument("--learning-rate",dest='learning_rate',type=float, default=0.1,help="Step size shrinkage used to prevent overfitting. Lower values require more n_estimators.")
args = parser.parse_args()

logging.getLogger().setLevel(logging.INFO)

def fetch_data():
    logging.info("Downloading training and testing data from the bucket")
    
    train_df = pd.read_csv("https://storage.googleapis.com/tymestack-artifacts/dataset/boston_housing_train.csv")
    test_df = pd.read_csv("https://storage.googleapis.com/tymestack-artifacts/dataset/boston_housing_test.csv")

    # data preprocessing for training data
    y_train = train_df['medv'].values
    train_df.drop(columns=['medv'],inplace=True)
    X_train = train_df.values

    # data preprocessing for testing data
    y_test = test_df['medv'].values
    test_df.drop(columns=['medv'],inplace=True)
    X_test = test_df.values
    return {"X_train":X_train,"X_test": X_test,"y_train": y_train, "y_test":y_test}

def train_model(data):
    logging.info("Start training ...")
    # Init a Regressor Model
    model = xgb.XGBRegressor(max_depth=args.max_depth, learning_rate=args.learning_rate, n_estimators=args.n_estimators,subsample=args.subsample,objective='reg:linear')
    # Train XGBoost model
    model.fit(data['X_train'],data['y_train'])
    logging.info("Training completed")
    return model

def evaluate_model(model,data):
    
    y_pred = model.predict(data['X_test'])
    
    # evaluate predictions
    mse = mean_squared_error(data['y_test'], y_pred)
    logging.info(f"Evaluation completed with model error: {mse}")

    # report metric for hyperparameter tuning
    hpt = hypertune.HyperTune()
    hpt.report_hyperparameter_tuning_metric(
        hyperparameter_metric_tag='mean_squared_error',
        metric_value=mse
    )
    return mse


data = fetch_data()
model = train_model(data)
mse = evaluate_model(model, data)

# GCSFuse conversion
gs_prefix = 'gs://'
gcsfuse_prefix = '/gcs/'
if args.model_dir.startswith(gs_prefix):
    args.model_dir = args.model_dir.replace(gs_prefix, gcsfuse_prefix)
    dirpath = os.path.split(args.model_dir)[0]
    if not os.path.isdir(dirpath):
        os.makedirs(dirpath)

# Export the classifier to a file
gcs_model_path = os.path.join(args.model_dir, 'model.bst')
logging.info("Saving model artifacts to {}". format(gcs_model_path))
model.save_model(gcs_model_path)

logging.info("Saving metrics to {}/metrics.json". format(args.model_dir))
gcs_metrics_path = os.path.join(args.model_dir, 'metrics.json')
with open(gcs_metrics_path, "w") as f:
    f.write(f"{'mean_squared_error: {mse}'}")