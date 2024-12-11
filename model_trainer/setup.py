from setuptools import setup, find_packages

setup(
    name="trainer",
    version="0.1",
    packages=find_packages(),
    install_requires=[
        "xgboost",
        "scikit-learn",
        "pandas",
        "google-cloud-storage",
        "gcsfs",
        "cloudml-hypertune"
    ],
)