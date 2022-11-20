# Containerized MLflow Image Caption Model


## Overview
- Cross-selling in insurance is the practice of promoting products that are complementary to the policies that existing customers already own.
- The goal of cross-selling is to create a win-win situation where customers can obtain comprehensive protection at a lower bundled cost, while insurers can boost revenue through enhanced policy conversions.
- The aim of this project is to build a predictive ML pipeline (on the Health Insurance Cross-Sell dataset) to identify health insurance customers who are interested in purchasing additional vehicle insurance, in a bid to make cross-sell campaigns more efficient and targeted.


## Objective
- Make cross-selling more efficient and targeted by building a predictive ML pipeline to identify health insurance customers interested in purchasing additional vehicle insurance.

___
## Pipeline Components
- MLflow tracking
- Deployment of best model via FastAPI
- Streamlit user interface to post data to FastAPI endpoint

![Image](images/architecture.jpg)


```
```

___
## Project Files and Folders
- `/backend` - Contains files to setup backend service e.g. MLflow and FastAPI
    - `/data` - Data used for model training
    - `/mlruns` - ML runs from ML training experiments
    - `/mlartifacts` - ML artifacts from ML training experiments
    - `beheaded_inception3.py` - Beheaded Incpection3 model from torchvision
    - `Dockerfile` - Dockerfile to build backend service
    - `main.py` - Python script to serve ML artifacts via FastAPI. Command: `uvicorn main:app --host 0.0.0.0 --port 8000`
    - `MLproject`
    - `python_env.yaml`
    - `requirements-backend.txt` - python libraries to be installed during docker image build
    - `train.py` - Python script for the execution of H2O AutoML training with MLflow tracking. Run with this command: `python train.py --target 'Response'`
- `/frontend` - Folder containining the frontend user interface (UI) aspect of project (i.e. Streamlit)
    - `app.py` - Python script for the Streamlit web app, connected with FastAPI endpoint for model inference. Run in CLI with `streamlit run ui.py`
    - `Dockerfile` - Dockerfile to build frontend service     
- `/demo` - Folder containing gif and webm of Streamlit UI demo
- `/notebooks` - Folder containing Jupyter notebooks for EDA, XGBoost baseline, and H2O AutoML experiments
    - `01_EDA_and_Data_PreProcessing.ipynb` - Notebook detailing the data acquisition, data cleaning and feature engineering steps
    - `02_XGBoost_Baseline_Model.ipynb` - Notebook running the XGBoost baseline model for subsequent comparison
    - `03_H2O_AutoML_with_MLflow.ipynb` - Notebook showing the full H2O AutoML training and MLflow tracking process, along with model inference to get predictions  
- `/submissions` - Folder containing CSV files for Kaggle submission to retrieve model accuracy scores
