# MLOps Course Project

This repository contains materials, code, and documentation for my MLOps course work. The project focuses on implementing machine learning operations best practices including model training, deployment, monitoring, and maintenance.

## Project Structure
mlops_project/
├── data/                # Dataset storage

├── notebooks/          # Jupyter notebooks for exploration and analysis

├── models/             # Trained ML models

├── requirements.txt    # Package dependencies

└── README.md           # Project documentation


## Environment Setup

This project uses a Python virtual environment with the following core dependencies:
- mlflow==2.15.1
- numpy==1.26.4
- pandas==2.2.2
- scikit-learn==1.5.1

To set up the environment:
1. Create a virtual environment: `python3 -m venv ~/tmp/env/mlops`
2. Activate the environment: `source ~/tmp/env/mlops/bin/activate`
3. Install dependencies: `pip install -r requirements.txt`

## Project Goals

- Implement ML pipeline automation
- Apply version control for data and models
- Create reproducible ML experiments
- Deploy ML models efficiently
- Monitor model performance in production

