## Information

1. This project will be running in CPU since we have small dataset.

## Setting Up environment

1.Run below command to create python environment
 ```bash
conda create -p venv python==3.11 -y  
```

2. Create requirements.txt file with required libraries.

3. Activate environment by running below command.
 ```bash
conda activate venv/ 
```
4. Install all libs in requirements.txt using below command.
 ```bash
pip install -r requirements.txt
```

5. Install ipykernel library by running below command.
```bash
pip install ipykernel
```

6. ANN Classification:
  
 1. Execute 'experiments.ipynb' to save encoders, scalers and training the model.
 2. Execute 'prediction.ipynb' for prediction.
 3. Run below command to execute the model in streamlit app 
 ```bash
 streamlit run app.py
 ```
 4. Also can test the deployed app at https://ann-classification-churn-ukasjudpgdbtwy8gc5yd3h.streamlit.app/

7. ANN Regression:
 1. Execute 'salaryregression.ipynb' to save encoders, scalers and training the model.
 2. Run below command to execute the model in streamlit app 
 ```bash
 streamlit run streamlit_regression.py
 ```
