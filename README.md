# Team members
* Linying ji, Biobehavioral Health Department, the Pennsylvania State Univeristy, University Park, U.S.
* Yuqi Shen, Biobehavioral Health Department, the Pennsylvania State Univeristy, University Park, U.S.
* Tanming Cui, Independent Researcher
* Young Won Cho, Health and Human Development Department, the Pennsylvania State Univeristy, University Park, U.S.
* Yanling Li, Health and Human Development Department, the Pennsylvania State Univeristy, University Park, U.S.
* Xiaoyue Xiong, Health and Human Development Department, the Pennsylvania State Univeristy, University Park, U.S.
* Christopher Michael Crawford, Health and Human Development Department, the Pennsylvania State Univeristy, University Park, U.S.
* Zachary Fisher, Health and Human Development Department, the Pennsylvania State Univeristy, University Park, U.S.
* Sy-Miin Chow, Health and Human Development Department, the Pennsylvania State Univeristy, University Park, U.S.
# Our approach
## Machine Learning Models
* XGBoost
  - version number, [citation]
  - One-hot-encoding?
  - hyper-parameter tuning:
  - Metric: RMSE

* Transformer
## Data Preprocessing
* Processing physiodata
* Reducing frequency
* data merge
* dynamic feature [maybe]

# Repository content
* "data_processing" folder:
  - *xxx.py*: Python code for processing physio data and merging physio and affect data
  - *extract_dynamic_features.R*: R code for extracting dynamic features based on processed physio data
* "results" folder: result files with predictions. Use the original naming and structure of directories, e.g., ./results/scenario_2/fold_3/test/annotations/sub_0_vid_2.csv
* "models" folder:
  - *xxx.py*: code for fitting XGBoost models
  - *xxx.py*: code for fitting transformer models
* "requirements.txt": dependencies file with required libs and their versions


# Notes for running the code
