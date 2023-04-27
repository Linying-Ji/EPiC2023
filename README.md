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
  - xgboost 1.7.5 [[1]](#1)
  - Dependencies: see requirements.txt
  - hyper-parameter tuning using Hyperopt 0.2.7 [[2]](#2)
  - Metric: RMSE

* Transformer
  -
  -
  -
  
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


## References
<a id="1">[1]</a> 
Tianqi Chen and Carlos Guestrin. "XGBoost: A Scalable Tree Boosting System." In Proceedings of the 22nd ACM SIGKDD International Conference on Knowledge Discovery and Data Mining (KDD '16), pp. 785-794, San Francisco, CA, USA, August 13-17, 2016.

<a id="2">[2]</a> 
Bergstra, J., Yamins, D., Cox, D. D. (2013) Making a Science of Model Search: Hyperparameter Optimization in Hundreds of Dimensions for Vision Architectures. To appear in Proc. of the 30th International Conference on Machine Learning (ICML 2013).
