# Team members
* Linying ji, Biobehavioral Health Department, the Pennsylvania State Univeristy, University Park, U.S.
* Yuqi Shen, Biobehavioral Health Department, the Pennsylvania State Univeristy, University Park, U.S.
* Tanming Cui, Independent Researcher
* Young Won Cho, Health and Human Development Department, the Pennsylvania State Univeristy, University Park, U.S.
* Yanling Li, Health and Human Development Department, the Pennsylvania State Univeristy, University Park, U.S.
* Xiaoyue Xiong, Health and Human Development Department, the Pennsylvania State Univeristy, University Park, U.S.
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
  - torch 1.13.1+cu117 [[3]](#3)
  - torchaudio 0.13.1+cu117
  - torchvision 0.14.1+cu117
  - mvts_transformer [[4]](#4)
  - standarlization normalization
  - RAdam optimizer
  - Metric: MSE
  
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

<a id="3">[3]</a> 
Paszke, A., Gross, S., Massa, F., Lerer, A., Bradbury, J., Chanan, G., Killeen, T., Lin, Z., Gimelshein, N., Antiga, L., Desmaison, A., Köpf, A., Yang, E., DeVito, Z., Raison, M., Tejani, A., Chilamkurthy, S., Steiner, B., Fang, L., … Chintala, S. (2019). PyTorch: An Imperative Style, High-Performance Deep Learning Library. Advances in Neural Information Processing Systems, 32

<a id="4">[4]</a>
George Zerveas et al. A Transformer-based Framework for Multivariate Time Series Representation Learning, in Proceedings of the 27th ACM SIGKDD Conference on Knowledge Discovery and Data Mining (KDD '21), August 14-18, 2021. ArXiV version: https://arxiv.org/abs/2010.02803