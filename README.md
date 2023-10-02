# Enhancing Prognostics with Self-Supervised Imputation
Code for the paper in the 34th International Workshop on Principles of Diagnoses (DX'23) by Austin Coursey, Abel Diaz-Gonzalez, Marcos Quinones-Grueiro, and Gautam Biswas. Citation coming soon.

## Abstract

Determining how long a component or system has before it reaches its End of Life (EOL), i.e., its remaining useful life (RUL), is an important task in the field of prognostics. Reliably estimating the RUL can lead to the implementation of cost-saving condition-based maintenance approaches and overall safety in system operations. Recently, much work has been done on developing data-driven RUL models for aircraft engines, hard-disk drives, and batteries. However, the training sets used are mostly noise free, do not consider outliers, and ignore missing data. In the real world, sensors are noisy, subject to disturbances, and may not reliably report data at each time instance. To address these problems, and derive more accurate and robust RUL models, especially when we have missing data, we propose a Transformer model that performs imputation as a pre-training step to estimate the missing sensor data. When data is missing at test time, this model can estimate the missing values. Furthermore, we take a self-supervised approach to leverage the learned representations of the Transformer model in a supervised RUL prediction model to improve RUL prediction capabilities. We empirically demonstrate that we can perform reasonable predictions under missing data using the C-MAPSS dataset as a case study.

## How to Use This Repository

This repository contains all code used for the paper. To reproduce experiments, ensure that the [CMAPSS Dataset](https://www.kaggle.com/datasets/behrad3d/nasa-cmaps) is in the top directory in a folder called `CMAPSSData`. Then, train the imputation SSL model in `Imputation Transformer.ipynb` and save the state dict of that model. Finally, train the RUL Transformer model using the saved state dict in `Imputation RUL Transformer.ipynb`. Note that you will have to modify the line of code `z = z + self.positional_embed + imputation_latent` to the appropriate method of incorporating the latent embeddings to try different configutations. This is what worked best for us.

## File Details

- `DataUtils.py`: contains code for the CMAPSS Dataloaders
- `Masking.py`: contains code to mask the dataset in different ways for the SSL task of missing data imputation
- `Models.py`: PyTorch definitions of the Imputation Transformer models
- `Imputation Transformer.ipynb`: training the Imputation Transformer models
- `Imputation RUL Transformer.ipynb`: training the RUL estimation model utilizing the embeddings