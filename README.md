# Self-Supervised Transformer Imputation for Reliable Prognostics Under Missing Data
Project for Vanderbilt University's Special Topics in Deep Learning course, Spring 2023

Austin Coursey and Abel Diaz-Gonzalez

## Abstract

Being able to determine how long a component or system has before it fails, the remaining useful life (RUL), is an important task in the field of prognostics. Reliably estimating the remaining useful life can lead to enhanced maintenence or safety of devices. Recently, much work has gone into developing data-driven models of the remaining useful life in domains such as aircraft engines, hard-disk drives, or batteries. However, most of this work trains on cleaned datasets free of outliers and missing data. In the real world, there is the possibility that sensors may not reliably report data at each time instance, especially if a fault occurs. To robustly estimate the RUL under potential missing data, we propose to use a Transformer model to perform imputation as a pretraining step that can learn to estimate missing sensor data. When data is missing at test time, this model can estimate the missing values. Furthermore, we take a self-supervised approach to leverage the learned representations of the Transformer model in a supervised RUL prediction model to decrease training time and improve RUL prediction capabilities.

## TODO

- [x] Decide on datasets
- [x] Preprocess data
- [ ] Design data augmentation algorithm
- [ ] Design Transformer architecture and decide which stage to extract
- [ ] Structure files and experiments
- [ ] Run imputation pretraining
- [ ] Validate imputation pretraining
- [ ] Run prognostics training
- [ ] Assess prognostics performance
- [x] Write literature review
- [ ] Write report
