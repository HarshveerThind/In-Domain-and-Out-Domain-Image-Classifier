In Domain and Out Domain Image Classifier
Author - Harshveer Thind

This project builds an image classification model using transfer learning to compare performance across in domain and out domain datasets. 
It focuses on how well a model trained on one distribution generalises when tested on a different distribution.

Dataset structure

The dataset is not included due to size limits. Add your data using the structure below:

data/
  in_domain_train/      # labelled
  in_domain_test/       # labelled
  out_domain_train/     # unlabelled
  out_domain_test/      # labelled

Features:
- Transfer learning with a deep CNN backbone
- Training pipeline with transforms and data loaders
- Evaluation on in domain and out domain test sets
- Supports custom image folders following the expected layout

How it works:
The model is trained using labelled in domain images. After training, performance is evaluated on both the in domain test set and the labelled out 
domain test set to measure robustness under domain shift. Unlabelled out domain training images can still be used for additional domain exposure.

Requirements:
python3
torch
torchvision

Install dependencies with:

pip install torch torchvision

Train and evaluate the model by running:

python main.py
