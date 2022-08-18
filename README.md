# Classifcation and Robustness of Deep Learning Networks with Medical Images
## by Jon Van Veen for the class ECE 697 at the University of Wisconsin-Madison

This repository contains code for the capstone project for my Masters degree in Machine Learning and Signal Processing at UW-Madison. 

My project investigated classification of two deep learning models, [Swin-T][https://github.com/microsoft/Swin-Transformer] and [ResNet50][https://pytorch.org/tutorials/beginner/transfer_learning_tutorial.html], on two medical imaging datasets, the [NIH Chest X-ray dataset](https://cloud.google.com/healthcare-api/docs/resources/public-datasets/nih-chest) and the [ISIC dataset][https://challenge.isic-archive.com/data/]. This project also investigates a first step towards adversarial robustness by evaluating Swin-T's performance to noisy validation data. More information can be found in the project report file.

The above links are to the original model implementations for Swin-T and ResNet50. Scripts that I modified from the Swin-T repo are included under the Models directory. I ran Swin-T and ResNet50 on a compute cluster (for which shell scripts can be found under Models). I also ran Swin-T on Google Colab via the Swin.ipynb notebook. Data organization was a major component of this project, so under the Data directory I have included scripts I used for data organization.

Also included is a small subset of the ISIC images with three classes. The ResNet50 I used can be trained on this toy dataset in the Resnet50_Demo.ipynb notebook. Download the notebook and run all the cells on Colab to perform the classification. 

