# Classifcation and Robustness of Deep Learning Networks with Medical Images
## by Jon Van Veen for the class ECE 697 at the University of Wisconsin-Madison

This repository contains code for the capstone project for my Masters degree in Machine Learning and Signal Processing at UW-Madison. 

My project investigated classification of two deep learning models, [Swin-T](https://github.com/microsoft/Swin-Transformer) and [ResNet50](https://pytorch.org/tutorials/beginner/transfer_learning_tutorial.html), on two medical imaging datasets, the [NIH Chest X-ray dataset](https://cloud.google.com/healthcare-api/docs/resources/public-datasets/nih-chest) and the [ISIC dataset](https://challenge.isic-archive.com/data/). This project also investigates a first step towards adversarial robustness by evaluating Swin-T's performance to noisy validation data. More information can be found in the project report file.

The above links are to the original model implementations for Swin-T and ResNet50. Scripts that I modified from the Swin-T repo are included under the Models directory. I ran Swin-T and ResNet50 on a compute cluster (for which shell scripts can be found under Models). I also ran Swin-T on Google Colab via the Swin.ipynb notebook. Data organization was a major component of this project, so under the Data directory I have included scripts I used for data organization. Project results can be found in the results folder, with discussion available in the project report.

Also included is a very small subset of the ISIC images with five classes (the full dataset has ten classes). The ResNet50 I used can be trained on this toy dataset in the Resnet50_Demo.ipynb notebook. To run the code:
1) Click [here](https://colab.research.google.com/github/jonvanveen/Medical-ViT-Classification-and-Robustness/blob/main/Resnet50_Demo.ipynb) to open the Colab notebook.
1) Download the notebook from this repo and upload it to Google Drive. Instructions on how to download an individual file from Github can be found [here.](https://www.wikihow.com/Download-a-File-from-GitHub)
2) Click on the isic_example_data directory, then copy the URL in your browser to [this tool](https://downgit.github.io/#/home) to download the dataset locally as a zip archive. 
3) Open the notebook with Colab and follow the remaining steps there.

