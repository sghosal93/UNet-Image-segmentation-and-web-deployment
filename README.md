# UNet-Image-segmentation-and-web-deployment
Image segmentation using UNet (Tensorflow) and deployment using Flask 

Dataset used for this project and trained models are proprietary and is not readily available. However, the strucure of this code and model architecture (available in model.py) and associated files can be utlized to deploy any image segmentation model using a simple web-API developed using Flask (Python).

The API allows the user to upload a single image or a batch of images. Once uploaded, a trained segmentation model (built using Tensorflow) segments the image, saves th eoutput image in a local directory and shows the results on the webpage as well.