# CNN Indoor/Outdoor Sport Classification

## Project Overview:
The goal is to classify sports images as either indoor (0) or outdoor (1) using deep learning techniques. A Convolutional Neural Network (CNN) is trained on a dataset of sports images categorized by environment type to evaluate its classification performance.

## Data Overview:
The dataset consists of images of athletes participating in various sports, categorized into indoor and outdoor environments. The images are organized into train, test, and validation folders. An example notebook with data loaders is provided in notebooks/data_loader.ipynb
![image](https://github.com/user-attachments/assets/2073b744-dbce-4729-8d94-5bd25c321a82)

## Methods Overview:

The project uses a Convolutional Neural Network (CNN) to classify images of athletes participating in various sports into two categories: indoor and outdoor. The following steps outline the method used:

1. **Data Preprocessing**:
   - The dataset consists of images of athletes, organized into training, validation, and test sets. Each set is further divided into two categories: indoor and outdoor.
   - The images are resized and normalized to ensure consistency in input data and to facilitate efficient training of the model.
   
2. **Model Architecture**:
   - The CNN model consists of three convolutional layers followed by ReLU activations, pooling layers, and two fully connected layers.
   - The model uses the **ReLU** activation function to introduce non-linearity and the **MaxPool2d** layer to reduce the spatial dimensions of the image after each convolution.
   - The output of the final fully connected layer is passed through a softmax function for classification into two classes: **Indoor (0)** and **Outdoor (1)**.

3. **Training**:
   - The model is trained using **CrossEntropyLoss** as the loss function and **Adam** optimizer for gradient descent optimization.
   - During each epoch, the model is evaluated on both the training and validation sets to monitor progress. The accuracy and loss are reported after each epoch.

4. **Model Evaluation**:
   - After training, the model is evaluated on a test set that was not seen during training. The final classification accuracy and loss on the test set are reported.

## Results:
In progress...

## Conclusion:
In progress...
