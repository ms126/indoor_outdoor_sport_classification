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
- Epoch [1/10], Train Loss: 0.4692, Train Acc: 78.64%, Valid Loss: 0.4257, Valid Acc: 79.20%
- Epoch [2/10], Train Loss: 0.4256, Train Acc: 81.00%, Valid Loss: 0.3984, Valid Acc: 79.80%
- Epoch [3/10], Train Loss: 0.3524, Train Acc: 84.92%, Valid Loss: 0.3558, Valid Acc: 84.80%
- Epoch [4/10], Train Loss: 0.2616, Train Acc: 89.38%, Valid Loss: 0.5081, Valid Acc: 84.20%
- Epoch [5/10], Train Loss: 0.1557, Train Acc: 94.20%, Valid Loss: 0.5929, Valid Acc: 84.80%
- Epoch [6/10], Train Loss: 0.0853, Train Acc: 96.90%, Valid Loss: 0.6017, Valid Acc: 83.80%
- Epoch [7/10], Train Loss: 0.0530, Train Acc: 98.30%, Valid Loss: 0.8172, Valid Acc: 85.40%
- Epoch [8/10], Train Loss: 0.0443, Train Acc: 98.69%, Valid Loss: 1.1107, Valid Acc: 84.80%
- Epoch [9/10], Train Loss: 0.0349, Train Acc: 98.97%, Valid Loss: 1.3094, Valid Acc: 84.00%
- Epoch [10/10], Train Loss: 0.0354, Train Acc: 99.21%, Valid Loss: 1.1485, Valid Acc: 83.00%

After 10 epochs, the model achieved a **train accuracy** of 99.21% and a **validation accuracy** of 83.00%. Despite the high training accuracy, validation performance fluctuated, indicating potential overfitting towards the training data.

![image](https://github.com/user-attachments/assets/05897d26-c652-4ba8-a0f0-fb03e335c8d7)

## Conclusion:
### **Conclusion**:

The CNN model showed significant improvement in accuracy on the training set, reaching a very high **train accuracy** of 99.21% by the end of the 10th epoch. However, the model experienced a decline in **validation accuracy** after the initial epochs, indicating that while it learned well on the training data, it struggled to generalize effectively to the validation set.

This suggests that the model may be overfitting, as evidenced by the decreasing validation accuracy and increasing validation loss in the later epochs. Overfitting typically occurs when the model learns to memorize the training data rather than generalize to unseen data. The validation accuracy fluctuating while the training accuracy steadily increases is a common symptom of this issue.

Several approaches can be taken to improve the model's performance:
- **Regularization**: Techniques such as **dropout** or **L2 regularization** could be applied to prevent overfitting and help the model generalize better to new data.
- **Data Augmentation**: Increasing the size and diversity of the training data by applying transformations like rotation, scaling, and flipping to the images could help the model learn more robust features.
- **More Epochs with Early Stopping**: Training for more epochs with an early stopping criterion based on the validation loss can help prevent the model from overfitting while still benefiting from longer training periods.
- **Model Architecture**: Experimenting with deeper architectures, adding more convolutional layers, or using pre-trained models like **ResNet** or **VGG** might help the model perform better on both training and validation datasets.
- **Hyperparameter Tuning**: Adjusting hyperparameters such as learning rate, batch size, and the optimizer could also improve performance.

The relatively stable validation accuracy during the first few epochs suggests that the model is learning useful features from the images. The validation loss and accuracy fluctuations in the later epochs highlight that additional strategies are necessary to improve the generalization ability of the model.

In summary, while the model demonstrates strong performance on the training data, further tuning and improvements are needed to enhance its generalization to the validation set. Future work will focus on implementing these strategies to optimize the model and achieve better overall performance in classifying indoor and outdoor sports images.
