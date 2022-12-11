# Deep_Learning_subject_assignments

## Assignment 1A - Create a Perceptron from scratch
Straighforward implemenation of a perception algorithm


## Assignment 1B - Identifying characters in KMNIST dataset using neural networks
Full report available here  

https://github.com/Declan-Stockdale/Deep_Learning_subject_assignments/blob/master/Assignment_1/Part_B_report.docx

### Overview
The MNIST (Modified National Institute of Standards and Technology database) is a commonly used database for benchmarking testing performance of models. An alternative dataset that is gaining attention is the Kuzushiji-MNIST (KMNIST) dataset. The original MNIST dataset consists of thousands of handwritten numbers 0 through to 9. The KMNIST dataset alternatively consists of thousands of examples of one of 10 handwritten Kuzushiji (outdated Japanese alphabet) characters with the aim to extract written history from the potentially hundreds of millions of books written using the outdated alphabet which haven’t yet been translated [1]. 




### Data
The data has been downloaded from the Centre for Open Data in the Humanities (CODH) website [2]. All analysis has been performed using the python language and implementing using various packages most notably the Keras package which is user-friendly version of the more well-known TensorFlow package written by Google. 
The combined dataset is comprised of 70,000 characters and was supplied as a train set of 60,000 images and a test set of the remaining 10,000 images. All images were 28x28 pixels and appeared to be centered. The images were converted into grayscale for easier processing. If this isn’t performed the model may add importance to the colour of the line or background rather than the shape of the lines.

![image](https://user-images.githubusercontent.com/53500810/206883828-86ca38a6-72e7-422c-9156-fb785cebe5a3.png)

![image](https://user-images.githubusercontent.com/53500810/206883852-024866a5-03d5-4157-af41-f6b68a446d07.png)

### Modelling

#### Setup
For each experiment the batch size was predefined as 128 and the number of epochs was set to 500. The output layer was a softmax activation function with an output of 10 classes, one for each unique Kuzushiji character in the dataset. The loss function used was the categorical cross entropy as this is a multiclass problem.  A validation split of 20% was of the training data was used to obtain the validation metrics.

#### Callbacks
reduceLR – Reduce learning rate
The learning rate is generally set at some value (0.001) but when approaching an optimum value may not be sufficiently small enough which results in a plateau of the accuracy of the model. This callback allows for the dynamic reduction of the learning rate once a plateau is reached in the aim of improving the accuracy of the model.

Early stopping
This is very useful as the computational power and time needed for training can be large. Early stopping stops the model once a given metric stops improving after a given number of epochs. The metric in this case is the validation accuracy which will be maximized. If the model plateaus and we continued the training process past this point, it would continue to learn the training data and would struggle to generalize to the testing set. It would also be a waste of power and potentially money if using a paid service.

Modelcheckpoint
This callback saves the best model for a given metric which is validation accuracy in our case. If during training a higher validation accuracy is achieved the previous best model is overwritten with the new best model. This may also be useful is there is a limit on how long training can occur or the potential of hardware, power failure or time limitations during training. 


#### Simple naive model

A very simple model was first implemented using only 2 hidden dense layers each of 512 neurons using the relu activation function for both hidden layers. The network architecture has been displayed in Figure 2 using the Keras plot plot_model function. 

![image](https://user-images.githubusercontent.com/53500810/206883907-1e74e7c3-4616-400c-8a07-9a55c8d8ae48.png)

#### Effect of increasing hidden layers

![image](https://user-images.githubusercontent.com/53500810/206883926-c4d5df03-30cb-415c-b321-047f4f04e69b.png)

#### Dropout of 20%
Dropout is a one of many regularization methods that we can implement to reduce overfitting. It works by ignore some percentage of neurons in the hope that other neurons contribute to the prediction. An oversimplified explanation would be that it limits the reliance on a single neuron for the prediction and asks more neurons for their opinion before making a prediction. This improves the ability of the network to generalize and reduces overfitting on the training data.
The accuracy of this model was 89.15% which is slightly lower than the model without dropout. This may indicate that the model is efficient, and neurons randomly removed during dropout are necessary. 

#### Convolutional Neural Network (CNN)
The model chosen was an implementation of a successful MNIST CNN model.

The accuracy of the CNN model was 96.11% which is a significant jump compared to the non-CNN models implemented earlier. The confusion matrix in figure 6 also shows a significant drop in incorrectly identified classes

![image](https://user-images.githubusercontent.com/53500810/206883996-b8ef2688-f834-4c41-95ec-82c0e47b9627.png)


#### CNN and synthetic data

In cases where the number of examples in the training set are limited, we have a few options. The absolute best option would be to collect more training examples but in this case that’s not an option. Instead, we can create more training examples by randomly modifying existing training examples. 
The Keras package ImageDataGenerator allows for multiple options modify images. The options chosen for this dataset are a random rotation of 15 degrees, a slight zoom, shifting the width and height of the image and distorting the image. All of these characteristics may be observed in the testing set due to the nature of handwritten characters with different slants, size of characters or may occur due to off centering of training examples in the data collection and cleaning process.



### Findings
Table 3 below shows the accuracy and the complexity of the trainable parameters of each model. It can clearly be seen the increasing the number of layers in the naïve model resulted in a reduction of performance possibly due to added complexity. If early stopping was removed, we may see improvements, but the training time would be prohibitive. 
It can also be seen the number of trainable parameters in the CNN model is comparable to the naïve model with 2 hidden layers. This is due to the usage of kernels and max pooling which substantially reduces the input between layers. 
![image](https://user-images.githubusercontent.com/53500810/206884050-a5a53a52-aa2f-4bc4-92a7-902a451e6253.png)

![image](https://user-images.githubusercontent.com/53500810/206884045-a7d402bc-54f0-42a2-a80d-f5d6dc28488b.png)


![image](https://user-images.githubusercontent.com/53500810/206884027-d2262acb-e2de-4c91-8048-73516932e5d3.png)

![image](https://user-images.githubusercontent.com/53500810/206883985-f96c8d47-7919-4912-905d-32a5fb9d90f2.png)
