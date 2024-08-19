# Interpreting Chest X-rays for disease classification

<div align="justify">
Chest X-ray is the most commonly performed medical imaging exam, with about 2 billion being conducted every year, and is used to diagnose and treat a variety of diseases. Chest X-ray interpretation is critical for detection of diseases such as pneumonia and lung cancer. The image below showcases different types of pneumonia as seen from X-ray against a health patient; for all 3 cases there are visible masses (lesions - damage of tissue) in the lung area.
</div>

<br>
<p align="center">
<img src="https://d2jx2rerrg6sh3.cloudfront.net/image-handler/picture/2020/12/Capture27.jpg" alt="Chest X-ray images showcasing various types of pneumonia" style="width:40%;">
<p align="center">Source: <a href="https://www.medrxiv.org/content/10.1101/2020.12.14.20248158v1.full.pdf">Transfer learning exploits chest-Xray to diagnose COVID-19 pneumonia (Katsamenis, I et al., 2020)</a></p>
</p>
<br>



## Challenges in medical datasets
<div align="justify">
Classification for medical imaging is faced with various challenges when it comes to training algorithms, namely class imbalance, multi-task and dataset size.
</div>

### Class Imbalance
<div align="justify">
Class imbalance refers to the disproportionate ratio of samples / images among the different classes, which ultimately leads to imbalanced models that classify the minority class(es) and results in poor classification performance. In medical datasets, there is not an equal number of examples of non-diseases and disease, especially for multi-class classification where we want to classify various types of diseases of similar nature. This is a direct reflection of the prevalence or disease in the real world, where there are a lot more examples of normal cases over affected ones. There are various ways to tackle the class imbalance problems, such as modifying the loss function and resampling, as discussed below.
</div>

#### Modifying the loss function
<div align="justify">
The Binary cross-entropy loss function, used for the case of classifying X-ray images, where X are the features / image we are feeding to the model, and y is the output probability of having a disease.
</div>

<br>
<p align="center">
<img src="./images/loss-function.png" alt="Binary Cross entropy loss function" style="width:45%;">
</p>
<br>

<div align="justify">
Because medical datasets are imbalanced, with the normal cases being more frequent, the total loss from normal examples will be higher than the ones from affected examples. So the algorithm will optimise its updates to get the normal examples right and not giving much relative weight to mass examples. The loss function can be modifies to weigh the normal and affected classes differently (w_p for the positive examples and w_n for the negative examples):
</div>

<br>
<p align="center">
<img src="./images/weighted-loss-function.png" alt="Binary Cross entropy weighted loss function" style="width:80%;">
</p>
<br>

#### Resampling method
<div align="justify">
We can group the normal classes and the affected classes together; the normal group will have more examples overall. From these groups we can now sample the images so that there is an equal number of positive and negative samples. The main issue here is that the resampled dataset may not have all samples from the normal cases, and will also have duplicates of the affected cases. If we now use the Binary cross-entropy loss function without the weights, there will be an equal contribution to the loss from the positive and negative examples.
</div>

### Multi-task
<div align="justify">
In binary classification, we only care if a case is positive or negative, however in many medical imaging applications we are interested in the presence or absence of many diseases. One simple way to tackle this is to have many models that each learn one of these tasks. Another way is to create a multi-task model that will allow the use of data more efficiently, as many of these diseases can have common features.
</div>
<br>
<div align="justify">
In multi-task classification, in comparison to binary, instead of having one label, we will now have a label for every disease, where 0 would denote the absence of the disease and 1 would denote its presence. The model will now also have multiple output probabilities instead of a single one. For multi-task classification we also need to modify the loss function to reflect the presence of multiple classes; the new loss will be the sum of the losses over the multiple diseases, in the case of chest x-rays it would be:
</div>
<br>

$$
L(X, y) = L(X, y_{mass}) + L(X, y_{pneumonia}) + L(X, y_{edema})
$$

<div align="justify">
And for class imbalance we can use the weighted loss function, where the weights now will be associated with each class. Below is shown the weighted loss function for edema:
</div>

<br>
<p align="center">
<img src="./images/weighted-loss-function-example.png" alt="Binary Cross entropy weighted loss function example" style="width:60%;">
</p>
<br>

### Dataset size
<div align="justify">
For many medical imaging problems, the architecture of choice is the convolutional neural network (CNN), which is designed to process 2D images. There are several different CNN architectures that have been proposed for image classification such as ResNet, DenseNet and Inception and all are composed of various building blocks. All of these architectures require large datasets, but medical imaging datasets typically only have 10,000 - 1,000,000 examples.
</div>

#### Transfer Learning
<div align="justify">
One solution to this problem is to pretrain the model on a different dataset, and then through transfer learning, copy over the learned features in a CNN for medical images. The network then can be fine-tuned to the medical imaging dataset to identify the presence and absence of diseases. 
</div>

<div align="justify">
By pretraining, the network learns general features, for example if we pretrain on a dataset containing images of animals we can identify the edges of the animals and this knowledge can be transferred in the new network to identify the edges of the lungs, providing a better starting point. We can choose to fine-tune both the early layers that provide the general features and the higher-level layers that are specific to the use-case, or only focus on the higher-level layers and leave the lower-level layers intact.
</div>

#### Data augmentation
<div align="justify">
We can generate more images from the ones we have in order to increase the size of our dataset. This can be done by applying transformations to the images, such as rotating, sideways translation, zoom in or out, change brightness or contrast, or even apply a combination of these transformations.
</div>
<br>

<div align="justify">
These transformations are helpful in augmenting the dataset, however these need to reflect variations that will help the model generalise the test set and real world scenarios, and also the transformations keep the label the same. For example, if we were to vertically flip a chest x-ray then the heart would show on the right side instead of left, and this a rare heart condition called dextrocardia, and thus the label would not be preserved.
</div>

## Model Testing

<div align="justify">
In all machine learning applications, we divide our dataset into smaller ones to be used for training, validation and testing. The training set is used for development, the validation set is used for tuning and selection of models (if no validation model exists then selection of models is done with the training set), and test set is for reporting results. We can also split the training and validation sets multiple times in a method called cross-validation to reduce variability in the estimate of model performance.
</div>
<br>

<div align="justify">
In the context of medicine, there are several challenges with building these sets, namely how to make the test sets independent (patient overlap), how we sample them and how we set the ground truth.
</div>

### Patient Overlap

<div align="justify">
An example of patient overlap is when a patient comes in twice within the span of a few months, and both times they were wearing a necklace when they had their x-ray taken. Then one of these images ends up in the training set and the other in the test set. We predict for the test image that it's normal, which can be true initially, however the problem is that the model can memorise the output normal for the necklace feature and become overconfident in its test set performance.
</div>
<br>
<div align="justify">
A way to solve this problem is to make sure that these images occur only in one of the sets, so the model won't memorise the necklace on the patient. This is done by splitting the datasets not by images, rather by patient so all the images of each patient will end up in the same set (be in training, validation or test set).
</div>

### Set Sampling

<div align="justify">
When the dataset is sampled and then split, we might end up with a test set that contains no positive or no negative cases, and we would have no way to actually test the performance of the model. This is especially a problem with medical datasets where we already don't have enough samples or no that may examples of a specific disease.
</div>
<br>
<div align="justify">
One way to tackle this is when creating a test set, is to sample the test set to have at least an X% of samples from the minority class(es), where typically X=50 (for binary classification) to ensure that there are sufficient examples to get a good estimate of the model performance for all cases. For the validation set we follow the same logic since we want it to reflect the distribution of classes as the test set.
</div>

### Ground Truth / Reference Standard

<div align="justify">
One major question when testing a model is how to determine the correct label (ground truth or reference standard) for a sample. In medicine, we might have experts that identify a positive case as a different disease, which is called interobserver disagreement.
</div>

<div align="justify">
In this case, the consensus vote method can be used, where a group of experts determine the ground truth which will be the majority vote or a single decision after consultation. Another way is to perform additional medical testing to provide additional information to determine the ground truth. 
</div>

## Key Evaluation Metrics

### Accuracy

<div align="justify">
For medical datasets, it's imperative to calculate the accuracy based on the correct predictions given that a patient has a disease or not. For this purpose, conditional probabilities are used to calculate the accuracy:
</div>
<br>

$$
Accuracy = P(correct \ \cap \ disease)+ P(correct \ \cap \ normal) = P(+ \ | \ disease)P(disease)+P(- \ | \ normal)P(normal)
$$

where $P(+ \ | \ disease)$ is called sensitivity and $P(- \ | \ normal)$ is called specificity.

### Positive and Negative Predictive Value

<div align="justify">
Positive predictive value (PPV) is the probability of a patient actually having a disease given that the model has predicted that they have it ( P(disease | +) ). Negative predictive value (NPV) is the probability of a patient being healthy given that the model has predicted that they are ( P(normal | -) ).
</div>

<div align="justify">
We can relate PPV and NPV to sensitivity and specificity using a confusion matrix. Below is presented a confusion matrix for heart disease.
</div>

<br>
<p align="center">
<img src="https://miro.medium.com/v2/resize:fit:874/1*h1MBLDA6bPxNpxwgSD1xNA.png" alt="Confusion matrix for heart disease" style="width:40%;">
<br>
<p align="center">Source: <a href="https://towardsdatascience.com/understanding-confusion-matrix-precision-recall-and-f1-score-8061c9270011">Understanding Confusion Matrix, Precision-Recall, and F1-Score</a></p>
</p>
<br>

Sensitivity, specificity, PPV and NPV will be calculated according to the following formulas:

$$
Sensitivity = \frac{TP}{TP+FN}
$$

$$
Specificity  = \frac{TN}{FP+TN}
$$

$$
PPV = \frac{TP}{TP+FP}
$$

$$
NPV = \frac{TN}{FN+TN}
$$

## ROC Curve and Threshold

<div align="justify">
The ROC curve aims to measure the quality of a binary estimator, as it considers how the estimator is able to split between both classes as we vary the threshold. Typically the threshold is set at 0.5.
</div>

To plot the ROC curve we need the True Positive Rate (TPR) and False Positive Rate (FPR).

$$
TPR = \frac{TP}{TP+FN} = sensitivity
$$

$$
FPR  = \frac{FP}{FP+TN} = 1 - specificity
$$



