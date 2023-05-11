# Mobile-Price-Classification
This repository contain the python code for Mobile price classification dataset
Project: Mobile Price Classification

TASK DESCRIPTION:
The task of our project is a predictive task where we are predicting the price_range attribute of the dataset using the other attributes given. The target attribute has fixed values i.e., 0,1,2,3; where 0 means low cost, 1 means medium cost, 2 means high cost and 3 means very high cost. As the target attribute has four class of price range in which the output lies, therefore, the task is a classification task and falls under supervised machine learning.

The dataset is taken from Kaggle named as mobile price classification. The data was designed to find the price range of the mobile using the given attributes therefore the prediction of price_range attribute value is chosen as the task for this project.

Before training the model we first copied the target attribute as ‘y’ and deleted it from the dataset. After this step data is distributed into training and testing that is discussed in the following section.

Data distribution:
The data is distributed into 75:25 ratio, where 75% contributes to the training set and 25% contributes to the testing set. The distribution is done using model_selection. train_test_split() function from sklearn library that takes the dataset and y as the parameters.

Dataset:
In this project we have used two datasets (Dataset1 and Dataset2) constructed on the basis of type of preprocessing. Dataset1 is constructed using the elimination of attributes on the basis of high correlation between the attributes and low or negative correlation of attributes with target attribute. The same preprocessing is done on training data and testing data.

Dataset2 is constructed by eliminating the attributes on the basis of a function that takes the original data as input and gives the feature names such that eliminating those features will give best accuracy on the data with the used parameters. The function finds the feature names by checking the accuracy without the features. If the accuracy is high without that attribute, then it will add to an array that stores the features to be removed. The same attributes are deleted from testing set too.

Models Used:
We have used two different models for the prediction of the target value. The two used models are Random Forest Classifier (parameters=’max_depth=19, random_state=0’) and Support Vector Classifier. Each model is used for both Dataset1 and Dataset2.

Accuracy, Precision, Recall, F1-score:
Random Forest Classifier with Dataset1:
Accuracy: 81.2
Precision: 0.81
Recall: 0.81
F1-score: 0.81

Random Forest Classifier with Dataset2:
Accuracy: 89.2
Precision: 0.89
Recall: 0.89
F1-score: 0.89
 

Support Vector Classifier with Dataset1:
Accuracy: 89.2
Precision: 0.89
Recall: 0.89
F1-score: 0.89

Support Vector Classifier with Dataset2:
Accuracy: 95.1
Precision: 0.95
Recall: 0.95
F1-score: 0.95

 

Comparison:

We have used the F1-score and root mean square error measures to compare the performance of two models.
F1-score: The closer is value of f1-score to 1, the better the performance of the model.
Root Mean Square Error: The closer the value of Root Mean Square Error to 0, the better the performance of the model.

For Dataset1 the Random Forest classifier gives the accuracy of 81.2 with the f1-score of 0.81 and Root Mean square error of 0.44, on the contrary, the Support Vector classifier gives the accuracy of 89.2 on the same dataset with the f1-score of 0.89 and Root Mean Square Error of 0.32. Therefore, Support Vector Classifier if performing good on Dataset1.

For Dataset2 the Random Forest classifier gives the accuracy of 89.2 with the f1-score of 0.89 and Root Mean square error of 0.32, on the contrary, the Support Vector classifier gives the accuracy of 95.1 on the same dataset with the f1-score of 0.95 and Root Mean Square Error of 0.21. Therefore, Support Vector Classifier if performing good on Dataset2.

Overall, the Support Vector Classifier if performing better than Random Forest classifier for Dataset1 as well as Dataset2. 

  

