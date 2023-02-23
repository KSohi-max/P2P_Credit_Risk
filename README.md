# P2P Lending_Credit Risk

### Purpose of Analysis

Credit risk poses a classification problem that’s inherently imbalanced. The reason is that healthy loans easily outnumber risky loans. This analysis uses various techniques to train and evaluate models with imbalanced classes. The dataset used consists of historical lending activity from a peer-to-peer lending services company.  These data will be used to build a model that can identify the creditworthiness of borrowers.

### Data Used

The *lending_data.csv* provides the following data:

* loan_size
* interest_rate
* borrower_income
* debt_to_income
* num_of_accounts
* derogatory_marks
* total_debt
* loan_status

'loan_status' is the target variable the model is attempting to predict accurately, i.e., the 'y'.

It is important to note that a value of '0' in the “loan_status” column means that the loan is 'healthy'. A value of '1' means that the loan has a 'high-risk' of defaulting.

All other features (or independent variables) are data that provide information on the loans that contribute to the prediction of 'y', i.e, 'X'.

### Prediction

For the orginal dataset, the 'y' variable value_counts show the following distribution:

![Original_valcount](https://github.com/KSohi-max/P2P_Credit_Risk/blob/main/Images/original_valcount.png)

As you can see, the number of cases where loan_status value is '1' or 'high-risk' of defaulting is only 2,500 while number of cases where loan_status value is '0' or 'healthy' is 75,036.  Based solely on this information, it is evident that there is a significant imbalance in the dataset being used to identify 'high-risk loans'.

After oversampling was applied to the dataset using RandomOverSampling(), the following distribution was observed:

![ROS_valcount](https://github.com/KSohi-max/P2P_Credit_Risk/blob/main/Images/ROS_valcount.png)

For each loan_status of '0' or 'healthy' loans and '1' or 'high-risk of default', 56,271 data rows were created to equalize the imbalance in the dataset.  The goal is to allow the model to 'learn' from each of the two cases/classes so that it can statistically predict a 'high-risk default' occurence based on the features input data (X).

### Stages of ML Process

#### Original Dataset

Original dataset was split into 'y' - target/dependent variable data and 'X' - feature/independent variables data as this allows the model to identify which variable it is attempting to predict.  Each of the X and y data were then split into training datasets and testing datasets using *train_test_split* function to allow for model validation process.  

The [`LogisticRegression`](https://scikit-learn.org/stable/modules/generated/sklearn.linear_model.LogisticRegression.html#sklearn.linear_model.LogisticRegression) regression model was instantiated.  The model was then trained or fitted on the training data (X_train, y_train).  

Using the X_test data that was split, the predicted 'y' or 'y_pred' was generated.  To validate the 'y_pred', the actual value or 'y_test' was used to determine how many of the predicted values matched the actual values. 

#### Random Over Sampling

The original dataset was used to resample the training and testing data using [`RandomOverSampler`](https://imbalanced-learn.org/stable/references/generated/imblearn.over_sampling.RandomOverSampler.html). 

The `RandomOverSampler` was instantiated and a new data split called X_oversampled, y_oversampled was generated to attempt to remove the imbalance in the training dataset.

The new 'oversample' data was used to re-train the `LogisticRegression` model once again. And similar to the orginal process, the 'X_test' data was used to generate 'y_pred_oversampled' predictions using the updated re-trained model. To validate the 'y_pred_oversampled', the actual value or 'y_test' was used to determine how many of the predicted values matched the actual values.

### Methods Used

[`LogisticRegression`](https://scikit-learn.org/stable/modules/generated/sklearn.linear_model.LogisticRegression.html#sklearn.linear_model.LogisticRegression) regression model assumes that the response variables (target variable) can only take on two possible outcomes – pass/fail, male/female, and malignant/benign. This assumption can be checked by simply counting the unique outcomes of the dependent variable, in our case, '0' or 'healthy' loans and '1' or 'high-risk of default'. 

(Source: https://www.spiceworks.com/tech/artificial-intelligence/articles/what-is-logistic-regression/#:~:text=The%20first%20assumption%20of%20logistic,outcomes%20of%20the%20dependent%20variable.)

[`RandomOverSampler`](https://imbalanced-learn.org/stable/references/generated/imblearn.over_sampling.RandomOverSampler.html) is from the imbalanced-learn library.  It's 'object is to over-sample the minority class(es) by picking samples at random with replacement.' So in our case, it was able to create equal number of datapoint for the two classes of '0' or 'healthy' loans and '1' or 'high-risk of default'. 

## Results

### Machine Learning Model_Original Data:

The following image illustrates Accuracy, Prediction and Recall scores:

!['Machine Learing Model_Original Dataset'](https://github.com/KSohi-max/P2P_Credit_Risk/blob/main/Images/results_orginal%20dataset.png)

Accuracy is ~95% based on the validation test using original dataset.

Precision or the proportion of positive calls that were correct (Precision = TP/(TP + FP)) is: For '0' or 'healthy loans' is 1.00 and for '1' or 'high-risk loans' 0.85. 

Recall or the proportion of truly positive samples that were correct (Recall = TP/(TP + FN)) is: For '0' or 'healthy loans' is 0.99 and for '1' or 'high-risk loans' 0.91. 

### Machine Learning Model_RandomOverSampled:

The following shows the Accuracy, Prediction and Recall scores:

!['Machine Learing Model_RandomOverSampled'](https://github.com/KSohi-max/P2P_Credit_Risk/blob/main/Images/results_ROS%20dataset.png)

Accuracy is ~99% based on the validation test.

Precision is: For '0' or 'healthy loans' is 1.00 and for '1' or 'high-risk loans' 0.85. 

Recall is: For '0' or 'healthy loans' is 0.99 and for '1' or 'high-risk loans' 0.91

## Summary

Based solely on Accuracy Score the model with oversampled dataset 'seems' to perform the best at 99%. Precision and Recall are essentially the same for models trained on orginal and oversampled datasets.

From a business perspective, it is far more important to predict the 'high-risk loans' class than the 'healthy loans' class. The `RandomOverSampler` functions is sampling from too small a dataset, namely 2,500 rows to 56,271 and so there is likely a lot of repetition of same datapoints in the training dataset from the original 2,500 data points.  This leads to the model learning from same data points without any variation.  This is the reason for very high Accuracy Score for both models (Original - 95% v. Oversampled - 99%).

Given that there aren't enough varied cases of 'high-risk loans' in the dataset to train a model sufficiently to predict the probability of occurence, I wouldn't recommend using either of the models.

My recommendation would be to explore the opportunity to:

* collect additional data for 'high-risk loans' for the model from the company
* generate additional cases using Sythetic Sampling (KNN) to simulate characteristics of borrower's who default and add to original dataset
