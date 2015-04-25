# Practical-Machine-Learning

---
title: 'Machine Learning : Writeup Project'
author: "Sevtap Ozisik"
date: "April 22, 2015"
output: pdf_document
---

Using devices such as Jawbone Up, Nike FuelBand, and Fitbit it is now possible to collect a large amount of data about personal activity relatively inexpensively. These type of devices are part of the quantified self movement – a group of enthusiasts who take measurements about themselves regularly to improve their health, to find patterns in their behavior, or because they are tech geeks. One thing that people regularly do is quantify how much of a particular activity they do, but they rarely quantify how well they do it. In this project, your goal will be to use data from accelerometers on the belt, forearm, arm, and dumbell of 6 participants. They were asked to perform barbell lifts correctly and incorrectly in 5 different ways. More information is available from the website here: http://groupware.les.inf.puc-rio.br/har (see the section on the Weight Lifting Exercise Dataset). 

The goal of your project is to predict the manner in which they did the exercise. This is the "classe" variable in the training set. We may use any of the other variables to predict with. We should create a report describing how you built your model, how you used cross validation, what you think the expected out of sample error is, and why we made the choices you did. we will also use the prediction model to predict 20 different test cases. 

## Data preparation

Load both datasets.
```{r}
raw_training <- read.csv('pml-training.csv')
raw_testing <- read.csv('pml-testing.csv')
```

Partition training data provided into two sets. One for training and one for cross validation.

```{r}
library(caret)
set.seed(1234)
trainingIndex <- createDataPartition(raw_training$classe, list=FALSE, p=.9)
training = raw_training[trainingIndex,]
testing = raw_training[-trainingIndex,]
```

Remove indicators with near zero variance.
```{r}
library(caret)
nzv <- nearZeroVar(training)

training <- training[-nzv]
testing <- testing[-nzv]
raw_testing <- raw_testing[-nzv]
```


Filter columns to only include numeric features and outcome. Integer and other non-numeric features can be trained to reliably predict values in the training file provided, but when used to predict values in the testing set provided, they lead to misclassifications.
```{r}
num_features_idx = which(lapply(training,class) %in% c('numeric')  )
```

We then would like to impute missing values as many exist in our training data.

```{r}
preModel <- preProcess(training[,num_features_idx], method=c('knnImpute'))

ptraining <- cbind(training$classe, predict(preModel, training[,num_features_idx]))
ptesting <- cbind(testing$classe, predict(preModel, testing[,num_features_idx]))
prtesting <- predict(preModel, raw_testing[,num_features_idx])

#Fix Label on classe
names(ptraining)[1] <- 'classe'
names(ptesting)[1] <- 'classe'
```

## Model

We can build a random forest model using the numerical variables provided. As we will see later this provides good enough accuracy to predict the twenty test cases. Using [caret][caret], we can obtain the optimal mtry parameter of 32. This is a computationally expensive process, so only the optimized parameter is shown below.

```{r}
library(randomForest)
rf_model  <- randomForest(classe ~ ., ptraining, ntree=500, mtry=32)
```

## Cross Validation

We are able to measure the accuracy using our training set and our cross validation set. With the training set we can detect if our model has bias due to ridgity of our mode. With the cross validation set, we are able to determine if we have variance due to overfitting.

### In-sample accuracy
```{r}
training_pred <- predict(rf_model, ptraining) 
print(confusionMatrix(training_pred, ptraining$classe))
```
The in sample accuracy is 100% which indicates, the model does not suffer from bias.

### Out-of-sample accuracy
```{r}
testing_pred <- predict(rf_model, ptesting) 
```

Confusion Matrix: 
```{r}
print(confusionMatrix(testing_pred, ptesting$classe))
```

The cross validation accuracy is greater than 99%, which should be sufficient for predicting the twenty test observations. Based on the lower bound of the confidence interval we would expect to achieve a 98.7% classification accuracy on new data provided. 

One caveat exists that the new data must be collected and preprocessed in a manner consistent with the training data.

## Test Set Prediction Results

Applying this model to the test data provided yields 100% classification accuracy on the twenty test observations.
```{r}
answers <- predict(rf_model, prtesting) 
answers
```

## Conclusion
We are able to provide very good prediction of weight lifting style as measured with accelerometers.


