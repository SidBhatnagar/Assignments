 Problem:
Credit card fraud detection. It is important that credit card companies can recognize fraudulent credit card transactions so that customers are not charged for items that they did not purchase.
The dataset contains transactions made by credit cards in September 2013 by European cardholders and presents transactions that occurred over two days, where we have 492 frauds out of 284,807 transactions. The dataset is highly unbalanced, the positive class (frauds) account for 0.172% of all transactions.
Show Class Bar Plot
We have 31 columns. Due to confidentiality issues, we know nothing about the original features or any background information about the data. The feature ‘Time’ contains the seconds elapsed between a transaction and the start of the dataset. The feature 'Amount' is the transaction amount. Features V1 – V28 are the principle components obtained with Principle Component Analysis. Feature 'Class' is the response variable where the positive class is “fraudulent” and “non-fraudulent” otherwise.
Show Cor Plot
Inspiration:
The main problem in the data set was the massive class imbalance, so simple prediction accuracy score or area under the ROC curve would not give an accurate measure of the fit of our model. Having such a small number of fraud observations in the data, training any model would yield an accuracy score of about 99%. What we wanted to look at was the precision and recall of our model.
Show Precision/Recall Wiki Diagram + Equations
https://en.wikipedia.org/wiki/Precision_and_recall

Method:
There were four methods of sampling that we wished to explore to deal with the problem of class imbalance:
	Oversampling – adds more examples of the minority class (in this case, more fraud observations) by replicating existing observations. This could lead to overfitting the classifier on the few samples we have.
	Undersampling – removes some examples from the majority class (non-fraud) so that it has less effect on the model. The majority class has higher distribution and so is more representative of the data, so removing any observations could discard useful information.
	SMOTE or Synthetic Minority Over-sampling Technique - creates synthetic minority data points using vectors of k nearest neighbours in feature space added on to existing data points.
	ROSE or Randomly Oversampled Examples – creates a sample of synthetic data by adding more random observations of minority and majority class examples.
Splitting the data into training and testing subsets:
As the two classes were very imbalanced, we had to be careful when splitting the data into training and testing subsets so as not to divide the data such that we had little to no “fraudulent” samples in our test split. This would mean training the data on all the fraud observations and having very few to test on. We combatted this by using stratified train-test splitting which split our full data set such that there was a proportionate number of both classes in the training and testing subsets. 

Training the model with LGOCV: 
Also known as 'Leave-Group-Out Cross-Validation', or 'Monte Carlo Cross-Validation’. It works similarly to k-fold cross-validation in that it partitions the data k times to train and test the model, however it differs in that each partition is taken independently for each run, so that the same observation can appear in the test set multiple times. This tends to yield a better result for our model as we have very few observations for our minority class, so reusing some observations for a low number of folds benefits our model more than if we were to use normal cross-validation.

Training the model with Boosted Logistic Regression:
Logistic regression is a generalized linear model that tries to minimize the log loss of a model. What boosting basically does is that it creates an ensemble classifier, i.e. you take many such classifiers, each slightly different, and then make a prediction based on all their predictions. Since each of them is slightly biased towards being right, most of them will be biased towards the right. The weights on each of these classifiers is computed based on how well it classifies a weighted training set. Why? Because you want the ensemble of classifiers to do well on all kinds samples drawn from the data distribution and weighting the training set enables you to cover different parts of this distribution in some sense. In our data set we have a heavy bias in the distribution of data towards the majority class, the ‘non-fraud’ class, so we can improve upon a simple logistic regression model by approximating many additive generalized linear models. We do this using the ‘LogitBoost’ method. 

Outputting the results:



Moving Forward:
Class imbalance is an issue in many applied data science and machine learning problems. It is not just unique to credit card fraud detection but is also common in medical diagnosis, with classifying rare conditions, and predicting rare events such as volcanic eruptions or extreme weather. 
Some ways in which we could improve our model would be to obtain more data, experiment with further dimensional reduction, explore the possibility of concepts within classes, as well as testing different methods of cost sensitive learning.
