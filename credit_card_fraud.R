library(readr)
library(ggplot2)
library(reshape2)
library(corrplot)
library(dplyr)
library(ROSE)
library(caret)
library(rpart)
library(skimr)
library(tidyr)
library(ROCR)
library(ModelMetrics)
library(PRROC)
library(MLmetrics)

credit_card <- read_csv("creditcard.csv")

# Rows: 284,807
#
# 284,807 Transactions, 492 are fraudulent. 
#
# Columns: 31
#
# Time = Number of seconds elapsed between this transaction and the first transaction in the dataset
# V1...V28 = Principal components obtained with PCA
# Amount = Transaction amount
# Class = 1 -> fraudulent, 0 -> non-fraudulent
#

credit_card %>%
  glimpse()

credit_card %>%
  skim()

#######################################

#Corrplot for V1-V28

Cor_credit_card <- cor(credit_card[,2:31])
corrplot(Cor_credit_card, method = "circle") 

#######################################

# Demonstration of potential relationship between time-of-day and fraud

day1 = subset(credit_card, Time < 86400, select = c(Time, Class))
day2 = subset(credit_card, Time >= 86400, select = c(Time, Class))
day2$Time = day2$Time - 86400

ggplot(data=day1, aes(x=Time)) +
  geom_smooth(aes(y=Class, color='Day 1'), se=F, color='cornflower blue') +
  geom_smooth(data = day2, aes(y=Class, color='Day 2'), se=F, color='salmon') +
  ggtitle('Distribution of Fraud vs. Time of Day') +
  xlab('Time of Day (Seconds)') +
  ylab('Distribution (Smoothed Class)')

#######################################

credit_card$Class <- factor(credit_card$Class, labels = c("non_fraudulent", "fraudulent")) 
table(credit_card$Class)

#######################################

credit_card %>%
  ggplot(aes(Class)) +
  geom_bar(fill='salmon', color='black') +
  #geom_text(aes(label =), position = position_stack(vjust = 0.5)) +
  ggtitle('Number of Values in each Class') +
  xlab('Class') +
  ylab('Count')


credit_card %>%
  ggplot(aes(Time, Amount)) +
  geom_line(color='salmon') +
  ggtitle('Transaction Amount over Time') +
  xlab('Time (Seconds)') +
  ylab('Amount (Dollars)')

#######################################

# check for missing values 

anyNA(credit_card)


#######################################

# sampling method - stratified

set.seed(998)
inTraining <- createDataPartition(credit_card$Class, p = .70, list = FALSE)
training <- credit_card[inTraining,]
testing  <- credit_card[-inTraining,]

table(training$Class)
table(testing$Class)

#######################################

# testing each method (up, down, rose, smote) with CV=10 and Boosted Logistic Regression

#######################################

# define a custome summary function to output specific metrics

customMetric = function(data, lev = NULL, model = NULL){
  f1 = f1Score(actual = data$obs, predicted = data$pred)
  rec = recall(actual = data$obs, predicted = data$pred)
  pre = precision(actual = data$obs, predicted = data$pred)
  auc = auc(actual = data$obs, predicted = data$pred)
  comb = rec*10 + pre
  c(customMetric = rec)
}

#######################################

# up sampling

train_control <- trainControl(method="CV", number=3, sampling="up", summaryFunction=customMetric) 

system.time({
  model <- train(Class~., data=training, trControl=train_control, method="LogitBoost", metric="customMetric")
})

pred.model <- predict(model, newdata=testing, type='prob')
pred.model_raw <- predict(model, newdata=testing)

print(model)
table(pred.model_raw)

caret::confusionMatrix(pred.model_raw, testing$Class, mode = "prec_recall")
prauc <- pr.curve(pred.model$fraudulent, weights.class0=as.numeric(testing$Class)-1, curve = T)
plot(prauc)

#######################################

# down sampling

train_control <- trainControl(method="CV", number=3, sampling="down", summaryFunction=customMetric) 

system.time({
  model <- train(Class~., data=training, trControl=train_control, method="LogitBoost", metric="customMetric")
})

pred.model <- predict(model, newdata=testing, type='prob')
pred.model_raw <- predict(model, newdata=testing)

print(model)
table(pred.model_raw)

caret::confusionMatrix(pred.model_raw, testing$Class, mode = "prec_recall")
prauc <- pr.curve(pred.model$fraudulent, weights.class0=as.numeric(testing$Class)-1, curve = T)
plot(prauc)

#######################################

# smote sampling - Synthetic Minority Over-sampling Technique

train_control <- trainControl(method="CV", number=3, sampling="smote", summaryFunction=customMetric) 

system.time({
  model <- train(Class~., data=training, trControl=train_control, method="LogitBoost", metric="customMetric")
})

pred.model <- predict(model, newdata=testing, type='prob')
pred.model_raw <- predict(model, newdata=testing)

print(model)
table(pred.model_raw)

caret::confusionMatrix(pred.model_raw, testing$Class, mode = "prec_recall")
prauc <- pr.curve(pred.model$fraudulent, weights.class0=as.numeric(testing$Class)-1, curve = T)
plot(prauc)

#######################################

# rose sampling - Randomly Over Sampling Examples

train_control <- trainControl(method="CV", number=3, sampling="rose", summaryFunction=customMetric) 

system.time({
  model <- train(Class~., data=training, trControl=train_control, method="LogitBoost", metric="customMetric")
})

pred.model <- predict(model, newdata=testing, type='prob')
pred.model_raw <- predict(model, newdata=testing)

print(model)
table(pred.model_raw)

caret::confusionMatrix(pred.model_raw, testing$Class, mode = "prec_recall")
prauc <- pr.curve(pred.model$fraudulent, weights.class0=as.numeric(testing$Class)-1, curve = T)
plot(prauc)

#######################################

# ramp it up 

train_control <- trainControl(method="CV", sampling="rose", summaryFunction=customMetric) 

system.time({
  model <- train(Class~., data=training, trControl=train_control, method="LogitBoost", metric="customMetric")
})


pred.model <- predict(model, newdata=testing, type='prob')
pred.model_raw <- predict(model, newdata=testing)

print(model)
table(pred.model_raw)

caret::confusionMatrix(pred.model_raw, testing$Class, mode = "prec_recall")
prauc <- pr.curve(pred.model$fraudulent, weights.class0=as.numeric(testing$Class)-1, curve = T)
plot(prauc)


ROSE::roc.curve(testing$Class, pred.model_raw)


#######################################
#######################################


# Fit Naive Bayes Model
# LogitBoost
model <- train(Class~., data=training, trControl=train_control, method="LogitBoost")
pred.model <- predict(model, newdata = testing)
View(pred.model)

#confusionMatrix(data = pred.model, reference = testing$Class)

accuracy.meas(testing$Class, pred.model[1])
roc.curve(testing$Class, pred.model[1], plotit = T)
