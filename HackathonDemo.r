#------------------------------------------------------
# Athenahacks: Intro to ML, AI, and Predictive Analysis (Neudesic)
#
# Author: Yueying Li and Karla Benefiel     
#         February 24, 2018
#
# This code predicts if a customer is likely to make a purchase and if they do, how much will they buy.
#
# The first section predicts who is like likey to make a purchase
# The second section predicts how much they are likely to buy
#
#------------------------------------------------------


#------------------------------------------------------
# Load Packages
#------------------------------------------------------
install.packages("dplyr")
library(dplyr)
install.packages("ggplot2")
library(ggplot2)
install.packages("VIM")
library(VIM)
#------------------------------------------------------


#------------------------------------------------------
# Loading the data
#------------------------------------------------------
# Set working directory
setwd("C:\\Users\\karla.benefiel\\OneDrive\\AthenaHacks")
# Read in data
data <- read.delim("untitled text.txt", stringsAsFactors=FALSE)
# Inspect the dataset
str(data)
head(data)
summary(data)
#------------------------------------------------------



# -------------------------------------------------------------------------------------------------------------
# -------------------------------------------------------------------------------------------------------------
# SECTION 1: Predict who is likely to make a purchase
# -------------------------------------------------------------------------------------------------------------
# -------------------------------------------------------------------------------------------------------------



#------------------------------------------------------
# Pre-processing
#------------------------------------------------------
# sequence number and spending are not informative for classification
# exclude sequence number and spending
purchase <- data %>% select(-sequence_number,-Spending)
# change eliminate numeric data
purchase_f <- purchase %>% select(-last_update_days_ago, -X1st_update_days_ago ,-Freq) %>% 
                          # then make all other columns of type factor
                           sapply(.,as.factor) %>% 
                          # make it a data frame
                           as.data.frame()
# check the results of the changes
str(purchase_f)
summary(purchase_f)
#------------------------------------------------------


#------------------------------------------------------
# Visualize pre-processed data
#------------------------------------------------------
# visualize missing values
# but from the visual, we can see there is nothing missing
aggr(purchase_f, prop=FALSE, numbers=TRUE,bars=FALSE,combined=TRUE,
     ylabs="Vendor",border=NA, labels=TRUE, col=c("mediumaquamarine","red"))
# bar chart to look distribution of purchasers
ggplot(purchase_f, aes(purchase_f$Purchase)) + geom_bar(fill = "mediumaquamarine")
# of each purchasers and non purchasers, how many were male?
ggplot(purchase_f, aes(purchase_f$Purchase)) + geom_bar(aes(fill = purchase_f$Gender.male))
# of people who purchased, how many purchased through the web
ggplot(purchase_f[purchase_f$Purchase==1,], aes(purchase_f[purchase_f$Purchase==1,"Web.order"])) + geom_bar(fill = "mediumaquamarine")
# how many days ago was lst update to customer record for people who made a purchase
ggplot(purchase, aes(purchase[,"X1st_update_days_ago"])) + geom_bar() +facet_grid(purchase$Purchase ~ .)
#how many transactions did they make in the last year at source catalog
ggplot(purchase, aes(purchase[,"Freq"])) + geom_bar() +facet_grid(purchase$Purchase ~ .)
#------------------------------------------------------


#------------------------------------------------------
# Check Outliers
#------------------------------------------------------
# extreme outliers and stats
Outliers<-function(x) {
  
  quantil=lapply(x, quantile, name=FALSE)
  
  stats=data.frame(name=cbind(names(x)),lower=NA,upper=NA, outlier.cnt=NA, 
                   minimum=NA, maximum=NA,variance=NA)
  
  for (i in 1:length(x)) {
    quantil3=quantil[[i]][4]
    quantil1=quantil[[i]][2]
    IQR=quantil3-quantil1
    upper=quantil3+3*IQR
    lower=quantil1-3*IQR
    stats[i,2]=lower
    stats[i,3]=upper
    stats[i,4]=subset(x,x[i]>upper | x[i]<lower) %>% nrow()
    stats[i,5]=min(x[i],na.rm = TRUE)
    stats[i,6]=max(x[i],na.rm = TRUE)
    stats[i,7]=var(x[i],na.rm = TRUE)
  }
  stats
}

purchase_num <- purchase %>% select(last_update_days_ago
                                    ,X1st_update_days_ago
                                    ,Freq)
Outliers(purchase_num)
#------------------------------------------------------


#------------------------------------------------------
# Normalize data
#------------------------------------------------------
# define a min-max normalization function to bring scale of data between 0 and 1
normalize <- function(x)
{
  return ((x - min(x))/(max(x) - min(x)))
}

# apply normalization to numerical features
purchase_n <- purchase_num %>% 
  # min-max normalization
 sapply(.,normalize) %>% 
  # z-score normalization
#sapply(.,scale) %>% 
  as.data.frame()

summary(purchase_n)

# combine factors and new normalized numerical features
purchase <- data.frame(purchase_f, purchase_n)
summary(purchase)
#------------------------------------------------------


#------------------------------------------------------
# Load tree-related packages
#------------------------------------------------------
# Load rpart package for decision tree
require(rpart)
# Load rpart.plot package for visualizing decision trees
require(rpart.plot)
# Load rattle package for visualizing decision trees
#install.packages('rattle')
require(rattle)
#------------------------------------------------------


#------------------------------------------------------
# Attribute selection (FILTER approach)
#------------------------------------------------------
# Load FSelector package for feature selection
require(FSelector)

# the function information.gain(dv ~ iv, data) weights features by information gain
#    dv is dependent variable
#    iv is independent variable
#   If iv = ".", then it will weight all other features (that are not dv)
weights <- information.gain(Purchase ~ ., purchase)
print(weights)

# Select top n=15 most informative features
features <- cutoff.k(weights, 15) 
print(features)
purchase_s <- subset(purchase, select = c(features,"Purchase"))

#these are the top n=15 features ranked plus the dependent variable
summary(purchase_s)

# Create training and testing subsets of data (80/20%)
require(caret)
set.seed(2983)
train.rows <- createDataPartition(y = purchase_s$Purchase,
                                  p = .80, list = FALSE)

purch_train <- purchase_s[train.rows, ]

# summarize training and testing data
str(purch_train)
head(purch_train)
purch_test  <- purchase_s[-train.rows, ]
str(purch_test)
head(purch_test)
#------------------------------------------------------


#------------------------------------------------------
# Build and evaluate model using cross-validation
#------------------------------------------------------
# 10-fold cross validation (this function does stratified sampling)
set.seed(2983)
folds <- createFolds(purch_train$Purchase, k = 10)
str(folds)
train<-function(x){
  purch_train[-x, ]
}

# create a list of 10 folds training data
train_list <-lapply(folds,train)
#------------------------------------------------------


#------------------------------------------------------
# Create 5 functions that handles classification model building 
#------------------------------------------------------
# 1. naive bayes model 
#------------------------------------------------------
library(e1071)
nb<-function(x){
  naiveBayes(Purchase ~ ., data = x)
}
#------------------------------------------------------
# 2. decision tree model
#------------------------------------------------------
tree<- function(x){
  rpart(Purchase ~ ., data = x, parms=list(split='infomation'),control = rpart.control(minsplit=4,minbucket = 2, cp = 0.01))
} 
#------------------------------------------------------
# 3. random forest model
#------------------------------------------------------
library(randomForest)
rf <- function(x){
  randomForest(Purchase ~ ., data = x, mtry=3, ntree=500)
}
#------------------------------------------------------
# 4. ssupport vector machine
#------------------------------------------------------
svmf <- function(x){
  svm(Purchase ~ ., data = x, kernel='radial')
}

# 5. neural network
#------------------------------------------------------
library(nnet)
nn<- function(x){
  nnet(Purchase ~ ., data = x,size=2,rang=0.5, decay = 5e-4)
}
#------------------------------------------------------


#------------------------------------------------------
# create a cross validation function and evaluation
#------------------------------------------------------
cv <- function(x,y){
  # Generating testing sets
  test <- purch_train[x, ]
  # Predict class labels for testing data
  predicted <- predict(y, test, type = "class")
    # Calculating F-measure
  actual <- test$Purchase
  num.pred.positive <- sum(predicted==1)
  num.act.positive <- sum(actual==1)
  num.both.positive <- sum(predicted==1 & actual==1)
  precision <- num.both.positive / num.pred.positive
  recall <- num.both.positive / num.act.positive
  Fmeasure <- 2 * precision * recall / (precision + recall)
  return(Fmeasure)
}
#------------------------------------------------------


#------------------------------------------------------
# apply each classification model to each fold
# and then apply each model to cross-validation process
#------------------------------------------------------
model_list<-lapply(train_list, nb)
results <- mapply(cv,folds,model_list)
summary(results)
sd(results)

model_list<-lapply(train_list, tree)
results <- mapply(cv,folds,model_list)
summary(results)
sd(results)

model_list<-lapply(train_list, rf)
results <- mapply(cv,folds,model_list)
summary(results)
sd(results)

model_list<-lapply(train_list, svmf)
results <- mapply(cv,folds,model_list)
summary(results)
sd(results)

model_list<-lapply(train_list, nn)
results <- mapply(cv,folds,model_list)
summary(results)
sd(results)
#------------------------------------------------------


#------------------------------------------------------
# 6. logistic regression
# This data is not converging so returns a warning message for each fold
#------------------------------------------------------
glm_cv <- function(x){
  # Generating training and testing sets
  train <- purch_train[-x, ]
  test <- purch_train[x, ]
  # Build a classifier
  model <- glm(Purchase ~ .,family=binomial(link='logit'), data = train)
  # Predict class labels for testing data
  predicted <- predict(model, test, type = "response")
  predicted <- ifelse(predicted > 0.5,1,0)
  # Calculating F-measure
  actual <- test$Purchase
  num.pred.positive <- sum(predicted==1)
  num.act.positive <- sum(actual==1)
  num.both.positive <- sum(predicted==1 & actual==1)
  precision <- num.both.positive / num.pred.positive
  recall <- num.both.positive / num.act.positive
  Fmeasure <- 2 * precision * recall / (precision + recall)
  return(Fmeasure)
}

# Apply the function on each fold, store the results into a new vector
results <- sapply(folds, glm_cv)
# Calculate the mean and standard deviation of accuracy
summary(results)
sd(results)
#------------------------------------------------------


#------------------------------------------------------
# Build a model with parameters that give the best performance
#------------------------------------------------------
# Train a decision tree classifier using the rpart() function
model <- rpart(Purchase ~ .
              , data = purch_train
              , parms=list(split='information')
              ,control = rpart.control(minsplit=4,minbucket = 2, cp = 0.01))
#------------------------------------------------------


#------------------------------------------------------
# Evaluate performance of that model on the test dataset
#------------------------------------------------------
pred <- predict(model, purch_test, type = "class")

# Alternative way to get performance (from caret package)
confusionMatrix(pred, purch_test$Purchase, positive = "1")

# Type = "prob" specifies the output to be class probabilities
pred_prob <- predict(model, purch_test, type = "prob")

head(pred_prob)
#------------------------------------------------------


#------------------------------------------------------
# Visualize the tree
#------------------------------------------------------
# Basic textual output
print(model)

# Visualize the tree (lots of advanced options available)
prp(model)
#prp(tree, varlen = 3)

# Interactively prune tree (very cool!)
#newtree <- prp(tree,snip=TRUE)$obj
#prp(newtree)

# More elaborate plot from rattle package
fancyRpartPlot(model)


# Load ROCR package for plotting ROC/lift curve
require(ROCR)

# Creating the ROC curve
# Input to the first parameter is the class probability for "1"
results <- prediction(predictions = pred_prob[, 2], labels = purch_test$Purchase)
ROC <- performance(results, measure = "tpr", x.measure = "fpr")
plot(ROC, main = "ROC curve for purchase", col = "red",type='b')
#------------------------------------------------------



# -------------------------------------------------------------------------------------------------------------
# -------------------------------------------------------------------------------------------------------------
# Section 2: This section predicts how much a customer is likely to puchase 
# -------------------------------------------------------------------------------------------------------------
# -------------------------------------------------------------------------------------------------------------


#------------------------------------------------------
# Pre-processing
#------------------------------------------------------
# sequence number is not informative for classification, exclude sequence number and purchase
purchase <- data %>% select(-sequence_number,-Purchase)

# change data type to factors
purchase_f <- purchase %>% 
  select(-last_update_days_ago
         ,-X1st_update_days_ago
         ,-Freq
         ,-Spending) %>% 
  sapply(.,as.factor) %>%
  as.data.frame()

str(purchase_f)
#------------------------------------------------------


#------------------------------------------------------
# Normalize data
#------------------------------------------------------
# min-max normalization
normalize <- function(x)
{
  return ((x - min(x))/(max(x) - min(x)))
}

#apply normalization to numerical features
purchase_n <- purchase %>% 
  select(last_update_days_ago
         ,X1st_update_days_ago
         ,Freq) %>% 
 # sapply(.,normalize) %>% #min-max
  sapply(.,scale)%>% #z-score transformation
  as.data.frame()

summary(purchase_n)
#combine factors and normalized numerical features
purchase <- data.frame(purchase_f, purchase_n,purchase%>% select(Spending))
summary(purchase)
#------------------------------------------------------


#------------------------------------------------------
# Feature selection (FILTER approach)
#------------------------------------------------------
# Load FSelector package for feature selection
require(FSelector)

# information.gain(dv ~ iv, data) weights features by information gain
#    dv is outcome variable
#    iv is features
#    If iv = ".", then it will weight all other features (that are not dv)
weights <- information.gain(Spending ~ ., purchase)
print(weights)

# Select top most informative features
features <- cutoff.k(weights, 6)
print(features)
purchase_s <- subset(purchase, select = c(features,"Spending"))
summary(purchase_s)

# Create train/test partitions (80/20%)
require(caret)
set.seed(2987)
train.rows <- createDataPartition(y = purchase_s$Spending,
                                  p = .80, list = FALSE)
purch_train <- purchase_s[train.rows, ]

purch_test  <- purchase_s[-train.rows, ]

# Create train/validation partitions (80/20%)
set.seed(2987)
train.rows <- createDataPartition(y = purch_train$Spending,
                                  p = .80, list = FALSE)
purch_train2 <- purch_train[train.rows, ]

purch_valid  <- purch_train[-train.rows, ]

# Train a decision tree classifier using the rpart() function
tree <- rpart(Spending ~ ., data = purch_train2,control = rpart.control(minsplit=6,minbucket = 3, cp = 0.05))
#------------------------------------------------------


#------------------------------------------------------
# evaluate on the validation dataset
#------------------------------------------------------
pred <- predict(tree, purch_valid)

# Calculate prediction errors
error <- pred - purch_valid$Spending
# Mean absolute error
MAE <- mean(abs(error))
print(MAE)
# Root mean squared error
RMSE <- sqrt(mean(error^2))
print(RMSE)
#------------------------------------------------------

#------------------------------------------------------
# Train and evaluate on the test dataset
#------------------------------------------------------
tree <- rpart(Spending ~ ., data = purch_train,control = rpart.control(minsplit=6,minbucket = 3, cp = 0.05))
pred <- predict(tree, purch_test)

# Calculate prediction errors
error <- pred - purch_test$Spending
# Mean absolute error
MAE <- mean(abs(error))
print(MAE)
# Root mean squared error
RMSE <- sqrt(mean(error^2))
print(RMSE)
#------------------------------------------------------


#------------------------------------------------------
# Visualize the tree
#------------------------------------------------------
# Basic textual output
print(tree)

# Visualize the tree (lots of advanced options available)
prp(tree)
#prp(tree, varlen = 3)

# Interactively prune tree (very cool!)
#newtree <- prp(tree,snip=TRUE)$obj
#prp(newtree)

# More elaborate plot from rattle package
fancyRpartPlot(tree)

#CREATE LIFT curve
ActualByModel <- purch_test[order(pred, decreasing=T), 7] #spending
ActualCumulative <- cumsum(ActualByModel)
plot(ActualCumulative, type="l")
l <- length(ActualCumulative)
segments(x0 = 0, y0 = 0, 
         x1 = l, y1 = ActualCumulative[l], 
         col = "red", lwd = 2)
# The curve indicates how much accumulative spending we will get if following the model, 
# in contrast to the random.
#------------------------------------------------------

