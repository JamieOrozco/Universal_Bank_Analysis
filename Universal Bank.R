install.packages("forecast")
install.packages("Metrics")
install.packages("caret")
install.packages("e1071")

library(forecast)
library(Metrics)

# Load File
UniversalBank.df <- read.csv("UniversalBank.csv", fileEncoding = "UTF-8-BOM") 

# View
View(UniversalBank.df)
str(UniversalBank.df)

# Drop Id
UniversalBank.df <- UniversalBank.df[-1]

# set as factor?
UniversalBank.df$PersonalLoan <- as.factor(UniversalBank.df$PersonalLoan)
              
# Table, 1 is yes                                                                                
table(UniversalBank.df$PersonalLoan)

# Create normalization function, Max and Min
normalize <- function(x) {
  return ((x - min(x)) / (max(x) - min(x)))
}                                         

# create normalization function based on z-score
normalize.z <- function(x) {
  return ((x - mean(x)) / sd(x))
}

# Normalize the data _N IS MAX _Z IS ZSCORE
UniversalBank.df_n <- as.data.frame(lapply(UniversalBank.df[1:8], normalize))

UniversalBank.df_z <- as.data.frame(lapply(UniversalBank.df[1:8], normalize.z))

# Confirm that normalization worked
summary(UniversalBank.df_n$Experience)
summary(UniversalBank.df_z$Experience)

# Partition the data
set.seed(500)

train.index <- sample(c(1:dim(UniversalBank.df_n)[1]), dim(UniversalBank.df_n)[1]*0.6)  

train.df_n <- UniversalBank.df_n[train.index, ]
valid.df_n <- UniversalBank.df_n[-train.index, ]

# Create labels for training and test data
train.labels <- UniversalBank.df[train.index, 9]
valid.labels <- UniversalBank.df[-train.index, 9]

# load the "class" library
library(class)

UniversalTest_pred <- knn(train = train.df_n, test = valid.df_n,
                             cl = train.labels, k = 13)

# Compare predicted versus actual
data.frame(predicted = UniversalTest_pred, actual = valid.labels)

install.packages("gmodels") #If not installed

# load the "gmodels" library
library(gmodels)

# Create the cross tabulation of predicted vs. actual
CrossTable(x = valid.labels, y = UniversalTest_pred, prop.chisq = FALSE)

require(caret)
require(e1071)

# Matrix
confusionMatrix(as.factor(UniversalTest_pred), as.factor(valid.labels))


# Part 2
# Create training based on z-score normalization
train.index <- sample(c(1:dim(UniversalBank.df_z)[1]), dim(UniversalBank.df_z)[1]*0.6)  

train_z.df <- UniversalBank.df_z[train.index, ]
valid_z.df <- UniversalBank.df_z[-train.index, ]

# Re-classify test cases usign z-score normalizaiton
UniversalTest_pred_z <- knn(train = train_z.df, test = valid_z.df,
                               cl = train.labels, k = 13)

# Compare predicted versus actual
data.frame(predicted = UniversalTest_pred_z, actual = valid.labels)

# Create the cross tabulation of predicted vs. actual
CrossTable(x = valid.labels, y = UniversalTest_pred_z,
           prop.chisq = FALSE)

require(caret)
require(e1071)

# Confusion matrix
confusionMatrix(as.factor(UniversalTest_pred_z), as.factor(valid.labels))

# Loop to run knn for k-values listed and provide a confusionMatrix for each model
for (i in c(1, 3, 5, 7, 9, 11, 13, 15, 17, 19))  {
  UniversalTest_pred <- knn(train = train.df_n, test = valid.df_n, 
                               cl = train.labels, k = i)
  print(paste("Nearest Neighbor for k = ", i ))
  print(confusionMatrix(as.factor(UniversalTest_pred), as.factor(valid.labels)))
}
