install.packages("caret")
install.packages("e1071")
#read the data
library(readr)
library(dplyr)
library(caret) 
library(tidyverse)
setwd("F:/semester 8/Distributed Computing/Project_Data")
#read data from files
train_dataset<-read.csv("train.csv")
test_dataset <-read_csv("test.csv")

# # Columns to remove from datasets
# cols_to_remove <- c("Street", "Alley", "Utilities", "PoolQC", "Fence", "MiscFeature", "FireplaceQu")
# 
# # Remove the columns from datasets
# train_dataset <- select(train_dataset, -one_of(cols_to_remove))
# test_dataset <- select(test_dataset, -one_of(cols_to_remove))



# Load the dataset into an R dataframe


# Specify the list of columns to keep
columns_to_keep <- c("Id","LotFrontage", "LotArea", "OverallQual", "OverallCond", "YearBuilt", "YearRemodAdd",
                     "MasVnrArea", "BsmtFinSF1", "BsmtFinSF2", "BsmtUnfSF", "TotalBsmtSF", "X1stFlrSF",
                     "X2ndFlrSF", "LowQualFinSF", "GrLivArea", "BsmtFullBath", "BsmtHalfBath", "FullBath",
                     "HalfBath", "BedroomAbvGr", "KitchenAbvGr", "TotRmsAbvGrd", "Fireplaces", "GarageCars",
                     "GarageArea", "WoodDeckSF", "OpenPorchSF", "EnclosedPorch", "X3SsnPorch", "ScreenPorch",
                     "PoolArea", "MiscVal","SalePrice")

# Drop all columns except for the ones in the list
train_dataset <- train_dataset[, columns_to_keep]
test_dataset <- train_dataset[, columns_to_keep]




# Function to impute missing values 
impute_missing <- function(data) {
  cols_with_missing <- data %>% 
    select_if(~any(is.na(.))) %>% 
    names()
  
  for (col in cols_with_missing) {
    if (is.numeric(data[[col]])) {
      data[[col]] <- ifelse(is.na(data[[col]]), mean(data[[col]], na.rm = TRUE), data[[col]])
    } else {
      data[[col]] <- ifelse(is.na(data[[col]]), mode(data[[col]]), data[[col]])
    }
  }
  
  data  
}

# Load and impute datasets
train_dataset <- impute_missing(train_dataset)
test_dataset <- impute_missing(test_dataset) 





#The important features to predict house price are: LotFrontage, LotArea, OverallQual, OverallCond, YearBuilt, YearRemodAdd, MasVnrArea, BsmtFinSF1, BsmtFinSF2, BsmtUnfSF, TotalBsmtSF, X1stFlrSF, X2ndFlrSF, LowQualFinSF, GrLivArea, BsmtFullBath, BsmtHalfBath, FullBath, HalfBath, BedroomAbvGr, KitchenAbvGr, TotRmsAbvGrd, Fireplaces, GarageCars, GarageArea, WoodDeckSF, OpenPorchSF, EnclosedPorch, X3SsnPorch, ScreenPorch, PoolArea, MiscVal. There are 31 features in total.



# #check the data set variables to get the missing 
# vars_to_check <- c("Id", "MSSubClass", "MSZoning", "LotArea", "LotShape",
#                    "LandContour", "LotConfig", "LandSlope", "Neighborhood", "Condition1",
#                    "Condition2", "BldgType", "HouseStyle", "OverallQual", "OverallCond",
#                    "YearBuilt", "YearRemodAdd", "RoofStyle", "RoofMatl", "Exterior1st",
#                    "Exterior2nd", "MasVnrType", "MasVnrArea", "ExterQual", "ExterCond",
#                    "Foundation", "BsmtQual", "BsmtCond", "BsmtExposure", "BsmtFinType1",
#                    "BsmtFinSF1", "BsmtFinType2", "BsmtFinSF2", "BsmtUnfSF", "TotalBsmtSF",
#                    "Heating", "HeatingQC", "CentralAir", "Electrical", "X1stFlrSF",
#                    "X2ndFlrSF", "LowQualFinSF", "GrLivArea", "BsmtFullBath", "BsmtHalfBath",
#                    "FullBath", "HalfBath", "BedroomAbvGr", "KitchenAbvGr", "KitchenQual",
#                    "TotRmsAbvGrd", "Functional", "Fireplaces", "GarageType",
#                    "GarageYrBlt", "GarageFinish", "GarageCars", "GarageArea", "GarageQual",
#                    "GarageCond", "PavedDrive", "WoodDeckSF", "OpenPorchSF", "EnclosedPorch",
#                    "X3SsnPorch", "ScreenPorch", "PoolArea", "MiscVal", "MoSold", "YrSold",
#                    "SaleType", "SaleCondition", "SalePrice","LotFrontage")

vars_to_check <- c("Id", "LotFrontage", "LotArea", "OverallQual", "OverallCond", "YearBuilt", "YearRemodAdd",
                   "MasVnrArea", "BsmtFinSF1", "BsmtFinSF2", "BsmtUnfSF", "TotalBsmtSF", "X1stFlrSF",
                   "X2ndFlrSF", "LowQualFinSF", "GrLivArea", "BsmtFullBath", "BsmtHalfBath", "FullBath",
                   "HalfBath", "BedroomAbvGr", "KitchenAbvGr", "TotRmsAbvGrd", "Fireplaces", "GarageCars",
                   "GarageArea", "WoodDeckSF", "OpenPorchSF", "EnclosedPorch", "X3SsnPorch", "ScreenPorch",
                   "PoolArea", "MiscVal","SalePrice")


# Apply the function to each variable
vis_train <- sapply(vars_to_check, function(var) sum(is.na(train_dataset[[var]])))
vis_test <- sapply(vars_to_check, function(var) sum(is.na(test_dataset[[var]])))

# Print the results
print(vis_train)
print(vis_test)

# 
# #convert the data to factor to convert it to numeric
# convert_to_factor <- function(data, column_names) {
#   for (col in column_names) {
#     data[[col]] <- factor(data[[col]])
#   }
#   return(data)
# }
# 
# train_dataset <- convert_to_factor(train_dataset, c("MSZoning", "HouseStyle","LotShape", "LotConfig", "LandContour", "LandSlope", "Neighborhood", "Condition1", 
#                                                     "BldgType", "RoofStyle", "RoofMatl", "Exterior1st", "Condition2", "Exterior2nd", "MasVnrType", 
#                                                     "RoofMatl", "ExterQual", "ExterCond", "Foundation", "BsmtQual", "BsmtCond", "BsmtExposure", 
#                                                     "BsmtFinType1", "BsmtFinType2", "Heating","HeatingQC", "CentralAir", "Electrical", "KitchenQual", "Functional", "GarageType",
#                                                     "GarageFinish", "GarageQual", "GarageCond", "PavedDrive", "SaleType", "SaleCondition"))
# 
# 
# test_dataset <- convert_to_factor(test_dataset, c("MSZoning", "HouseStyle","LotShape", "LotConfig", "LandContour", "LandSlope", "Neighborhood", "Condition1", 
#                                                   "BldgType", "RoofStyle", "RoofMatl", "Exterior1st", "Condition2", "Exterior2nd", "MasVnrType", 
#                                                   "RoofMatl", "ExterQual", "ExterCond", "Foundation", "BsmtQual", "BsmtCond", "BsmtExposure", 
#                                                   "BsmtFinType1", "BsmtFinType2", "Heating","HeatingQC", "CentralAir", "Electrical", "KitchenQual", "Functional", "GarageType",
#                                                   "GarageFinish", "GarageQual", "GarageCond", "PavedDrive", "SaleType", "SaleCondition"))
# 
# 
# 
# convert_to_numeric <- function(data, column_names) {
#   for (col in column_names) {
#     data[[col]] <- as.numeric(data[[col]])
#   }
#   return(data)
# }
# 
# train_dataset <- convert_to_numeric(train_dataset, c("MSZoning","HouseStyle", "LotShape", "LotConfig", "LandContour", "LandSlope", "Neighborhood", "Condition1",
#                                                    "BldgType", "RoofStyle", "RoofMatl", "Exterior1st", "Condition2", "Exterior2nd", "MasVnrType",
#                                                    "RoofMatl", "ExterQual", "ExterCond", "Foundation", "BsmtQual", "BsmtCond", "BsmtExposure",
#                                                    "BsmtFinType1", "BsmtFinType2", "Heating","HeatingQC", "CentralAir", "Electrical", "KitchenQual", "Functional", "GarageType",
#                                                    "GarageFinish", "GarageQual", "GarageCond", "PavedDrive", "SaleType", "SaleCondition"))
# 
# 
# test_dataset <- convert_to_numeric(test_dataset, c("MSZoning","HouseStyle", "LotShape", "LotConfig", "LandContour", "LandSlope", "Neighborhood", "Condition1",
#                                                     "BldgType", "RoofStyle", "RoofMatl", "Exterior1st", "Condition2", "Exterior2nd", "MasVnrType",
#                                                     "RoofMatl", "ExterQual", "ExterCond", "Foundation", "BsmtQual", "BsmtCond", "BsmtExposure",
#                                                     "BsmtFinType1", "BsmtFinType2", "Heating","HeatingQC", "CentralAir", "Electrical", "KitchenQual", "Functional", "GarageType",
#                                                     "GarageFinish", "GarageQual", "GarageCond", "PavedDrive", "SaleType", "SaleCondition"))
# 



#convert the data to factor to convert it to numeric
convert_to_factor <- function(data, column_names) {
  for (col in column_names) {
    data[[col]] <- factor(data[[col]])
  }
  return(data)
}

train_dataset <- convert_to_factor(train_dataset, c("LotFrontage", "LotArea", "OverallQual", "OverallCond", "YearBuilt", "YearRemodAdd",
                                                    "MasVnrArea", "BsmtFinSF1", "BsmtFinSF2", "BsmtUnfSF", "TotalBsmtSF", "X1stFlrSF",
                                                    "X2ndFlrSF", "LowQualFinSF", "GrLivArea", "BsmtFullBath", "BsmtHalfBath", "FullBath",
                                                    "HalfBath", "BedroomAbvGr", "KitchenAbvGr", "TotRmsAbvGrd", "Fireplaces", "GarageCars",
                                                    "GarageArea", "WoodDeckSF", "OpenPorchSF", "EnclosedPorch", "X3SsnPorch", "ScreenPorch",
                                                    "PoolArea", "MiscVal"))


test_dataset <- convert_to_factor(test_dataset, c("LotFrontage", "LotArea", "OverallQual", "OverallCond", "YearBuilt", "YearRemodAdd",
                                                  "MasVnrArea", "BsmtFinSF1", "BsmtFinSF2", "BsmtUnfSF", "TotalBsmtSF", "X1stFlrSF",
                                                  "X2ndFlrSF", "LowQualFinSF", "GrLivArea", "BsmtFullBath", "BsmtHalfBath", "FullBath",
                                                  "HalfBath", "BedroomAbvGr", "KitchenAbvGr", "TotRmsAbvGrd", "Fireplaces", "GarageCars",
                                                  "GarageArea", "WoodDeckSF", "OpenPorchSF", "EnclosedPorch", "X3SsnPorch", "ScreenPorch",
                                                  "PoolArea", "MiscVal"))



convert_to_numeric <- function(data, column_names) {
  for (col in column_names) {
    data[[col]] <- as.numeric(data[[col]])
  }
  return(data)
}

train_dataset <- convert_to_numeric(train_dataset, c("LotFrontage", "LotArea", "OverallQual", "OverallCond", "YearBuilt", "YearRemodAdd",
                                                     "MasVnrArea", "BsmtFinSF1", "BsmtFinSF2", "BsmtUnfSF", "TotalBsmtSF", "X1stFlrSF",
                                                     "X2ndFlrSF", "LowQualFinSF", "GrLivArea", "BsmtFullBath", "BsmtHalfBath", "FullBath",
                                                     "HalfBath", "BedroomAbvGr", "KitchenAbvGr", "TotRmsAbvGrd", "Fireplaces", "GarageCars",
                                                     "GarageArea", "WoodDeckSF", "OpenPorchSF", "EnclosedPorch", "X3SsnPorch", "ScreenPorch",
                                                     "PoolArea", "MiscVal"))


test_dataset <- convert_to_numeric(test_dataset, c("LotFrontage", "LotArea", "OverallQual", "OverallCond", "YearBuilt", "YearRemodAdd",
                                                   "MasVnrArea", "BsmtFinSF1", "BsmtFinSF2", "BsmtUnfSF", "TotalBsmtSF", "X1stFlrSF",
                                                   "X2ndFlrSF", "LowQualFinSF", "GrLivArea", "BsmtFullBath", "BsmtHalfBath", "FullBath",
                                                   "HalfBath", "BedroomAbvGr", "KitchenAbvGr", "TotRmsAbvGrd", "Fireplaces", "GarageCars",
                                                   "GarageArea", "WoodDeckSF", "OpenPorchSF", "EnclosedPorch", "X3SsnPorch", "ScreenPorch",
                                                   "PoolArea", "MiscVal"))





#check the missing values from data set
result <- sapply(vars_to_check, function(var) sum(is.na(train_dataset[[var]])))
result1 <- sapply(vars_to_check, function(var) sum(is.na(test_dataset[[var]])))
print(result)
print(result1)


##Check if columns you want to select exist
##Function to apply Min-Max scaling
# min_max_scale <- function(x) {
#   return ((x - min(x)) / (max(x) - min(x)))
# }
# 
# # Check if columns you want to select exist
# cols_to_scale <- c("MSSubClass", "MSZoning", "LotArea", "LotShape",
#                    "LandContour", "LotConfig", "LandSlope", "Neighborhood", "Condition1",
#                    "Condition2", "BldgType", "HouseStyle", "OverallQual", "OverallCond",
#                    "YearBuilt", "YearRemodAdd", "RoofStyle", "RoofMatl", "Exterior1st",
#                    "Exterior2nd", "MasVnrType", "MasVnrArea", "ExterQual", "ExterCond",
#                    "Foundation", "BsmtQual", "BsmtCond", "BsmtExposure", "BsmtFinType1",
#                    "BsmtFinSF1", "BsmtFinType2", "BsmtFinSF2", "BsmtUnfSF", "TotalBsmtSF",
#                    "Heating", "HeatingQC", "CentralAir", "Electrical", "X1stFlrSF",
#                    "X2ndFlrSF", "LowQualFinSF", "GrLivArea", "BsmtFullBath", "BsmtHalfBath",
#                    "FullBath", "HalfBath", "BedroomAbvGr", "KitchenAbvGr", "KitchenQual",
#                    "TotRmsAbvGrd", "Functional", "Fireplaces", "GarageType",
#                    "GarageYrBlt", "GarageFinish", "GarageCars", "GarageArea", "GarageQual",
#                    "GarageCond", "PavedDrive", "WoodDeckSF", "OpenPorchSF", "EnclosedPorch",
#                    "X3SsnPorch", "ScreenPorch", "PoolArea", "MiscVal", "MoSold", "YrSold",
#                    "SaleType", "SaleCondition","LotFrontage")
# 
# cols_to_scale <- trimws(cols_to_scale)
# 
# # Apply Min-Max scaling to the selected columns
# train_dataset[, cols_to_scale] <- lapply(train_dataset[, cols_to_scale], min_max_scale)


#Check if columns you want to select exist
#Function to apply Min-Max scaling
min_max_scale <- function(x) {
  return ((x - min(x)) / (max(x) - min(x)))
}

# Check if columns you want to select exist
cols_to_scale <- c("LotFrontage", "LotArea", "OverallQual", "OverallCond", "YearBuilt", "YearRemodAdd",
                   "MasVnrArea", "BsmtFinSF1", "BsmtFinSF2", "BsmtUnfSF", "TotalBsmtSF", "X1stFlrSF",
                   "X2ndFlrSF", "LowQualFinSF", "GrLivArea", "BsmtFullBath", "BsmtHalfBath", "FullBath",
                   "HalfBath", "BedroomAbvGr", "KitchenAbvGr", "TotRmsAbvGrd", "Fireplaces", "GarageCars",
                   "GarageArea", "WoodDeckSF", "OpenPorchSF", "EnclosedPorch", "X3SsnPorch", "ScreenPorch",
                   "PoolArea", "MiscVal")

cols_to_scale <- trimws(cols_to_scale)

# Apply Min-Max scaling to the selected columns
train_dataset[, cols_to_scale] <- lapply(train_dataset[, cols_to_scale], min_max_scale)


#/////////////////////////////////////////////////////////////////////////////////////





data_df <- as.data.frame(train_dataset)
data_df_test <- as.data.frame(test_dataset)







library(caret)
# model <- lm(SalePrice ~ MSZoning + LotShape + LotConfig + LandContour + LandSlope + Neighborhood + Condition1 +
#               BldgType + RoofStyle + RoofMatl + Exterior1st + Condition2 + Exterior2nd + MasVnrType +
#               ExterQual + ExterCond + Foundation + BsmtQual + BsmtCond + BsmtExposure + BsmtFinType1 +
#               BsmtFinType2 + Heating + HeatingQC + CentralAir + Electrical + KitchenQual + Functional +
#               GarageType + GarageFinish + GarageQual + GarageCond + PavedDrive + SaleType + SaleCondition+LotFrontage,
#             data = data_df)

model <- lm(SalePrice ~ LotFrontage + LotArea + OverallQual + OverallCond + YearBuilt + YearRemodAdd +
              MasVnrArea + BsmtFinSF1 + BsmtFinSF2 + BsmtUnfSF + TotalBsmtSF + X1stFlrSF +
              X2ndFlrSF +LowQualFinSF + GrLivArea + BsmtFullBath + BsmtHalfBath + FullBath +
              HalfBath + BedroomAbvGr + KitchenAbvGr + TotRmsAbvGrd + Fireplaces + GarageCars +
              GarageArea + WoodDeckSF + OpenPorchSF + EnclosedPorch + X3SsnPorch + ScreenPorch +
              PoolArea + MiscVal,
            data = data_df)


# Split data to training and validation 
trainIndex <- createDataPartition(data_df$SalePrice, p = 0.8, list = FALSE)
train_data <- data_df[trainIndex, ]
validation_data <- data_df[-trainIndex, ]


#Fit linear regression model
linear_model <- lm(SalePrice ~ ., data = data_df)

# Make predictions on validation set
predictions <- predict(linear_model, newdata = validation_data)

# Calculate MSE, RMSE and R2
mse <- mean((predictions - validation_data$SalePrice)^2)
rmseLM <- sqrt(mse)
R2LM <- 1 - mse/var(validation_data$SalePrice)

# Report evaluation metrics
cat(paste0("Linear Regression MSE: ", round(mse, 2)))
cat(paste0("Linear Regression RMSE: ", round(rmseLM, 2)))
cat(paste0("Linear Regression R-squared: ", round(R2LM, 2)))

#predictions <- predict(linear_model, newdata = validation_data)

# Subset validation_data to include only rows used for prediction
sub_validation_data <- validation_data[1:length(predictions), ]

# Create a data frame with Id and SalePrice predictions
myData <- data.frame(Id = sub_validation_data$Id,
                     Actual = sub_validation_data$SalePrice,
                     Predicted = predictions)

# Create scatter plot of Actual vs Predicted SalePrice
myPlot <- ggplot(data = myData, aes(x = Actual, y = Predicted)) +
  geom_point() +
  geom_abline(slope = 1, intercept = 0, linetype = "dashed", color = "red") +
  labs(title = "Actual vs Predicted SalePrice LR",x = "Actual SalePrice", y = "Predicted SalePrice")

# Show the plot
print(myPlot)


#fit for svm
library(e1071)
# Load required packages
svm_model <- svm(SalePrice ~ ., data = data_df)

predictions <- predict(svm_model, newdata = validation_data)

mse <- mean((predictions - validation_data$SalePrice)^2)
rmseSVM <- sqrt(mse)
R2SVM <- 1 - mse/var(validation_data$SalePrice)

cat(paste0("SVM MSE: ", round(mse, 2)))
cat(paste0("SVM RMSE: ", round(rmseSVM, 2)))
cat(paste0("SVM R-squared: ", round(R2SVM, 2)))

# Make predictions on validation set
#predictions <- predict(svm_model, newdata = validation_data)

# Subset validation_data to include only rows used for prediction
sub_validation_data <- validation_data[1:length(predictions), ]

# Create a data frame with Id and SalePrice predictions
myData <- data.frame(Id = sub_validation_data$Id,
                     Actual = sub_validation_data$SalePrice,
                     Predicted = predictions)

# Create scatter plot of Actual vs Predicted SalePrice
myPlot <- ggplot(data = myData, aes(x = Actual, y = Predicted)) +
  geom_point() +
  geom_abline(slope = 1, intercept = 0, linetype = "dashed", color = "red") +
  labs(title = "Actual vs Predicted SalePrice SVM",x = "Actual SalePrice", y = "Predicted SalePrice")

# Show the plot
print(myPlot)




library(randomForest)

# Fit random forest model
rf_model <- randomForest(SalePrice ~ ., data = data_df)

# Make predictions on validation set 
predictions <- predict(rf_model, newdata = validation_data)

# Calculate MSE, RMSE and R2 
mse <- mean((predictions - validation_data$SalePrice)^2)     
rmseRF <- sqrt(mse)
R2RF <- 1 - mse/var(validation_data$SalePrice)

# Report evaluation metrics
cat(paste0("Random Forest MSE: ", round(mse, 2)))       
cat(paste0("Random Forest RMSE: ", round(rmseRF, 2)))
cat(paste0("Random Forest R-squared: ", round(R2RF, 2)))





# Make predictions on validation set 
predictions <- predict(rf_model, newdata = validation_data)

# Subset validation_data to include only rows used for prediction
sub_validation_data <- validation_data[1:length(predictions), ]

# Create a data frame with Id and SalePrice predictions
myData <- data.frame(Id = sub_validation_data$Id, 
                     Actual = sub_validation_data$SalePrice, 
                     Predicted = predictions)

# Create scatter plot of Actual vs Predicted SalePrice
myPlot <- ggplot(data = myData, aes(x = Actual, y = Predicted)) + 
  geom_point() + 
  geom_abline(slope = 1, intercept = 0, linetype = "dashed", color = "red") + 
  labs(title = "Actual vs Predicted SalePrice RF",x = "Actual SalePrice", y = "Predicted SalePrice")

# Show the plot
print(myPlot)


# Model names
models <- c("linear_model", "svm_model", "rf_model")

# Bar plot of RMSE
barplot(c(rmseLM, rmseSVM, rmseRF), 
        names.arg = models,
        xlab = "Models", 
        ylab = "RMSE", 
        main = "RMSE")

#Bar plot of R2
barplot(c(R2LM, R2SVM, R2RF),
        names.arg = models,
        xlab = "Models",
        ylab = "R2",
        main = "R2")










# Generate predictions for test dataset
predictions <- predict(rf_model, newdata = test_dataset)

# Create data frame with Id and SalePrice predictions
submission <- data.frame(Id = test_dataset$Id, SalePrice = predictions)

# Write data frame to CSV file
write.csv(submission, file = "newSubmissionData1.csv", row.names = FALSE)





# 
# # Load test data 
# testScript <- read.csv("testScript.csv")
# 
# # Load trained model
# save(rf_model, file = "rf_model.Rdata")
# load("rf_model.Rdata")
# predictions <- predict(rf_model, newdata = testScript)
# 
# # Create submission data frame
# submission <- data.frame(Id = testScript$Id, SalePrice = predictions)
# 
# # Write submission to CSV
# write.csv(submission, file = "subForTestScript.csv",row.names = FALSE)
# 







