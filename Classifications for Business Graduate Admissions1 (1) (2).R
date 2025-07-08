## --------------------------------------------------------------------------------------------------------------------------------------------------
#install.packages("e1071")
#install.package("caret")

rm(list=ls())

set.seed(123)  # set seed for consistency

# libraries
library(e1071) 
library(caret)
library(ggplot2)
library(dplyr)
library(mosaic)

#load libraries
data <- read.csv("~/Desktop/admission.csv", header = TRUE)
data$De <- as.factor(data$De)

data <- na.omit(data) # removing missing data 

# Select rows for training and testing
rows4train <- c(1:26, 32:54, 60:80)
rows4test <- c(27:31, 55:59, 81:85)

# spliting the data
train_data <- data[rows4train, ]
test_data <- data[rows4test, ]

# Check data structure and size
str(train_data)
str(test_data)
nrow(train_data)
nrow(test_data)



## --------------------------------------------------------------------------------------------------------------------------------------------------

# Part A: EDA

# Table Summary:
favstats_GPA <- favstats(GPA ~ Group, data = train_data)
favstats_GMAT <- favstats(GMAT ~ Group, data = train_data)
favstats_GPA
favstats_GMAT

# Scatterplot
ggplot(train_data, aes(x = GPA, y = GMAT, color = De)) +
  geom_point(size = 4, alpha = 0.7) +
  theme_minimal() +
  labs(title = "Scatterplot of GPA vs GMAT (Training Data)",
       x = "GPA", y = "GMAT") +
  scale_color_manual(values = c("red", "blue", "green"))

# Density plot for GPA
ggplot(train_data, aes(x = GPA, fill = De)) +
  geom_density(alpha = 0.8) +
  theme_minimal() +
  labs(title = "Density Plot of GPA (Training Data)",
       x = "GPA", y = "Density") +
  scale_fill_manual(values = c("red", "blue", "green"))

# Density plot for GMAT
ggplot(train_data, aes(x = GMAT, fill = De)) +
  geom_density(alpha = 0.8) +
  theme_minimal() +
  labs(title = "Density Plot of GMAT (Training Data)",
       x = "GMAT", y = "Density") +
  scale_fill_manual(values = c("red", "blue", "green"))



## --------------------------------------------------------------------------------------------------------------------------------------------------
set.seed(123) 
# perform hyper parameter tuning for an SVM with a linear kernel using tune function
tune_linear <- tune(svm, De ~ GPA + GMAT, data = train_data,
                        kernel = "linear",
                        ranges = list(cost = c(0.001, 0.01, 0.1, 1, 10, 100)),
                        tunecontrol = tune.control(cross = 10)) 

summary(tune_linear)


best_svm_linear <- tune_linear$best.model # getting the best model from the tuning 
predictions_linear <- predict(best_svm_linear, newdata = test_data) # make on predection on test data 
confusionMatrix(predictions_linear, test_data$De) # evaluate model performance with a confusion matrix



## --------------------------------------------------------------------------------------------------------------------------------------------------
set.seed(123) 
# perform hyper parameter tuning for support vector machine with polynomial kernel of degree two using tune function
tune_poly <- tune(svm, De ~ GPA + GMAT, data = train_data,
                      kernel = "polynomial",
                      ranges = list(cost = c(0.001, 0.01, 0.1, 1, 10, 100),
                                    degree = 2),  
                      tunecontrol = tune.control(cross = 10)) 
summary(tune_poly)

best_svm_poly <- tune_poly$best.model # getting the best model from the tuning 
predictions_poly <- predict(best_svm_poly, newdata = test_data) # make prediction on test data 
confusionMatrix(predictions_poly, test_data$De) # evaluate model performance with a confusion matrix


## --------------------------------------------------------------------------------------------------------------------------------------------------
set.seed(123) 
# perform hyper parameter tuning for support vector machine with a radial kernel with both 𝛾 and cost of
# parameter using tune function
tune_rbf <- tune(svm, De ~ GPA + GMAT, data = train_data,
                     kernel = "radial",
                     ranges = list(cost = c(0.1, 1, 10, 100, 1000),
                                   gamma = c(0.1, 1, 2, 3, 4)),   # our gammas
                     tunecontrol = tune.control(cross = 10))

best_cost <- tune_rbf$best.parameters$cost
best_gamma <- tune_rbf$best.parameters$gamma
print(paste(best_cost, best_gamma))

summary(tune_rbf)
best_svm_rbf <- tune_rbf$best.model  # getting the best model from the tuning
predictions_rbf <- predict(best_svm_rbf, newdata = test_data) # make prediction on test data 
confusionMatrix(predictions_rbf, test_data$De) # evaluate model performance with a confusion matrix



## --------------------------------------------------------------------------------------------------------------------------------------------------
# calculate accuracy for each SVM model
accuracy_linear <- sum(predictions_linear == test_data$De) / nrow(test_data)
accuracy_poly <- sum(predictions_poly == test_data$De) / nrow(test_data)
accuracy_rbf <- sum(predictions_rbf == test_data$De) / nrow(test_data)

# print accuracy results
print(paste("linear SVM Accuracy: ", accuracy_linear))
print(paste("polynomial SVM Accuracy: ", accuracy_poly))
print(paste("RBF SVM Accuracy: ", accuracy_rbf))

# print confusion matrices for each model

print("linear SVM Confusion Matrix:")
print(confusionMatrix(predictions_linear, test_data$De))

print("polynomial SVM Confusion Matrix:")
print(confusionMatrix(predictions_poly, test_data$De))

print("RBF SVM Confusion Matrix:")
print(confusionMatrix(predictions_rbf, test_data$De))



## --------------------------------------------------------------------------------------------------------------------------------------------------
set.seed(123) 
# perform hyper parameter tuning for support vector machine with polynomial kernel of degree three using tune function
tune_poly_3 <- tune(svm, De ~ GPA + GMAT, data = train_data,
                      kernel = "polynomial",
                      ranges = list(cost = c(0.001, 0.01, 0.1, 1, 10, 100),
                                    degree = 3),  
                      tunecontrol = tune.control(cross = 10)) 
summary(tune_poly_3)

best_svm_poly_3 <- tune_poly_3$best.model # getting the best model from the tuning 
predictions_poly_3 <- predict(best_svm_poly_3, newdata = test_data) # make prediction on test data 
confusionMatrix(predictions_poly_3, test_data$De) # evaluate model performance with a confusion matrix


## --------------------------------------------------------------------------------------------------------------------------------------------------
# SVM with sigmoid kernel
set.seed(123)

svm_sigmoid <- tune(
  svm, De ~ GPA + GMAT, data = train_data,  
  kernel = "sigmoid",
  ranges = list(
    gamma = 2^seq(-6, 0, by = 1),
    coef0 = seq(-2, 2, by = 0.5),
    cost = 2^seq(-2, 5, by = 1)
  ),
  tunecontrol = tune.control(cross = 10)  
)

best_sigmoid <- svm_sigmoid$best.model
print(best_sigmoid)
predictions <- predict(best_sigmoid, test_data)  

conf_matrix <- confusionMatrix(predictions, test_data$De)
print(conf_matrix)




## --------------------------------------------------------------------------------------------------------------------------------------------------
plot_svm_test <- function(model, train_data, test_data, title) {
  # find the GPA range 
  gpa_min <- min(train_data$GPA, test_data$GPA) - 0.2
  gpa_max <- max(train_data$GPA, test_data$GPA) + 0.2
  
  # find the GMAT range 
  gmat_min <- min(train_data$GMAT, test_data$GMAT) - 20
  gmat_max <- max(train_data$GMAT, test_data$GMAT) + 20
  
  # creating a grid over the GPA and GMAT space
  grid_points <- expand.grid(GPA = seq(gpa_min, gpa_max, length.out = 100),
                      GMAT = seq(gmat_min, gmat_max, length.out = 100))
  grid_points$De <- predict(model, newdata = grid_points) # getting the model's predictions for every point on the grid
  test_data$Predicted_De <- predict(model, newdata = test_data) #  # predicting the class for the testing set
  
   # plot the decision regions along with the test data points using the ggplot
  ggplot() +
    geom_tile(data = grid_points, aes(x = GPA, y = GMAT, fill = De), alpha = 0.3) +
    geom_point(data = test_data, aes(x = GPA, y = GMAT, color = Predicted_De, shape = De), size = 4) +
    theme_minimal() +
    labs(title = title, x = "GPA", y = "GMAT") +
    scale_color_manual(values = c("black", "yellow", "purple")) +
    theme(legend.position = "bottom")
}

# printing the plots:
plot_svm_test(best_svm_linear, train_data, test_data, "SVC (Linear) with Test Data")
plot_svm_test(best_svm_poly, train_data, test_data, "Polynomial SVM (Degree 2) with Test Data")
plot_svm_test(best_svm_rbf, train_data, test_data, "Radial SVM with Test Data")
plot_svm_test(best_sigmoid, train_data, test_data, "SVM with sigmoid kernel")
plot_svm_test(best_svm_poly_3, train_data, test_data, "Polynomial SVM (Degree 3) with Test Data")

