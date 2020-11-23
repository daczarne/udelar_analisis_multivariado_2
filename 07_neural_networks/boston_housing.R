#======================#
#### BOSTON HOUSING ####
#======================#


library(magrittr)
library(keras)
library(tfdatasets)


# Feature engineering -----------------------------------------------------


# We'll use the Boston housing data set to build a NN for a regression problem

# A list of length 2:
#   train: a list of length 2
#     x: a matrix of 404 observations and 13 covariates for housing characteristics in Boston suburbes
#     y: a 1 dim array of length 404 with house prices in 1000s of US dollars
#   test: a list of length 2
#     x: a matrix of 102 observations and 13 covariates for housing characteristics in Boston suburbes
#     y: a 1 dim array of length 102 with house prices in 1000s of US dollars
boston_housing <- keras::dataset_boston_housing()



c(train_data, train_labels) %<-% boston_housing$train
c(test_data, test_labels) %<-% boston_housing$test



# Model building ----------------------------------------------------------




# Model training ----------------------------------------------------------




# Prediction --------------------------------------------------------------




#===============#
#### THE END ####
#===============#