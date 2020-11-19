#=================================#
#### BAGGING AND RANDOM FOREST ####
#=================================#


library(magrittr)


# Bootstrap aggregation (bagging) is a general-purpose procedure for reducing the variance of a statistical learning method

boston <- MASS::Boston %>% tibble::as_tibble()


# Bagging -----------------------------------------------------------------

# A special case of random forest with m = p

base::set.seed(1L)
train <- base::sample(1L:base::nrow(boston), .75 * base::nrow(boston))


# mtry = 13 means that all 13 predictors should be used (ie: bagging)
# replace = TRUE means sampling with replacement
#   Other args to define sample are: strata, sampsize
# importance = TRUE means that predictor importance should be assessed
# ntree = 25 means that 25 trees will be grown
# nPerm = 1 (default) is the number of times OOB obs are permuted per tree for assessing variable importance
bag_boston <- randomForest::randomForest(
  medv ~ .,
  data = boston,
  subset = train,
  mtry = 13L,
  replace = TRUE,
  importance = TRUE,
  ntree = 100L,
  nPerm = 1L
)


base::class(bag_boston)
base::names(bag_boston)

# A randomForest object of type regression
bag_boston[["type"]]

# Predicted values for the OOB samples
bag_boston[["predicted"]]

# Varaible importance matrix
bag_boston[["importance"]]


# %IncMSE: mean decrease of accuracy in predictions on the OOB sample when a given variable is excluded from the model
# IncNodePurity: total decrease in node impurity that results from splits over that variable, averaged over all trees
#   for regression: mean impurity is measured by training RSS
#   for classification: mean impurity is measured by deviance
randomForest::importance(bag_boston)


randomForest::varImpPlot(bag_boston)


# For a regression problem (see classification below) the MSE 
bag_boston[["mse"]]
graphics::hist(bag_boston[["mse"]])

randomForest:::plot.randomForest(bag_boston)


# Test error rate
yhat_bag <- stats::predict(bag_boston, newdata = boston[-train, ])

base::plot(yhat_bag, boston[-train, ][["medv"]])
graphics::abline(0, 1)


bag_mse <- base::mean((yhat_bag - boston[-train, ][["medv"]])^2)


# Random Forest -----------------------------------------------------------


# By default randomForest() uses mtry = p / 3 for regression and sqrt(p) for classification
rf_boston <- randomForest::randomForest(
  medv ~ .,
  data = boston,
  subset = train,
  replace = TRUE,
  importance = TRUE,
  ntree = 100L,
  nPerm = 1L
)


# Variable importance
randomForest::varImpPlot(rf_boston)

vip::vip(rf_boston)

randomForest:::plot.randomForest(rf_boston)


# Predictions and error
yhat_rf <- stats::predict(rf_boston, newdata = boston[-train, ])
rf_mse <- base::mean((yhat_rf - boston[-train, ][["medv"]])^2)

rf_mse / bag_mse


# Classification ----------------------------------------------------------


# How about classification problems?

# GlaucomaM is a dataset with 196 patients of eye health related variables (62 of them) and a Class variable
# that takes the value "glaucoma" if the patient has glaucoma or "normal" if it does not.

# glaucoma (a neuro-degenerative disease of the optic nerve) or not.
# measurements are derived from laser scanning images
glaucoma <- TH.data::GlaucomaM %>% tibble::as_tibble()

base::table(glaucoma[["Class"]])

base::set.seed(1L)
train <- base::sample(1L:base::nrow(glaucoma), .75 * base::nrow(glaucoma))

base::length(train)
base::dim(glaucoma)[1] - base::length(train)


# randomForest training
rf_glaucoma <- randomForest::randomForest(
  Class ~ .,
  data = glaucoma,
  subset = train,
  importance = TRUE,
  ntree = 100L
)


# A ntrees x (nclass + 1) matrix of error rates of the prediction of the input data
rf_glaucoma[["err.rate"]][1:10,]
base::dim(rf_glaucoma[["err.rate"]])

# Confusion matrix
rf_glaucoma[["confusion"]]

# A matrix of base::length(train) rows and two nclass columns with OOB votes for each observation
rf_glaucoma[["votes"]][1:10,]
base::dim(rf_glaucoma[["votes"]])

# MeanDecreaseAccuracy: loss in prediction performance when that particular variable is omitted from the training set
# MeanDecreaseGini: node purity decrease when splitting over a particular variable
randomForest::varImpPlot(rf_glaucoma)

randomForest:::plot.randomForest(rf_glaucoma)


# Test error
confusion_matrix <- base::table(
  obs = glaucoma[-train, ][["Class"]],
  pred = stats::predict(rf_glaucoma, newdata = glaucoma[-train,])
)

confusion_matrix

(confusion_matrix[1L, 2L] + confusion_matrix[2L, 1L]) / base::sum(confusion_matrix)


# ipred -------------------------------------------------------------------


# An other library for bagging is ipred

kyphosis <- rpart::kyphosis %>% tibble::as_tibble()


# Obs by class
base::table(kyphosis[["Kyphosis"]])


# 79% of obs don't have kyphosis, 21% do
# We need to sample so that prop is kept
kyphosis %>% 
  dplyr::group_by(
    Kyphosis
  ) %>% 
  dplyr::tally() %>% 
  dplyr::mutate(
    prop = n / base::sum(n)
  )


# Select 75% of obs in each group
grouped_sample <- kyphosis %>% 
  dplyr::mutate(
    row_number = dplyr::row_number()
  ) %>% 
  dplyr::group_by(
    Kyphosis
  ) %>% 
  dplyr::slice_sample(
    prop = .75
  )

grouped_sample %>% 
  dplyr::tally() %>% 
  dplyr::mutate(
    prop = n / base::sum(n)
  )

train <- dplyr::pull(grouped_sample, row_number)

base::length(train)

bag_kyphosis <- ipred::bagging(
  Kyphosis ~ Age + Number + Start,
  data = kyphosis,
  subset = train,
  nbagg = 200L,
  coob = TRUE
)


# Fitted trees
base::class(bag_kyphosis[["mtrees"]])
base::length(bag_kyphosis[["mtrees"]])


base::class(bag_kyphosis[["mtrees"]][[1]])
base::length(bag_kyphosis[["mtrees"]][[1]])


bag_kyphosis[["mtrees"]][[1]][["btree"]]


## OOB misclassification  error
bag_kyphosis[["err"]]


## Test predictions
stats::predict(bag_kyphosis, newdata = kyphosis[-train, ])


confusion_matrix <- base::table(
  obs = kyphosis[-train, ][["Kyphosis"]],
  pred = stats::predict(bag_kyphosis, newdata = kyphosis[-train, ])
)

confusion_matrix


## Test error0
(confusion_matrix[1L, 2L] + confusion_matrix[2L, 1L]) / base::sum(confusion_matrix)


#===============#
#### THE END ####
#===============#