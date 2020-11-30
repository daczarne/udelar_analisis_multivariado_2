#======================#
#### MODEL BUILDING ####
#======================#


library(magrittr)


# Data split --------------------------------------------------------------


train <- base::sample(base::nrow(iris), 0.8 * base::nrow(iris))
iris_train <- iris[train, ]
iris_test <- iris[-train, ]


# Model training ----------------------------------------------------------

iris_rf <- ranger::ranger(
  formula = Species ~ .,
  data = iris_train,
  num.trees = 500, # Number of trees to build
  mtry = 2 # Number of variables to possibly split at in each node
)


# Model reduction ---------------------------------------------------------


# In order to deploy a model into production, we need to strip it of everything that its "unnecessary"
# This can be challenging to do since, though from a statistical point of view we might think that something
# is not important, it might be user in the predict method


# For example, if we strip the importance.mode, prediction() will not work
iris_rf_no_im <- iris_rf
iris_rf_no_im[["importance.mode"]] <- NULL

stats::predict(iris_rf_no_im, data = iris_test)


# The butcher package provides a method for doing exactly this

## butcher::butcher() reduces the model object to its minimum prediction requirements
iris_rf_axed <- butcher::butcher(iris_rf, verbose = TRUE)


lobstr::obj_size(iris_rf)
lobstr::obj_size(iris_rf_axed)


# butcher does not eliminate the slots, it just empties them
base::names(iris_rf)
base::names(iris_rf_axed)

iris_rf[["predictions"]]
iris_rf_axed[["predictions"]]

iris_rf[["call"]]
iris_rf_axed[["call"]]


# Our model can still call the predict method
iris_pred <- stats::predict(iris_rf, data = iris_test)
base::table(iris_test[["Species"]], iris_pred[["predictions"]])


iris_pred_axed <- stats::predict(iris_rf_axed, data = iris_test)
base::table(iris_test[["Species"]], iris_pred_axed[["predictions"]])


# Save the model object ---------------------------------------------------
base::saveRDS(object = iris_rf_axed, file = "08_model_sharing/iris_api/iris_model.RDS")


#===============#
#### THE END ####
#===============#