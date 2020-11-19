#================#
#### BOOSTING ####
#================#


library(magrittr)


# AdaBag ------------------------------------------------------------------


# Weighted re-sampling based on miss classification errors


## Iris

# formula: an lm-style formula expression
# data: a data-set (there is no subset argument)
# mfinal: number of iterations
# coeflearn: one of
#     i. "Breiman" for Adaboost.M1 algorithm with alpha = 1 / 2 * ln((1 - err) / err) 
#    ii. "Freund" for Adaboost.M1 algorithm with alpha = ln((1 - err) / err)
#   iii. "Zhu" for SAMME algorithm with alpha = ln((1 - err) / err) + ln(nclasses - 1)
iris_boost <- adabag::boosting(
  formula = Species ~ .,
  data = iris,
  mfinal = 4L,
  coeflearn = "Zhu"
)


# A list of the trained trees
iris_boost[["trees"]]


# A vector of the same length as mfinal with tree weights
iris_boost[["weights"]]


# Matrix of n x nclass. Each value corresponds to the weighted sum of the number of trees
# that assigned each obs to the class in that column
iris_boost[["votes"]][1:10,]

base::dim(iris_boost[["votes"]])

# If an obs was always assigned to the same class, then it's value in the votes matrix should
# be equal to the sum of weights
base::sum(iris_boost[["weights"]])


# Matrix of n x nclass. Each value corresponds to the proportion of votes for that obs on that
# class in the final ensemble. If an obs was always classified as belonging to a specific class
# then prob is equal 1
iris_boost[["prob"]][1:10,]

iris_boost[["prob"]][100:110,]


# A vector of length n of class assigned by the ensemble
iris_boost[["class"]]

base::table(
  obs = iris[["Species"]],
  pred = iris_boost[["class"]]
)


# Relative variable importance (based on the Gini for a tree and the weight of the tree)
iris_boost[["importance"]]


# Error evolution (as the ensemble grows)
adabag::errorevol(iris_boost, newdata = iris)

adabag::plot.errorevol(adabag::errorevol(iris_boost, newdata = iris))


## Iris CV

# boosting.cv implements v-fold cross-validation
# boos = TRUE means that bootstrap samples of the training set are used for each tree
# v: integer between 2 and base::dim(data)[1] indicating the number of cross-validation groups (v-1 are user
#   for training and the v-th for testing). If v = base::dim(data)[1], the LOOCV is implemented.
# par = TRUE means parallelization will be used as defined in the doParallel package
#   default is FALSE
#   if par is set to TRUE
#     i. rpart.control must be specified
#     ii. beware of memory usage
iris_boost_cv <- adabag::boosting.cv(
  Species ~ .,
  data = iris,
  boos = TRUE,
  v = 10L,
  mfinal = 4L,
  coeflearn = "Zhu",
  control = rpart::rpart.control(
    cp = 0.01
  ),
  par = TRUE
)


# Ensemble class prediction
iris_boost_cv[["class"]]

# Ensemble class prediction
iris_boost_cv[["confusion"]]

base::table(
  obs = iris[["Species"]],
  pred = iris_boost_cv[["class"]]
)

# Average MSE for the ensemble model
iris_boost_cv[["error"]]


## Breast Cancer
breast_cancer <- readr::read_csv(
  file = "http://archive.ics.uci.edu/ml/machine-learning-databases/breast-cancer-wisconsin/breast-cancer-wisconsin.data",
  col_names = base::c(
    "id",
    "clump_thickness",
    "uniformity_of_cell_size",
    "uniformity_of_cell_shape",
    "marginal_adhesion",
    "single_epithelial_cell_size",
    "bare_nuclei",
    "bland_chromatin",
    "normal_nucleoli",
    "mitoses",
    "class"
  ),
  col_types = stringr::str_c("c", stringr::str_flatten(base::rep("i", 10L))),
  na = "?"
) %>% 
  dplyr::mutate(
    class = base::factor(
      x = class,
      levels = base::c(2L, 4L),
      labels = base::c("benign", "malignant")
    )
  ) %>% 
  dplyr::select(
    -id
  ) %>% 
  dplyr::relocate(
    class,
    .before = 1L
  )


breast_cancer_boost <- adabag::boosting(
  formula = class ~ .,
  data = breast_cancer,
  mfinal = 100L,
  coeflearn = "Breiman"
)


# GBM ---------------------------------------------------------------------

# In boosting each tree is grown using information from previously grown trees
# Each tree is built on a modified version of the original data set
# Boosting is based on "slow learners".
#   Given the current model, a decision tree is fit to the model residuals
#   The new decision tree is added to the original and the residuals are updated


# Boosting involves 3 parameters
#   B: the number of trees to be trained
#   d: the number of terminal nodes in each subsequent tree
#   lambda: a shrinkage parameter greater than 0 that controls the rate at which the boosting learns


boston <- MASS::Boston %>% tibble::as_tibble()

base::set.seed(1L)
train <- base::sample(1L:base::nrow(boston), .75 * base::nrow(boston))


# formula: a formula specifying the model
# data: a data.frame containing the data for training. gbm has no subset argument, so only training data must be supplied.
# distribution: character specifying the name of the distribution. Multiple options are available. Most notably:
#   gaussina: regression problems
#   bernoulli: two-class classification problems
#   multinomial: multi-class classification problems
#   poisson: count data
# n.trees: number of trees (B). Default is 100
# interaction.depth: number of terminal nodes (d). Default is 1 (stump tree)
# shrinkage: shrinkage parameter (lambda). Default is 0.1
boston_boost <- gbm::gbm(
  formula = medv ~ .,
  data = boston[train,],
  distribution = "gaussian",
  n.trees = 5000L,
  interaction.depth = 4L,
  shrinkage = 0.1,
  cv.folds = 10L
)

# Variable importance
base::summary(boston_boost)


# A vector of the same length as the number of trees, each slot containing the train error for that tree
boston_boost[["train.error"]]


# Minimum cv error
boston_boost[["cv.error"]][base::which.min(boston_boost[["cv.error"]])]


# Evolution of the training error and the cv error as the number of trees gets larger
base::plot(1:5000, boston_boost[["train.error"]], type = "l", col = "dodgerblue")
graphics::lines(1:5000, boston_boost[["cv.error"]], col = "red")
graphics::abline(v = base::which.min(boston_boost[["cv.error"]]))


gbm::gbm.perf(boston_boost, method = "cv")


# Predictions
yhat <- stats::predict(boston_boost, newdata = boston[-train, ], n.trees = 5000L)

base::mean((yhat - boston[-train, ][["medv"]])^2L)


#===============#
#### THE END ####
#===============#