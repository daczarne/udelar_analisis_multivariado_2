#============================#
#### CLASSIFICATION TREES ####
#============================#


library(magrittr)
library(tree)


# Fitting a classification tree --------------------------------------------

carseats <- ISLR::Carseats %>% tibble::as_tibble()


## Build binary variable
carseats %<>% 
  tibble::as_tibble() %>% 
  dplyr::mutate(
    high = forcats::as_factor(dplyr::if_else(carseats[["Sales"]] < 8L, "No", "Yes"))
  )


## Fit tree to predict sales using all features except for Sales
carseats_tree <- tree::tree(
  formula = high ~ . - Sales,
  data = carseats
)

base::summary(carseats_tree)

# For classification trees the deviance reported by summary() is given by
# -2 sum_m sum_k n_{mk} log(hat{p}_{mk})
# where n_{mk} is the number of observations in the m-th terminal node that belongs to the k-th class
# A small deviance indicates a tree that provides a good fit to the training data

# The residual mean deviance is the deviance divided by n - |T_0|


## Plotting a tree
base::plot(carseats_tree)
graphics::text(carseats_tree, pretty = 0L)

# Shelve location is the most important indicator of Sales

## Printing a tree
carseats_tree

# Printing the tree object shows a tree like date structure with the following data
#   split: the split criterion. A range for continuous variables, a factor level for categorical variables
#   n: the number of observations that are left to the left of the tree branch
#   deviance: 
#   yval: overall prediction for the branch
#   yprob: proportion of observations of each factor level for the observations in the branch
#   an asterisk denotes a terminal node


# Predictions -------------------------------------------------------------


# In order to properly assess the model error, we need to compute the test error. For that we need to split the data into a training
# and a testing set.

# Select training observations (set seed so that results are reproducible)
base::set.seed(2L)
train <- base::sample(1L:base::nrow(carseats), 200L)

# Build test data by eliminating training data
carseats_test <- carseats[-train, ]

# Re-train the model
carseats_tree <- tree::tree(
  formula = high ~ . -Sales,
  data = carseats,
  subset = train
)

base::summary(carseats_tree)

## Plotting the tree
base::plot(carseats_tree)
graphics::text(carseats_tree, pretty = 0L)

## Printing the tree
tree:::print.tree(carseats_tree)

## Predictions
carseats_pred <- stats::predict(carseats_tree, carseats_test, type = "class")

stats::xtabs(~carseats_pred + carseats_test[["high"]])

1L - (53L + 103L) / 200L


# Pruning -----------------------------------------------------------------


## We use cross-validation to determine the optimal level of tree complexity
## Use FUN = prune.misclass to indicate that classification error should guide the cross-validation process (default is deviance)

base::set.seed(2L)
carseats_cv <- tree::cv.tree(
  object = carseats_tree,
  FUN = prune.misclass
)

## Object contains
# size: number of terminal nodes of each tree considered
# dev: error rate
# k: cost-complexity parameter
carseats_cv

# The tree with 19 terminal nodes results in the lowest cross-validated error rate


## Plot
graphics::par(mfrow = base::c(1L, 2L))
base::plot(carseats_cv[["size"]], carseats_cv[["dev"]], type = "b")
base::plot(carseats_cv[["k"]], carseats_cv[["dev"]], type = "b")


# We now use prune.misclass() to prune the tree
carseats_pruned <- tree::prune.misclass(
  tree = carseats_tree,
  best = 5L
)

# and we plot the pruned tree
base::plot(carseats_pruned)
graphics::text(carseats_pruned, pretty = 0L)

# WE now use the test set to compute the error rate
carseats_pred <- stats::predict(carseats_pruned, carseats_test, type = "class")

stats::xtabs(~carseats_pred + carseats_test[["high"]])

1L - (67L + 82L) / 200L


#===============#
#### THE END ####
#===============#