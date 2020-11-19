#========================#
#### REGRESSION TREES ####
#========================#


library(magrittr)
library(tree)


# Fitting a regression tree -----------------------------------------------

# We fit a regression tree to the Boston data set
boston <- MASS::Boston %>% tibble::as_tibble()


# Split testing and training set
base::set.seed(1L)
train <- base::sample(1L:base::nrow(boston), base::nrow(boston) / 2L)

boston_tree <- tree::tree(
  formula = medv ~ .,
  data = boston,
  subset = train
)

base::summary(boston_tree)

# Plotting the tree
base::plot(boston_tree)
graphics::text(boston_tree, pretty = 0L)


# Pruning the tree --------------------------------------------------------
boston_tree_cv <- tree::cv.tree(
  object = boston_tree
)

base::plot(boston_tree_cv[["size"]], boston_tree_cv[["dev"]], type = "b")

# Results point at using a complex tree. Still we can use the prune.tree() function to prune the tree
boston_tree_pruned <- tree::prune.tree(
  tree = boston_tree,
  best = 5L
)

base::plot(boston_tree_pruned)
graphics::text(boston_tree_pruned, pretty = 0L)


# Predictions -------------------------------------------------------------

# We use the un-pruned tree to make predictions

yhat <- stats::predict(boston_tree, newdata = boston[-train, ])
y <- boston[-train, ][["medv"]]

base::plot(yhat, y)
graphics::abline(0L, 1L)


# Test MSE
mse <- base::mean((yhat - y)^2L)

# The model makes predictions that are within around $6k of the true median value for that suburb
base::sqrt(mse)


#===============#
#### THE END ####
#===============#