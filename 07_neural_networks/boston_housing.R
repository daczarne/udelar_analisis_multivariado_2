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


c(train_data, train_labels) %<-% boston_housing[["train"]]
c(test_data, test_labels) %<-% boston_housing[["test"]]


column_names <- base::names(MASS::Boston)
column_names <- column_names[-base::length(column_names)]


train_df <- train_data %>% 
  tibble::as_tibble(
    .name_repair = "minimal"
  ) %>% 
  stats::setNames(
    column_names
  ) %>% 
  dplyr::mutate(
    label = train_labels
  )


test_df <- test_data %>% 
  tibble::as_tibble(
    .name_repair = "minimal"
  ) %>% 
  stats::setNames(
    column_names
  ) %>% 
  dplyr::mutate(
    label = test_labels
  )


# We standardize all features to have mean 0 and sd 1
spec <- tfdatasets::feature_spec(
  train_df,
  label ~ . 
) %>% 
  tfdatasets::step_numeric_column(
    tfdatasets::all_numeric(),
    normalizer_fn = tfdatasets::scaler_standard()
  ) %>% 
  keras::fit()

spec

base::class(spec)


# We pre-process the (dense) layer features in the TensorFlow graph
# The result is a 2-dim tensor (ie: a matrix)
layer <- keras::layer_dense_features(
  feature_columns = tfdatasets::dense_features(spec), 
  dtype = tf[["float32"]]
)

layer(train_df)


# Model building ----------------------------------------------------------


# We define the input layer as a Python dictionary
input <- tfdatasets::layer_input_from_dataset(train_df %>% dplyr::select(-label))

output <- input %>% 
  keras::layer_dense_features(
    tfdatasets::dense_features(spec)
  ) %>% 
  keras::layer_dense(
    units = 64,
    activation = "relu"
  ) %>%
  keras::layer_dense(
    units = 64,
    activation = "relu"
  ) %>%
  keras::layer_dense(
    units = 1
  ) 

boston_housing_model <- keras::keras_model(input, output)

base::summary(boston_housing_model)


boston_housing_model %>% 
  keras::compile(
    loss = "mse",
    optimizer = keras::optimizer_rmsprop(),
    metrics = base::list("mean_absolute_error")
  )


# Model training ----------------------------------------------------------


boston_housing_fit <- boston_housing_model %>%
  keras::fit(
  x = dplyr::select(train_df, -label),
  y = train_df[["label"]],
  epochs = 100,
  validation_split = 0.2,
  callbacks = base::list(
    keras::callback_early_stopping(
      monitor = "val_loss",
      patience = 20
    )
  ),
  verbose = 2
)


# Prediction --------------------------------------------------------------

c(loss, mae) %<-% (keras::evaluate(boston_housing_model, dplyr::select(test_df, -label), test_df[["label"]], verbose = 2))

base::sprintf("%.2f", mae * 1000)

test_predict <- base::as.numeric(stats::predict(boston_housing_model, dplyr::select(test_df, -label)))

base::sum((test_predict - test_df[["label"]])^2)


#===============#
#### THE END ####
#===============#