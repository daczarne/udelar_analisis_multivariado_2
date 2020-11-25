#=======================#
#### MNIST BLUEPRINT ####
#=======================#


# Hyperparameter flags ---------------------------------------------------


FLAGS <- keras::flags(
  keras::flag_numeric("nodes_layer_1", 256, "Nodes in layer 1"),
  keras::flag_numeric("nodes_layer_2", 128, "Nodes in layer 2"),
  keras::flag_string("optimizer", "rmsprop", "Optimizer")
)


# Network architecture ----------------------------------------------------


model <- keras::keras_model_sequential() %>% 

  keras::layer_dense(
    units = FLAGS$nodes_layer_1,
    activation = "relu",
    input_shape = base::ncol(x = mnist_x)
  ) %>%
  keras::layer_batch_normalization() %>%
  keras::layer_dropout(
    rate = 0.3
  ) %>%
  keras::layer_dense(
    units = FLAGS$nodes_layer_2,
    activation = "relu"
  ) %>%
  keras::layer_batch_normalization() %>%
  keras::layer_dropout(
    rate = 0.3
  ) %>%
  keras::layer_dense(
    units = 10,
    activation = "softmax"
  ) %>% 
  keras::compile(
    loss = "categorical_crossentropy",
    optimizer = FLAGS$optimizer,
    metrics = base::c("accuracy")
  ) %>% 
  keras::fit(
    x = mnist_x,
    y = mnist_y,
    batch_size = 128,
    epochs = 20,
    validation_split = 0.2,
    verbose = 2
  )


#===============#
#### THE END ####
#===============#