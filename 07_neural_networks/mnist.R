#=============#
#### MNIST ####
#=============#


library(magrittr)
library(keras)
library(tensorflow)


# Feature engineering -----------------------------------------------------


# We are going to use the MNIST data set to build a Feedforward DNN. This dataset consists of 70.000 images of handwritten digits.
# Each picture is a 28 x 28 pixel (784 pixels in total) and it's black and white only
# Since the digits are black and white, each picture can be broken down to a vector of length 784 where each value corresponds to
# the gray-scale value of that pixel (0: white, 255: black).
# This data set has already been split into 60.000 pics for training and 10.000 pics for testing

#  The goal is to predict the hand-written digit (0 to 9)

# A list of length 2
#   training: a list of length 2
#     images: a matrix of 60.000 x 784. Each row corresponds to an image, each col to a pixel value
#     labels: an integer vector of length 60.000 with the labels for each image
#   test: a list of length 2
#     images: a matrix of 10.000 x 784. Each row corresponds to an image, each col to a pixel value
#     labels: an integer vector of length 10.000 with the labels for each image
mnist <- dslabs::read_mnist()

# keras requires two separate objects as arguments, a matrix of covariates (X) and a one-hot matrix of responses (y) 
mnist_x <- mnist[["train"]][["images"]]
mnist_y <- mnist[["train"]][["labels"]]


base::class(mnist_x)
base::dim(mnist_x)


base::class(mnist_y)
base::length(mnist_y)


# Rename columns and standardize feature values
base::colnames(mnist_x) <- base::paste0("V", 1:base::ncol(mnist_x))


# Due to the nature of how DNNs work, all features need to be standardized
mnist_x <- mnist_x / (base::max(mnist_x) - base::min(mnist_x))


# One-hot encode response
# A one-hot representation of data is a sequence of bits in which only one of those bits takes on the value 1 and the rest take on the value 0
# Similarly, in one-cold, only one bit is 0 and the rest are 1
# Since we have 10 possible digits (0-9), we'll use a n x 10 matrix to represent our data
mnist_y <- keras::to_categorical(
  y = mnist_y,
  num_classes = 10,
  dtype = "float32"
)


# Model building ----------------------------------------------------------


# Model:
#   i. first we initialize a sequential model
#   ii. we define two hidden (dense) layers. One with 128 nodes and one with 64
#   iii. we define the output layer with 10 nodes (one for each class)
# The input_shape specifies how many features the first hidden layer is going to receive. This needs to be the same as features in our input matrix.
# We use dense layers since we want each neuron (node) to be connected to every node in the previous layer.

# Node activation:
#   We use the activation argument of keras::layer_dense() to specify the activation function for that layer
#   When using rectangular data it is common practice to use a ReLU for hidden layers
#   Output layer usually use linear for regression, sigmoid (logistic) for binary, or softmax for categorical

# Linear: f(x) = x
# ReLU: f(x) = max(0, x)
# Sigmoid: f(x) = 1 / (1 + e^{-x})
# Softmax: f(x) = e^{x} / sum(e^{x})

# Backpropagation:
# loss: loss function to measure performance
#   regression: MSE
#   classification: categorical cross entropy: CE = - log( e^s / sum(e^s) )
# optimizer: controls how the backpropagation is implemented.
# Usually rmsprop is used for mini-batch SGD. rmsprop stands for Root Mean Square Propagation
#   lr: learning rate
# other optimizers are available: adadelta, adagrad, adamax, sgd, etc
# metrics: list of metrics to be measured and used to evaluate the model
# kernel_regularizer: layer parameter regularization (conceptualy the same as with any other ML model). Can be L1, L2, or both
mnist_model <- keras::keras_model_sequential() %>% 
  
  ## Network architecture
  keras::layer_dense(
    units = 256,
    input_shape = base::ncol(x = mnist_x),
    activation = "relu",
    kernel_regularizer = keras::regularizer_l2(l = 0.001)
  ) %>% 
  keras::layer_batch_normalization() %>% # Re-normalize data after layer
  keras::layer_dense(
    units = 128,
    activation = "relu",
    kernel_regularizer = keras::regularizer_l2(l = 0.001)
  ) %>% 
  keras::layer_batch_normalization() %>% # Re-normalize data after layer
  keras::layer_dense(
    units = 64,
    activation = "relu",
    kernel_regularizer = keras::regularizer_l2(l = 0.001)
  ) %>% 
  keras::layer_batch_normalization() %>% # Re-normalize data after layer
  keras::layer_dense(
    units = 10,
    activation = "softmax"
  ) %>% 
  
  ## Backpropagation
  keras::compile(
    loss = "categorical_crossentropy",
    optimizer = keras::optimizer_adam(
      lr = 0.001
    ),
    metrics = base::c("accuracy")
  )


mnist_model


# Model traning -----------------------------------------------------------


# We use the fit function to train our model
# x and y specify the features and response of our model respectively
# batch_size: the number of observations to run through the mini-batch SGD process
# epochs: data is fed one batch at a time. An epoch is complete when the training algorithm has seen all data.
#   The epochs arguments controls how many times the fitting process must see all data
# validation_split: the proportion of the data to be withheld to estimate the out-of-sample error
# callbacks: a list of model callbacks (ie, a way of adjusting parameters while the model is training)
#     callback_early_stopping: will stop the training if after 5 epochs there's no loss improvement
#     callback_reduce_lr_on_plateau: will reduce the optimizers learning rate if a plateau is reached
keras::tensorboard(
  log_dir = "07_neural_networks/logs/",
  action = base::c("start"),
  host = "127.0.0.1",
  launch_browser = TRUE,
  reload_interval = 1
)

mnist_fit <- mnist_model %>%
  keras::fit(
    x = mnist_x,
    y = mnist_y,
    batch_size = 128,
    epochs = 25,
    validation_split = 0.2,
    callbacks = base::list(
      keras::callback_early_stopping(
        patience = 5
      ),
      keras::callback_reduce_lr_on_plateau(
        factor = 0.05
      ),
      keras::callback_tensorboard(
        log_dir = "07_neural_networks/logs/"
      )
    ),
    verbose = 0
  )


# An object of class keras training history (a list of length 2)
base::length(mnist_fit)


# A list with model metrics 
mnist_fit[["metrics"]]


# A plot displaying loss and accuracy for training and validation
mnist_training_history <- tibble::tibble(
  epoch = 1:base::length(mnist_fit[["metrics"]][["loss"]]),
  loss = mnist_fit[["metrics"]][["loss"]],
  accuracy = mnist_fit[["metrics"]][["accuracy"]],
  val_loss = mnist_fit[["metrics"]][["val_loss"]],
  val_accuracy = mnist_fit[["metrics"]][["val_accuracy"]],
  lr = mnist_fit[["metrics"]][["lr"]]
)


mnist_training_history %>%
  tidyr::pivot_longer(
    cols = -epoch,
    names_to = "variable"
  ) %>% 
  dplyr::mutate(
    var_type = dplyr::case_when(
      variable %in% base::c("loss", "accuracy") ~ "training",
      variable %in% base::c("val_loss", "val_accuracy") ~ "validation",
      variable == "lr" ~ "lr"
    ),
    plot_type = dplyr::case_when(
      variable %in% base::c("val_accuracy", "accuracy") ~ "accuracy",
      variable %in% base::c("val_loss", "loss") ~ "loss",
      variable == "lr" ~ "lr"
    )
  ) %>% 
  ggplot2::ggplot(
    ggplot2::aes(
      x = epoch,
      y  = value,
      color = var_type
    )
  ) + 
  ggplot2::geom_point(
    alpha = 0.3
  ) +
  ggplot2::geom_line(
    alpha =  0.3
  ) +
  ggplot2::geom_smooth(
    se = FALSE
  ) +
  ggplot2::facet_wrap(
    ~plot_type,
    ncol = 1L,
    scales = "free"
  ) + 
  ggplot2::labs(
    y = NULL,
    color = NULL
  )


# Prediction --------------------------------------------------------------


predicted_labels <- keras::predict_classes(
  object = mnist_model,
  x = mnist[["test"]][["images"]] / 255
)


confusion_matrix <- base::table(
  obs = mnist[["test"]][["labels"]],
  pred = predicted_labels
)


confusion_matrix


base::sum(base::diag(confusion_matrix)) / base::sum(confusion_matrix)


#===============#
#### THE END ####
#===============#