#=======================#
#### NEURAL NETWORKS ####
#=======================#


library(magrittr)
library(keras)
library(tensorflow)


# keras -------------------------------------------------------------------


# We are going to use the MNIST data set to build a Feedforward DNN. This dataset consists of 120.000 images of handwritten digits.
# Each picture is a 28 x 28 pixel (784 pixels in total) and it's black and white only
# Since the digits are black and white, each picture can be broken down to a vector of length 784 where each value corresponds to
# the gray-scale value of that pixel (0: white, 255: black).
# This data set has already been split into 60.000 pics for training and 60.000 pics for testing

#  The goal is to predict the hand-written digit (0 to 9)

# A list of length 2
#   training: a list of length 2
#     images: a matrix of 60.000 x 784. Each row corresponds to an image, each col to a pixel value
#     labels: an integer vector of length 60.000 with the labels for each image
#   test: a list of length 2
#     images: a matrix of 60.000 x 784. Each row corresponds to an image, each col to a pixel value
#     labels: an integer vector of length 60.000 with the labels for each image
mnist <- dslabs::read_mnist()

# keras requires two separate objects as arguments, a matrix of covariates (X) and a one-hot matrix of responses (y) 
mnist_x <- mnist[["train"]][["images"]]
mnist_y <- mnist[["train"]][["labels"]]


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
mnist_model <- keras::keras_model_sequential() %>% 
  
  ## Network architecture
  keras::layer_dense(
    units = 128,
    input_shape = base::ncol(mnist_x),
    activation = "relu"
  ) %>% 
  keras::layer_dense(
    units = 64,
    activation = "relu"
  ) %>% 
  keras::layer_dense(
    units = 10,
    activation = "softmax"
  ) %>% 
  
  ## Backpropagation
  keras::compile(
    loss = "categorical_crossentropy",
    optimizer = keras::optimizer_rmsprop(),
    metrics = base::c("accuracy")
  )

mnist_model

mnist_fit1 <- mnist_model %>%
  keras::fit(
    x = mnist_x,
    y = mnist_y,
    epochs = 25,
    batch_size = 128,
    validation_split = 0.2,
    verbose = FALSE
  )


mnist_fit1

plot(mnist_fit1)

#===============#
#### THE END ####
#===============#