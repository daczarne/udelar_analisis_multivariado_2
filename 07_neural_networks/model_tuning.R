#=====================#
#### MODOEL TUNING ####
#=====================#


library(keras)
library(tfruns)


# Feature engineering -----------------------------------------------------


mnist <- dslabs::read_mnist()
mnist_x <- mnist[["train"]][["images"]]
mnist_y <- mnist[["train"]][["labels"]]

base::colnames(mnist_x) <- base::paste0("V", 1:base::ncol(mnist_x))

mnist_x <- mnist_x / (base::max(mnist_x) - base::min(mnist_x))

mnist_y <- keras::to_categorical(
  y = mnist_y,
  num_classes = 10,
  dtype = "float32"
)


# Run model ---------------------------------------------------------------


model_runs <- tfruns::tuning_run(
  file = "07_neural_networks/model_blueprint.R", 
  flags = base::list(
    nodes_layer_1 = base::c(128, 256),
    optimizer = base::c("rmsprop", "adam")
  ),
  confirm = FALSE,
  runs_dir = "07_neural_networks/runs"
)


#===============#
#### THE END ####
#===============#