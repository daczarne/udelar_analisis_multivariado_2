#====================#
#### IMDB Reviews ####
#====================#


library(magrittr)
library(keras)


# Fetch data --------------------------------------------------------------


pins::board_register_kaggle(token = "07_neural_networks/kaggle.json")

paths <- pins::pin_get("nltkdata/movie-review", "kaggle")
path <- paths[1]


# The IMDB data set contains 64k reviews
# text contains the actual review
# tag contains the classification (pos/neg)
imdb_data <- readr::read_csv(path)

# About half of the reviews are positive and half negative
imdb_data %>% dplyr::count(tag)


# We'll use 80% of the set for training and 20% for testing
training_id <- base::sample.int(base::nrow(imdb_data), size = base::nrow(imdb_data) * 0.8)
training <- imdb_data[training_id,]
testing <- imdb_data[-training_id,]


# Feature engineering -----------------------------------------------------


# Before we can feed the reviews into a network, we need to transform the reviews into tensors.
# We'll represent each of the 10.000 most common words by an integer. Every review will be represented by a sequence of integers
# Then we will pad the arrays (the tensors) so that they all have the same length
# Next will build an integer tensor of shape 10.000 x 50 (where 50 is the maximum length that we'll permit)
# We'll use an embedding layer that can handle this type of data as the first layer of our NN

number_of_words <- 10000
max_length <- 50

# max_tokens: maximum size of the vocabulary for the layer
# output_sequence_length: length of the output sequence (tensor shape will be batch_size x output_sequence_length)
text_vectorization <- keras::layer_text_vectorization(
  max_tokens = number_of_words,
  output_sequence_length = max_length,
)


text_vectorization %>% 
  keras::adapt(
    imdb_data[["text"]]
  )


## We can see the actual words with the get_vocabulary function
keras::get_vocabulary(text_vectorization)

base::class(keras::get_vocabulary(text_vectorization))
base::length(keras::get_vocabulary(text_vectorization))


# As an example, lets look at the first review.
# This review is 40 words long. Each word gets replaced by its corresponding integer value.
# 10 extra 0s are added after word number 40
text_vectorization(base::matrix(imdb_data[["text"]][1], ncol = 1))


# Should we choose output_sequence_length to be greater than 50??
imdb_data[["text"]] %>% 
  base::strsplit(" ") %>% 
  base::sapply(length) %>% 
  plotly::plot_ly(
    x = .,
    type = "box",
    name = "number of words",
    hoverinfo = "x"
  )


# Model building ----------------------------------------------------------


# To sum-up: this model consists of an input layer made out of the indices for each word in the review and 
# a output of 0 or 1

# Our first layer will generate an embedding
# Embedding is the concept of mapping from discrete objects (such as words in our case) to vectors (or real numbers)

# The global_average_pooling_1d layer returns a fixed-length output vector for each review
# This is achieved by averaging over the sequence dimension
# It allows the model to handle inputs of variable length

# Next comes a dense layer of 16 neurons with ReLU activation
# We add a layer dropout here
# A dropout selects a random fraction of the input units to 0 at each update
# There's no mathematical proof (at least that I know off) that this should work, but it does (a random forest-like situation)

# The output layer is fully connected with 1 output layer (class predictions) with a sigmoid activation

input <- keras::layer_input(shape = base::c(1), dtype = "string")

output <- input %>% 
  text_vectorization() %>% 
  keras::layer_embedding(
    input_dim = number_of_words + 1,
    output_dim = 16
  ) %>%
  keras::layer_global_average_pooling_1d() %>%
  keras::layer_dense(
    units = 16,
    activation = "relu"
  ) %>%
  keras::layer_dropout(
    rate = 0.5
  ) %>% 
  keras::layer_dense(
    units = 1,
    activation = "sigmoid"
  )

model <- keras::keras_model(input, output)

model %>% 
  keras::compile(
  optimizer = 'adam',
  loss = 'binary_crossentropy',
  metrics = base::list('accuracy')
)


# Model training ----------------------------------------------------------


history <- model %>%
  keras::fit(
  training[["text"]],
  base::as.numeric(training[["tag"]] == "pos"),
  epochs = 10,
  batch_size = 512,
  validation_split = 0.2,
  verbose = 2
)


# Evaluate ----------------------------------------------------------------


keras:::plot.keras_training_history(history)


results <- model %>% keras::evaluate(testing[["text"]], base::as.numeric(testing[["tag"]] == "pos"), verbose = 1)
results

confusion_matrix <- base::table(
  obs = testing[["tag"]],
  pred = dplyr::if_else(stats::predict(model, testing[["text"]]) > 0.5, "pos", "neg")
)

base::sum(base::diag(confusion_matrix)) / base::sum(confusion_matrix)


#===============#
#### THE END ####
#===============#