#=================#
#### PR6 - EJ1 ####
#=================#

# Con los datos iris y usando la librería tree

library(magrittr)


# a) ----------------------------------------------------------------------

## Construya un árbol de clasificación (que llamaremos arbol) con Species como variable de respuesta.
## Utilice la función summary() para describir los resultados obtenidos.

iris <- datasets::iris %>%
  tibble::as_tibble() %>% 
  dplyr::rename_with(
    .fn = snakecase::to_snake_case,
    .cols = tidyselect::everything()
  )


arbol <- tree::tree(
  formula = species ~ .,
  data = iris
)

# dev / (obs - leafs)
base::summary(arbol)


# b) ----------------------------------------------------------------------

## Escriba arbol para visualizar los detalles del modelo.
arbol


# c) ----------------------------------------------------------------------

## Construya e interprete el gráfico del modelo arbol.
base::plot(arbol)
graphics::text(arbol, pretty = 0L)


# d) ----------------------------------------------------------------------

## ¿Qué se obtiene mediante prune.tree?
tree::prune.tree(arbol)


# e) ----------------------------------------------------------------------

## Utilizando la función cv.tree pode arbol e interprete el resultado.
arbol_cv <- tree::cv.tree(
  object = arbol,
  FUN = prune.misclass
)

arbol_pruned <- tree::prune.tree(
  tree = arbol,
  best = 4L
)

tree:::print.tree(arbol_pruned)


# f) ----------------------------------------------------------------------

## Divida el conjunto de datos en entrenamiento y testeo. Construya un nuevo árbol con el conjunto de entrenamiento y
## evalúe la precisión con el conjunto de testeo.

base::set.seed(2L)
train <- base::sample(1L:base::nrow(iris), size = .75 * base::nrow(iris))

base::length(train) / base::nrow(iris)

# Construye data set de test
iris_test <- iris[-train, ]

# Re entrena modelo
iris_tree <- tree::tree(
  formula = species ~ .,
  data = iris,
  subset = train
)

base::plot(iris_tree)
graphics::text(iris_tree, pretty = 0L)

# Selecciona prune
base::set.seed(2L)
iris_cv <- tree::cv.tree(
  object = iris_tree,
  FUN = prune.misclass
)

iris_tree <- tree::prune.tree(
  tree = iris_tree,
  best = 3L
)

iris_pred <- stats::predict(iris_tree, iris_test, type = "class")

stats::xtabs(~iris_pred + iris_test[["species"]])

1 - base::sum(base::diag(stats::xtabs(~iris_pred + iris_test[["species"]]))) / base::sum(stats::xtabs(~iris_pred + iris_test[["species"]]))


#===============#
#### THE END ####
#===============#