#=================#
#### PR6 - EJ2 ####
#=================#

# Con los datos iris y usando la librería rpart

library(magrittr)


# a) ----------------------------------------------------------------------

## Construya un éarbol de clasificación (que llamaremos arbol) con Species como variable de respuesta.
## Utilice la función summary() para describir los resultados obtenidos.

iris <- datasets::iris %>%
  tibble::as_tibble() %>% 
  dplyr::rename_with(
    .fn = snakecase::to_snake_case,
    .cols = tidyselect::everything()
  )


# formula: model formula
# data: a data frame of model data
# control: split stopping rules. minsplit = 10 means that there must be at least 10 obs in a rectangle for a split to be attempted 
arbol <- rpart::rpart(
  formula = species ~ .,
  data = iris,
  control = rpart::rpart.control(
    minsplit = 10L
  )
)

arbol


base::summary(arbol)
rpart:::summary.rpart(arbol)


# b) ----------------------------------------------------------------------

## Construya e interprete el gráfico del modelo arbol.
base::plot(arbol)
graphics::text(arbol, pretty = 0L)


base::plot(partykit::as.party(arbol))


# c) ----------------------------------------------------------------------

# nsplit: number of splits
# xerror: estimates of the cross-validated prediction error for nsplit
rpart::printcp(arbol)


# d) ----------------------------------------------------------------------

opt <- base::which.min(arbol[["cptable"]][, "xerror"])

cp <- arbol[["cptable"]][opt, "CP"]

arbol_pruned <- rpart::prune(arbol, cp = cp)

base::plot(partykit::as.party(arbol_pruned))
base::plot(partykit::as.party(arbol))


#===============#
#### THE END ####
#===============#