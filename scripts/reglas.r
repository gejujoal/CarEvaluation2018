# Script para extraccion de reglas de asociacion

# Librerias
# install.packages("arules")
library(arules)
library(arulesViz)

# Lectura de datos en R
path = "."
completePath = paste(path, "/dataset/car.data", sep="")
car_data <- read.csv(completePath, sep=",", header=FALSE)
# levels(car_data$class) # obtener los posibles valores para la clase
colnames(car_data) <- c("buying","maint","doors","persons","lug_boot","safety","class")


# Aplica Apriori
rules <- apriori(car_data, parameter = list(minlen=2, supp=0.09, conf=0.8), control = list(verbose=F))
#rules <- apriori(car_data, parameter = list(minlen=2, supp=0.02, conf=0.8), appearance = list(rhs=c("class=unacc", "class=acc", "class=good", "class=vgood"), default="lhs"), control = list(verbose=F))
length(rules)

rules.sorted <- sort(rules, by="support")
inspect(rules.sorted)

# Grafico de relaciones
# plot(rules, method="graph", control=list(type="items"))
