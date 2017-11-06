# Lectura de datos en R

car_data <- read.csv("../dataset/car.data", sep=",", header=FALSE)
colnames(car_data) <- c("buying","maint","doors","persons","lug_boot","safety","class")
