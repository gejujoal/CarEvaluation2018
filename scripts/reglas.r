#ibrerias necesarias, si no estan instaladas se instalaran
if (!require(caret)) install.packages("caret")
if (!require(arules)) install.packages("arules")
if (!require(arulesViz)) install.packages("arulesViz")


#lectura de datos
car_data <- read.csv("https://archive.ics.uci.edu/ml/machine-learning-databases/car/car.data", sep=",", header=FALSE)
colnames(car_data) <- c("buying","maint","doors","persons","lug_boot","safety","class")


#visualización para ver como se reparte la tabla
apply(car_data,2,table)

#nos centramos en ver los coches no seguros y seguros para tener una idea a priori
cars_unacc <- car_data[car_data$class=="unacc",]
apply(cars_unacc,2,table)

cars_good <- car_data[car_data$class=="vgood",]
apply(cars_good,2,table)

###################APLICAMOS REGLAS DE ASOCIACION############################
# Aplica Apriori
rules <- apriori(car_data, parameter = list(minlen=2, supp=0.09, conf=0.8), control = list(verbose=F))
#rules <- apriori(car_data, parameter = list(minlen=2, supp=0.02, conf=0.8), appearance = list(rhs=c("class=unacc", "class=acc", "class=good", "class=vgood"), default="lhs"), control = list(verbose=F))
length(rules)

rules.sorted <- sort(rules, by="support")
inspect(rules.sorted)

# lhs                            rhs           support    confidence lift     count
# [1]  {safety=low}                => {class=unacc} 0.33333333 1.0000000  1.428099 576  
# [2]  {persons=2}                 => {class=unacc} 0.33333333 1.0000000  1.428099 576  
# [3]  {buying=vhigh}              => {class=unacc} 0.20833333 0.8333333  1.190083 360  
# [4]  {maint=vhigh}               => {class=unacc} 0.20833333 0.8333333  1.190083 360  
# [5]  {lug_boot=big,safety=low}   => {class=unacc} 0.11111111 1.0000000  1.428099 192  
# [6]  {persons=2,lug_boot=big}    => {class=unacc} 0.11111111 1.0000000  1.428099 192  
# [7]  {persons=4,safety=low}      => {class=unacc} 0.11111111 1.0000000  1.428099 192  
# [8]  {persons=more,safety=low}   => {class=unacc} 0.11111111 1.0000000  1.428099 192  
# [9]  {lug_boot=med,safety=low}   => {class=unacc} 0.11111111 1.0000000  1.428099 192  
# [10] {persons=2,lug_boot=med}    => {class=unacc} 0.11111111 1.0000000  1.428099 192  
# [11] {persons=2,safety=high}     => {class=unacc} 0.11111111 1.0000000  1.428099 192  
# [12] {persons=2,safety=med}      => {class=unacc} 0.11111111 1.0000000  1.428099 192  
# [13] {lug_boot=small,safety=low} => {class=unacc} 0.11111111 1.0000000  1.428099 192  
# [14] {persons=2,safety=low}      => {class=unacc} 0.11111111 1.0000000  1.428099 192  
# [15] {persons=2,lug_boot=small}  => {class=unacc} 0.11111111 1.0000000  1.428099 192  
# [16] {lug_boot=small,safety=med} => {class=unacc} 0.09085648 0.8177083  1.167769 157  


#######PASAMOS A RANDOM FOREST PARA PREDECIR LA SEGURIDAD DEL COCHE SEGUN ATRIBUTOS

## 75%  para entrenar
smp_size <- floor(0.75 * nrow(car_data))

set.seed(123)
train_ind <- sample(seq_len(nrow(car_data)), size = smp_size)

train <- car_data[train_ind, ]
test <- car_data[-train_ind, ]

#como control haremos un CV con los siguientes parámertos
control <- trainControl(method="repeatedcv", number=10, repeats=3)
metric <- "Accuracy"
set.seed(seed)
mtry <- sqrt(ncol(train))
tunegrid <- expand.grid(.mtry=mtry)
rf_default <- train(class~., data=train, method="rf", metric=metric, tuneGrid=tunegrid, trControl=control)

#visualizamos el acc del RF
print(rf_default)
# 
# Random Forest 
# 
# 1296 samples
# 6 predictor
# 4 classes: 'acc', 'good', 'unacc', 'vgood' 
# 
# No pre-processing
# Resampling: Cross-Validated (10 fold, repeated 3 times) 
# Summary of sample sizes: 1165, 1166, 1166, 1167, 1168, 1167, ... 
# Resampling results:
#   
#   Accuracy   Kappa    
# 0.8608195  0.6643527
# 
# Tuning paramter 'mtry' was held constant at a value of 2.645751
# 

# Con los datos de test validamos el modelo y vemos la matriz de confusión
test_predict <- predict(rf_default,test)
cfMatrix <- confusionMatrix(data = test$class, test_predict)

cfMatrix
# 
# 
# Confusion Matrix and Statistics
# 
# Reference
# Prediction acc good unacc vgood
# acc    72    1    28     0
# good   15    0     2     0
# unacc   1    0   294     0
# vgood  14    0     3     2
# 
# Overall Statistics
# 
# Accuracy : 0.8519         
# 95% CI : (0.8148, 0.884)
# No Information Rate : 0.7569         
# P-Value [Acc > NIR] : 8.744e-07      
# 
# Kappa : 0.6535         
# Mcnemar's Test P-Value : NA             
# 
# Statistics by Class:
# 
#                      Class: acc Class: good Class: unacc Class: vgood
# Sensitivity              0.7059    0.000000       0.8991      1.00000
# Specificity              0.9121    0.960557       0.9905      0.96047
# Pos Pred Value           0.7129    0.000000       0.9966      0.10526
# Neg Pred Value           0.9094    0.997590       0.7591      1.00000
# Prevalence               0.2361    0.002315       0.7569      0.00463
# Detection Rate           0.1667    0.000000       0.6806      0.00463
# Detection Prevalence     0.2338    0.039352       0.6829      0.04398
# Balanced Accuracy        0.8090    0.480278       0.9448      0.98023

#Guardamos la tabla de la matriz de confusion para usarlo con gpplot
cm_table <- as.data.frame(cfMatrix$table)

#Para cada tipo de clase se guardan en diferentes DF y se usa ggplot para ver de manera mas grafica en que clase es mejor el modelo
tab_acc <- cm_table[cm_table$Prediction=="acc",]

ggplot(tab_acc, aes(Prediction, Freq)) +   
  ggtitle("Predicciones reales vs la asignada como 'acc'")+
  geom_bar(aes(fill = Reference), position = "dodge", stat="identity")

tab_good <- cm_table[cm_table$Prediction=="good",]

ggplot(tab_good, aes(Prediction, Freq)) +   
  ggtitle("Predicciones reales vs la asignada como 'good'")+
  geom_bar(aes(fill = Reference), position = "dodge", stat="identity")

tab_unacc <- cm_table[cm_table$Prediction=="unacc",]

ggplot(tab_unacc, aes(Prediction, Freq)) +   
  ggtitle("Predicciones reales vs la asignada como 'unacc'")+
  geom_bar(aes(fill = Reference), position = "dodge", stat="identity")

tab_vgood <- cm_table[cm_table$Prediction=="vgood",]

ggplot(tab_vgood, aes(Prediction,  Freq )) +   
  ggtitle("Predicciones reales vs la asignada como 'vgood'")+
  geom_bar(aes(fill = Reference), position = "dodge", stat="identity")


#Conclusiones




