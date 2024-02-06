# Load in the libraries
library(dplyr)
library(tidymodels)
library(vip)
library(keras)
library(rBayesianOptimization)
library(caret)
library(tensorflow)


#### Set up the Data ####
load("processed_AnalysisData_no200.Rdata")
processed_data<-processed_data_no200

processed_data%>%group_by(spCode,fishNum)%>%count()

# Remove individuals with missing transducers
processed_data<-processed_data%>%filter(is.na(F100)==F)

# also remove individual LWF23018 (only two pings)
processed_data<-processed_data%>%filter(fishNum!="LWF23018")

processed_data<-processed_data%>%select(-F90)
processed_data<-processed_data%>%select(-F90.5)

processed_data<-processed_data%>%filter(spCode == "81" |spCode == "91"|spCode == "316")

processed_data$species<-ifelse(processed_data$spCode==81, "LT", 
                               ifelse(processed_data$spCode == 91, "LWF", "SMB"))

processed_data<-processed_data%>%filter(is.na(aspectAngle)==F & is.na(Angle_major_axis)==F)

processed_data<-processed_data%>%filter(F100>-1000)

#### Train/Test Set Dataset 1&2 ####
set.seed(73)
split<-group_initial_split(processed_data,group=fishNum,strata = species, prop=0.7)
train<-training(split)
val_test<-testing(split)
split2<-group_initial_split(val_test,group=fishNum,strata = species, prop=0.5)
validate<-training(split2)
test<-testing(split2)

train%>%group_by(species)%>%count()
validate%>%group_by(species)%>%count()
test%>%group_by(species)%>%count()

train<-slice_sample(train, n = 6882, by = species)
validate<-slice_sample(validate, n = 1125, by = species)
test<-slice_sample(test, n = 1127, by = species)

train%>%group_by(species)%>%count()
validate%>%group_by(species)%>%count()
test%>%group_by(species)%>%count()

train$y<-NA
train$y[train$species=="LT"]<-0
train$y[train$species=="LWF"]<-1
train$y[train$species=="SMB"]<-2
summary(train$y)
dummy_y_train<-to_categorical(train$y, num_classes = 3)

validate$y<-NA
validate$y[validate$species=="LT"]<-0
validate$y[validate$species=="LWF"]<-1
validate$y[validate$species=="SMB"]<-2
summary(validate$y)
dummy_y_validate<-to_categorical(validate$y, num_classes = 3)

test$y<-NA
test$y[test$species=="LT"]<-0
test$y[test$species=="LWF"]<-1
test$y[test$species=="SMB"]<-2
summary(test$y)
dummy_y_test<-to_categorical(test$y, num_classes = 3)

#### Dataset 1 #####
x_train <- train %>% 
  select(c(21:24,52:300))
x_train[,5:253]<-exp(x_train[,5:253]/10)
x_train<-x_train%>%scale()
x_train<-as.matrix(x_train)

xmean<-attributes(x_train)$`scaled:center`
xsd<-attributes(x_train)$`scaled:scale`

x_validate <- validate %>% 
  select(c(21:24,52:300))
x_validate[,5:253]<-exp(x_validate[,5:253]/10)
x_validate<-x_validate%>%scale(xmean,xsd)
x_validate<-as.matrix(x_validate)

x_test <- test %>%
  select(c(21:24,52:300))
x_test[,5:253]<-exp(x_test[,5:253]/10)
x_test<-x_test%>%scale(xmean,xsd)
x_test<-as.matrix(x_test)

## Shuffle training data
set.seed(250)
x<-sample(1:nrow(x_train))
x_train_S= x_train[x, ] 
dummy_y_train_S= dummy_y_train[x, ] 

set.seed(250)
x<-sample(1:nrow(x_validate))
x_validate_S= x_validate[x, ] 
dummy_y_validate_S= dummy_y_validate[x,] 

set.seed(250)
x<-sample(1:nrow(x_test))
x_test_S= x_test[x, ]
dummy_y_test_S= dummy_y_test[x,]

#### Dataset 1; Initial Model ####
set_random_seed(78)

model <- keras_model_sequential()
model %>%
  layer_dense(units = 126, input_shape = c(253),kernel_regularizer = regularizer_l1_l2(0,0)) %>%
  layer_activation_leaky_relu()%>%
  layer_dropout(rate=0)%>%
  layer_dense(units = 63,kernel_regularizer = regularizer_l1_l2(0,0)) %>%
  layer_activation_leaky_relu()%>%
  layer_dropout(rate=0)%>%
  layer_dense(units = 31,kernel_regularizer = regularizer_l1_l2(0,0)) %>%
  layer_activation_leaky_relu()%>%
  layer_dropout(rate=0)%>%
  layer_dense(units = 15,kernel_regularizer = regularizer_l1_l2(0,0)) %>%
  layer_activation_leaky_relu()%>%
  layer_dropout(rate=0)%>%
  layer_dense(units = 3, activation = "softmax")

#summary(model)

model %>% compile(
  loss = 'categorical_crossentropy',
  optimizer =  optimizer_adam(),
  metrics = c('accuracy'))

callbacks <- list(callback_early_stopping(monitor = "val_loss",patience = 25,
                                           min_delta=0.01,restore_best_weights = F))

history <- model %>% fit(
  x_train_S, dummy_y_train_S,
  batch_size = 25000, 
  epochs = 150,
validation_data = list(x_validate_S,dummy_y_validate_S),
  callbacks = callbacks)

evaluate(model, x_test, dummy_y_test) 

plot(history)

preds<-predict(model, x=x_test)

species.predictions<-apply(preds,1,which.max)
species.predictions<-as.factor(ifelse(species.predictions == 1, "LT",
                                      ifelse(species.predictions ==2, "LWF", "SMB")))
confusionMatrix(species.predictions,as.factor(test$species))



#### Dataset 1; Bayes Optimization ####
keras_fit <- function(dropout_1, dropout_2, dropout_3,dropout_4,reg1,reg2,reg3,reg4){
  model <- keras_model_sequential()
  model %>%
    layer_dense(units = 126, input_shape = c(253), kernel_regularizer = regularizer_l2(l=reg1)) %>%
    layer_activation_leaky_relu()%>%
    layer_dropout(rate=dropout_1)%>%
    layer_dense(units = 63, kernel_regularizer = regularizer_l2(l=reg2)) %>%
    layer_activation_leaky_relu()%>%
    layer_dropout(rate=dropout_2)%>%
    layer_dense(units = 31, kernel_regularizer = regularizer_l2(l=reg3)) %>%
    layer_activation_leaky_relu()%>%
    layer_dropout(rate=dropout_3)%>%
    layer_dense(units = 15, kernel_regularizer = regularizer_l2(l=reg4)) %>%
    layer_activation_leaky_relu()%>%
    layer_dropout(rate=dropout_4)%>%
    layer_dense(units = 3, activation = "softmax")
  
  
  model %>% compile(
    loss = 'categorical_crossentropy',
    optimizer =  optimizer_adam(0.005),
    metrics = c('accuracy'))
  
  callbacks <- list( callback_early_stopping(monitor = "val_loss",patience = 10,
                                             min_delta=0.01,restore_best_weights = F))
  
  history <- model %>% fit(
    x_train_S, dummy_y_train_S,
    batch_size = 1000, 
    epochs = 150,
    validation_split = 0.3,
    callbacks = callbacks)
  
  result <- list(Score = -min(history$metrics$val_loss), 
                 Pred = 0)
  
  return(result)
  
}


## Define the search boundaries
search_bound_keras <- list(dropout_1 = c(0,0.5),
                           dropout_2 = c(0,0.5),
                           dropout_3 = c(0,0.5),
                           dropout_4 = c(0,0.5),
                           reg1 = c(1e-6,1e-2),
                           reg2 = c(1e-6,1e-2),
                           reg3 = c(1e-6,1e-2),
                           reg4 = c(1e-6,1e-2))

search_grid_keras <- data.frame(dropout_1 = runif(10, 0, 0.5),
                                dropout_2 = runif(10, 0, 0.5),
                                dropout_3 = runif(10, 0, 0.5),
                                dropout_4 = runif(10, 0, 0.5),
                                reg1 = runif(10,1e-6,1e-2),
                                reg2 = runif(10,1e-6,1e-2),
                                reg3 = runif(10,1e-6,1e-2),
                                reg4 = runif(10,1e-6,1e-2))
head(search_grid_keras)

## Run the Bayesian Optimization
#set.seed(1)
bayes_keras <- rBayesianOptimization::BayesianOptimization(FUN = keras_fit, bounds = search_bound_keras, init_points = 0, init_grid_dt = search_grid_keras, n_iter = 20, acq = "ucb")


#### Dataset 1; Best model ####
set.seed(75)
best_model <- keras_model_sequential()
best_model %>%
  layer_dense(units = 126, input_shape = c(253), kernel_regularizer = regularizer_l2(l=0)) %>%
  layer_activation_leaky_relu()%>%
  layer_dropout(rate=bayes_keras$Best_Par[1])%>%
  layer_dense(units = 63, kernel_regularizer = regularizer_l2(l=bayes_keras$Best_Par[5])) %>%
  layer_activation_leaky_relu()%>%
  layer_dropout(rate=bayes_keras$Best_Par[2])%>%
  layer_dense(units = 31, kernel_regularizer = regularizer_l2(l=0)) %>%
  layer_activation_leaky_relu()%>%
  layer_dropout(rate=0)%>%
  layer_dense(units = 15, kernel_regularizer = regularizer_l2(l=bayes_keras$Best_Par[7])) %>%
  layer_activation_leaky_relu()%>%
  layer_dropout(rate=0)%>%
  layer_dense(units = 3, activation = "softmax")


best_model %>% compile(
  loss = 'categorical_crossentropy',
  optimizer =  optimizer_adam(0.005),
  metrics = c('accuracy'))

callbacks <- list( callback_early_stopping(monitor = "val_loss",patience = 10,
                                           min_delta=0.01,restore_best_weights = F))

history <- best_model %>% fit(
  x_train_S, dummy_y_train_S,
  batch_size = 1000, 
  epochs = 150,
  validation_split = 0.3,shuffle=T,
  callbacks = callbacks)

history
evaluate(best_model, x_validate_S, dummy_y_validate_S) 

preds<-predict(best_model, x=x_validate)

species.predictions<-apply(preds,1,which.max)
species.predictions<-as.factor(ifelse(species.predictions == 1, "LT",
                                      ifelse(species.predictions ==2, "LWF", "SMB")))
confusionMatrix(species.predictions,as.factor(validate$species))



#### Dataset 2 ####
x_train <- train %>% 
  select(c(21:24,52:300))
x_train[,5:253]<-x_train[,5:253]+10*log10(450/train$totalLength)
x_train[,5:253]<-exp(x_train[,5:253]/10)
x_train<-x_train%>%scale()
x_train<-as.matrix(x_train)

xmean<-attributes(x_train)$`scaled:center`
xsd<-attributes(x_train)$`scaled:scale`

x_validate <- validate %>%
  select(c(21:24,52:300))
x_validate[,5:253]<-x_validate[,5:253]+10*log10(450/validate$totalLength)
x_validate[,5:253]<-exp(x_validate[,5:253]/10)
x_validate<-x_validate%>%scale(xmean,xsd)
x_validate<-as.matrix(x_validate)

x_test <- test %>% 
  select(c(21:24,52:300))
x_test[,5:253]<-x_test[,5:253]+10*log10(450/test$totalLength)
x_test[,5:253]<-exp(x_test[,5:253]/10)
x_test<-x_test%>%scale(xmean,xsd)
x_test<-as.matrix(x_test)

## Shuffle training data
set.seed(250)
x<-sample(1:nrow(x_train))
x_train_S= x_train[x, ] 
dummy_y_train_S= dummy_y_train[x, ] 

set.seed(250)
x<-sample(1:nrow(x_validate))
x_validate_S= x_validate[x, ]
dummy_y_validate_S= dummy_y_validate[x,]


#### Dataset 2; Initial model ####
set_random_seed(78)
model1 <- keras_model_sequential()
model1 %>%
  layer_dense(units = 150, input_shape = c(253),kernel_regularizer = regularizer_l2()) %>%
  layer_activation_leaky_relu()%>%
  layer_dropout(rate=0)%>%
  layer_dense(units = 100,kernel_regularizer = regularizer_l2()) %>%
  layer_activation_leaky_relu()%>%
  layer_dropout(rate=0)%>%
  layer_dense(units = 50,kernel_regularizer = regularizer_l2()) %>%
  layer_activation_leaky_relu()%>%
  layer_dropout(rate=0)%>%
  layer_dense(units = 25,kernel_regularizer = regularizer_l2()) %>%
  layer_activation_leaky_relu()%>%
  layer_dropout(rate=0)%>%
  layer_dense(units = 12,kernel_regularizer = regularizer_l2()) %>%
  layer_activation_leaky_relu()%>%
  layer_dropout(rate=0)%>%
  # layer_dense(units = 66,kernel_regularizer = regularizer_l1_l2(0,0)) %>%
  # layer_activation_leaky_relu()%>%
  # layer_dropout(rate=0)%>%
  # layer_dense(units = 53,kernel_regularizer = regularizer_l1_l2(0,0)) %>%
  # layer_activation_leaky_relu()%>%
  # layer_dropout(rate=0)%>%
  # layer_dense(units = 42,kernel_regularizer = regularizer_l1_l2(0,0)) %>%
  # layer_activation_leaky_relu()%>%
  # layer_dropout(rate=0)%>%
  # layer_dense(units = 34,kernel_regularizer = regularizer_l1_l2(0,0)) %>%
  # layer_activation_leaky_relu()%>%
  # layer_dropout(rate=0)%>%
  layer_dense(units = 3, activation = "softmax")

#summary(model)

model1 %>% compile(
  loss = 'categorical_crossentropy',
  optimizer =  optimizer_adam(),
  metrics = c('accuracy'))

callbacks <- list(callback_early_stopping(monitor = "val_loss",patience = 25,
                                          min_delta=0.01,restore_best_weights = F))

history <- model1 %>% fit(
  x_train_S, dummy_y_train_S,
  batch_size = 1000, 
  epochs = 150,
  validation_data = list(x_validate_S,dummy_y_validate_S),
  callbacks = callbacks)

evaluate(model1, x_test, dummy_y_test) 

# Test Accuracy = 0.49

set_random_seed(78)
model2 <- keras_model_sequential()
model2 %>%
  layer_dense(units = 100, input_shape = c(253),kernel_regularizer = regularizer_l2()) %>%
  layer_activation_leaky_relu()%>%
  layer_dropout(rate=0)%>%
  layer_dense(units = 50,kernel_regularizer = regularizer_l2()) %>%
  layer_activation_leaky_relu()%>%
  layer_dropout(rate=0)%>%
  layer_dense(units = 25,kernel_regularizer = regularizer_l2()) %>%
  layer_activation_leaky_relu()%>%
  layer_dropout(rate=0)%>%
  layer_dense(units = 12,kernel_regularizer = regularizer_l2()) %>%
  layer_activation_leaky_relu()%>%
  layer_dropout(rate=0)%>%
  # layer_dense(units = 12,kernel_regularizer = regularizer_l2()) %>%
  # layer_activation_leaky_relu()%>%
  # layer_dropout(rate=0)%>%
  # layer_dense(units = 66,kernel_regularizer = regularizer_l1_l2(0,0)) %>%
  # layer_activation_leaky_relu()%>%
  # layer_dropout(rate=0)%>%
  # layer_dense(units = 53,kernel_regularizer = regularizer_l1_l2(0,0)) %>%
  # layer_activation_leaky_relu()%>%
  # layer_dropout(rate=0)%>%
  # layer_dense(units = 42,kernel_regularizer = regularizer_l1_l2(0,0)) %>%
  # layer_activation_leaky_relu()%>%
  # layer_dropout(rate=0)%>%
  # layer_dense(units = 34,kernel_regularizer = regularizer_l1_l2(0,0)) %>%
  # layer_activation_leaky_relu()%>%
# layer_dropout(rate=0)%>%
layer_dense(units = 3, activation = "softmax")

#summary(model)

model2 %>% compile(
  loss = 'categorical_crossentropy',
  optimizer =  optimizer_adam(),
  metrics = c('accuracy'))

callbacks <- list(callback_early_stopping(monitor = "val_loss",patience = 10,
                                          min_delta=0.01,restore_best_weights = F))

history <- model2 %>% fit(
  x_train_S, dummy_y_train_S,
  batch_size = 1000, 
  epochs = 150,
  #validation_data = list(x_validate_S,dummy_y_validate_S),
  validation_split = 0.3,
  callbacks = callbacks)

evaluate(model2, x_test, dummy_y_test) 

# Test Accuracy = 0.545

#### Dataset 2; Bayes Optimization ####
keras_fit <- function(reg1,reg2,reg3,reg4,reg5,learningrate,batchsize){
  model1 <- keras_model_sequential()
  model1 %>%
    layer_dense(units = 150, input_shape = c(253),kernel_regularizer = regularizer_l2(reg1)) %>%
    layer_activation_leaky_relu()%>%
    layer_dropout(rate=0)%>%
    layer_dense(units = 100,kernel_regularizer = regularizer_l2(reg2)) %>%
    layer_activation_leaky_relu()%>%
    layer_dropout(rate=0)%>%
    layer_dense(units = 50,kernel_regularizer = regularizer_l2(reg3)) %>%
    layer_activation_leaky_relu()%>%
    layer_dropout(rate=0)%>%
    layer_dense(units = 25,kernel_regularizer = regularizer_l2(reg4)) %>%
    layer_activation_leaky_relu()%>%
    layer_dropout(rate=0)%>%
    layer_dense(units = 12,kernel_regularizer = regularizer_l2(reg5)) %>%
    layer_activation_leaky_relu()%>%
    layer_dropout(rate=0)%>%
    layer_dense(units = 3, activation = "softmax")  
  
  model1 %>% compile(
    loss = 'categorical_crossentropy',
    optimizer =  optimizer_adam(learningrate),
    metrics = c('accuracy'))
  
  callbacks <- list( callback_early_stopping(monitor = "val_loss",patience = 10,
                                             min_delta=0.01,restore_best_weights = F))
  
  history <- model1 %>% fit(
    x_train_S, dummy_y_train_S,
    batch_size = batchsize, 
    epochs = 150,
    validation_split=0.3,
    #validation_data = list(x_validate_S,dummy_y_validate_S),
    callbacks = callbacks)
  
  result <- list(Score = -min(history$metrics$val_loss), 
                 Pred = 0)
  
  return(result)
  
}


## Define the search boundaries
search_bound_keras <- list(reg1=c(1e-7,1e-2),
                           reg2=c(1e-7,1e-2),
                           reg3=c(1e-7,1e-2),
                           reg4=c(1e-7,1e-2),
                           reg5=c(1e-7,1e-2),
                           learningrate=c(1e-4,1e-2),
                           batchsize=c(250,2000))

search_grid_keras <- data.frame(reg1 = runif(5,1e-7,1e-2),
                                reg2 = runif(5,1e-7,1e-2),
                                reg3 = runif(5,1e-7,1e-2),
                                reg4 = runif(5,1e-7,1e-2),
                                reg5 = runif(5,1e-7,1e-2),
                                learningrate = runif(5,1e-3,1e-2),
                                batchsize=runif(5,250,2000))
head(search_grid_keras)

## Run the Bayesian Optimization
#set.seed(1)
bayes_keras <- rBayesianOptimization::BayesianOptimization(FUN = keras_fit, bounds = search_bound_keras, init_points = 0, init_grid_dt = search_grid_keras , n_iter = 5, acq = "ucb")

#### Dataset 2; Best model ####
set.seed(12)
best_model1 <- keras_model_sequential()
best_model1 %>%
  layer_dense(units = 150, input_shape = c(253),kernel_regularizer = regularizer_l2(bayes_keras$Best_Par[1])) %>%
  layer_activation_leaky_relu()%>%
  layer_dropout(rate=0)%>%
  layer_dense(units = 100,kernel_regularizer = regularizer_l2(bayes_keras$Best_Par[2])) %>%
  layer_activation_leaky_relu()%>%
  layer_dropout(rate=0)%>%
  layer_dense(units = 50,kernel_regularizer = regularizer_l2(bayes_keras$Best_Par[3])) %>%
  layer_activation_leaky_relu()%>%
  layer_dropout(rate=0)%>%
  layer_dense(units = 25,kernel_regularizer = regularizer_l2(bayes_keras$Best_Par[4])) %>%
  layer_activation_leaky_relu()%>%
  layer_dropout(rate=0)%>%
  layer_dense(units = 12,kernel_regularizer = regularizer_l2(bayes_keras$Best_Par[5])) %>%
  layer_activation_leaky_relu()%>%
  layer_dropout(rate=0)%>%
  layer_dense(units = 3, activation = "softmax")


best_model1 %>% compile(
  loss = 'categorical_crossentropy',
  optimizer =  optimizer_adam(bayes_keras$Best_Par[6]),
  metrics = c('accuracy'))

callbacks <- list( callback_early_stopping(monitor = "val_loss",patience = 10,
                                           min_delta=0.01,restore_best_weights = T))

history <- best_model1 %>% fit(
  x_train_S, dummy_y_train_S,
  batch_size = 2000, 
  epochs = 150,
 validation_data=list(x_validate_S,dummy_y_validate_S),
  callbacks = callbacks)

plot(history)
evaluate(best_model1, x_test, dummy_y_test) 
#0.562

preds<-predict(best_model1, x=x_test)

species.predictions<-apply(preds,1,which.max)
species.predictions<-as.factor(ifelse(species.predictions == 1, "LT",
                                      ifelse(species.predictions ==2, "LWF", "SMB")))
confusionMatrix(species.predictions,as.factor(test$species))

# Good with bass, confuses LT with LWF


keras_fit <- function(reg1,reg2,reg3,reg4,learningrate,batchsize){
  model2 <- keras_model_sequential()
  model2 %>%
    layer_dense(units = 100, input_shape = c(253),kernel_regularizer = regularizer_l2(reg1)) %>%
    layer_activation_leaky_relu()%>%
    layer_dropout(rate=0)%>%
    layer_dense(units = 50,kernel_regularizer = regularizer_l2(reg2)) %>%
    layer_activation_leaky_relu()%>%
    layer_dropout(rate=0)%>%
    layer_dense(units = 25,kernel_regularizer = regularizer_l2(reg3)) %>%
    layer_activation_leaky_relu()%>%
    layer_dropout(rate=0)%>%
    layer_dense(units = 12,kernel_regularizer = regularizer_l2(reg4)) %>%
    layer_activation_leaky_relu()%>%
    layer_dropout(rate=0)%>%
    layer_dense(units = 3, activation = "softmax")  
  
  model2 %>% compile(
    loss = 'categorical_crossentropy',
    optimizer =  optimizer_adam(learningrate),
    metrics = c('accuracy'))
  
  callbacks <- list( callback_early_stopping(monitor = "val_loss",patience = 10,
                                             min_delta=0.01,restore_best_weights = F))
  
  history <- model2 %>% fit(
    x_train_S, dummy_y_train_S,
    batch_size = batchsize, 
    epochs = 150,
    validation_split=0.3,
    #validation_data = list(x_validate_S,dummy_y_validate_S),
    callbacks = callbacks)
  
  result <- list(Score = -min(history$metrics$val_loss), 
                 Pred = 0)
  
  return(result)
  
}


## Define the search boundaries
search_bound_keras <- list(reg1=c(1e-7,1e-2),
                           reg2=c(1e-7,1e-2),
                           reg3=c(1e-7,1e-2),
                           reg4=c(1e-7,1e-2),
                           learningrate=c(1e-4,1e-2),
                           batchsize=c(250,2000))

search_grid_keras <- data.frame(reg1 = runif(5,1e-7,1e-2),
                                reg2 = runif(5,1e-7,1e-2),
                                reg3 = runif(5,1e-7,1e-2),
                                reg4 = runif(5,1e-7,1e-2),
                                learningrate = runif(5,1e-3,1e-2),
                                batchsize=runif(5,250,2000))
head(search_grid_keras)

## Run the Bayesian Optimization
#set.seed(1)
bayes_keras <- rBayesianOptimization::BayesianOptimization(FUN = keras_fit, bounds = search_bound_keras, init_points = 0, init_grid_dt = search_grid_keras , n_iter = 5, acq = "ucb")

set.seed(12)
best_model2 <- keras_model_sequential()
best_model2 %>%
  layer_dense(units = 100, input_shape = c(253),kernel_regularizer = regularizer_l2(bayes_keras$Best_Par[1])) %>%
  layer_activation_leaky_relu()%>%
  layer_dropout(rate=0)%>%
  layer_dense(units = 50,kernel_regularizer = regularizer_l2(bayes_keras$Best_Par[2])) %>%
  layer_activation_leaky_relu()%>%
  layer_dropout(rate=0)%>%
  layer_dense(units = 25,kernel_regularizer = regularizer_l2(bayes_keras$Best_Par[3])) %>%
  layer_activation_leaky_relu()%>%
  layer_dropout(rate=0)%>%
  layer_dense(units = 12,kernel_regularizer = regularizer_l2(bayes_keras$Best_Par[4])) %>%
  layer_activation_leaky_relu()%>%
  layer_dropout(rate=0)%>%
  layer_dense(units = 3, activation = "softmax")


best_model2 %>% compile(
  loss = 'categorical_crossentropy',
  optimizer =  optimizer_adam(bayes_keras$Best_Par[5]),
  metrics = c('accuracy'))

callbacks <- list( callback_early_stopping(monitor = "val_loss",patience = 10,
                                           min_delta=0.01,restore_best_weights = T))

history <- best_model2 %>% fit(
  x_train_S, dummy_y_train_S,
  batch_size = ,kernel_regularizer = regularizer_l2(bayes_keras$Best_Par[6]), 
  epochs = 150,
  validation_split = 0.3,
  # validation_data=list(x_validate_S,dummy_y_validate_S),
  callbacks = callbacks)

plot(history)
evaluate(best_model2, x_test, dummy_y_test) 
#0.533

preds<-predict(best_model2, x=x_test)

species.predictions<-apply(preds,1,which.max)
species.predictions<-as.factor(ifelse(species.predictions == 1, "LT",
                                      ifelse(species.predictions ==2, "LWF", "SMB")))
confusionMatrix(species.predictions,as.factor(test$species))

# Good with bass, confuses LT with LWF
