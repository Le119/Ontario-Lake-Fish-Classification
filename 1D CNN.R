## Load in the necessary libraries
library(dplyr)
library(tidyr)
library(keras)
library(rBayesianOptimization)
library(tidymodels)
library(caret)
library(tensorflow)

# Load the data
load("processed_AnalysisData_no200.Rdata")

# make the name easier to type
processed_data<-processed_data_no200

# Look at the structure of individuals
processed_data%>%group_by(spCode,fishNum)%>%count()

# Remove individuals with missing transducers
processed_data<-processed_data%>%filter(is.na(F100)==F)

# also remove individual LWF23018 (only two pings)
processed_data<-processed_data%>%filter(fishNum!="LWF23018")

# remove the 90kHZ and 90.5kHZ columns
processed_data<-processed_data%>%select(-F90)
processed_data<-processed_data%>%select(-F90.5)

# only keep LT (81), LWF (91), and SMB (316)
processed_data<-processed_data%>%filter(spCode == "81" |spCode == "91"|spCode == "316")

processed_data$species<-ifelse(processed_data$spCode==81, "LT", 
                               ifelse(processed_data$spCode == 91, "LWF", "SMB"))

# remove the one ping that has a VERY low TS
processed_data<-processed_data%>%filter(F100>-1000)

#### Train/Test Set####
set.seed(73)
split<-group_initial_split(processed_data,group=fishNum,strata = species, prop=0.5)
train<-training(split)
val_test<-testing(split)
split2<-group_initial_split(val_test,group=fishNum,strata = species, prop=0.7)
validate<-training(split2)
test<-testing(split2)

train%>%group_by(species)%>%count()
validate%>%group_by(species)%>%count()
test%>%group_by(species)%>%count()

# make the dataset balanced between classes
train<-slice_sample(train, n = 5029, by = species)
validate<-slice_sample(validate, n = 2916, by = species)
test<-slice_sample(test, n = 995, by = species)


train%>%group_by(species)%>%count()
validate%>%group_by(species)%>%count()
test%>%group_by(species)%>%count()

# Create the y variable
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

# Select just the frequency columns and standardise to length 450
x_train <- train %>% 
  select(c(52:300))
x_train<-x_train+10*log10(450/train$totalLength)
x_train<-exp(x_train/10)
x_train<-x_train%>%scale()
x_train<-as.matrix(x_train)

xmean<-attributes(x_train)$`scaled:center`
xsd<-attributes(x_train)$`scaled:scale`

x_validate <- validate %>% 
  select(c(52:300))
x_validate<-x_validate+10*log10(450/validate$totalLength)
x_validate<-exp(x_validate/10)
x_validate<-x_validate%>%scale(xmean,xsd)
x_validate<-as.matrix(x_validate)

x_test <- test %>% 
  select(c(52:300))
x_test<-x_test+10*log10(450/test$totalLength)
x_test<-exp(x_test/10)
x_test<-x_test%>%scale(xmean,xsd)
x_test<-as.matrix(x_test)

# ## Augment data trial
# A<-matrix(rnorm(17728800,0,0.0001),nrow=71200,ncol=249)
# x_train_A2<-(rbind(x_train,(x_train+A[1:14240,]),(x_train+A[14241:28480,]),(x_train+A[28481:42720,]),(x_train+A[42721:56960,]),(x_train+A[56961:71200,])))
# dummy_y_train_A2<-rbind(dummy_y_train,dummy_y_train,dummy_y_train,dummy_y_train,dummy_y_train,dummy_y_train)

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

### Fitting the same structure as the chemistry paper
input_shape <- c(249,1)

set_random_seed(5)
model <- keras_model_sequential()
model%>%
  layer_conv_1d(filters = 4, kernel_size = 3, strides = 1, input_shape = c(249, 1),padding="same",kernel_regularizer=regularizer_l2())%>%
  layer_activation_relu()%>%
  layer_batch_normalization()%>%
  layer_conv_1d(filters = 4,kernel_size = 3, strides = 1,padding="same",kernel_regularizer=regularizer_l2())%>%
  layer_activation_relu()%>%
  layer_batch_normalization()%>%
  layer_max_pooling_1d(pool_size = c(2),padding="same")%>%
  layer_conv_1d(filters=8,kernel_size=3, strides = 1,padding="same",kernel_regularizer=regularizer_l2())%>%
  layer_activation_relu()%>%
  layer_batch_normalization()%>%
  layer_conv_1d(filters=8,kernel_size=3, strides = 1,padding="same",kernel_regularizer=regularizer_l2())%>%
  layer_activation_relu()%>%
  layer_max_pooling_1d(pool_size = (2),padding="same")%>%
  layer_batch_normalization()%>%
  layer_flatten()%>%
  layer_dense(units = 3, activation = 'softmax')
  
summary(model)

# Compile model
model %>% compile(
  loss = loss_categorical_crossentropy,
  optimizer = optimizer_adam(),
  metrics = c('accuracy')
)

callbacks <- list( callback_early_stopping(monitor = "val_loss",patience = 200,
                                           min_delta=0.01,restore_best_weights = T))



cnn_history <- model %>% fit(
  x_train_S, dummy_y_train_S,
  batch_size = 10000,
  epochs = 100,
  validation_data = list(x_validate_S,dummy_y_validate_S)
)

plot(cnn_history)

evaluate(model, x_test, dummy_y_test) #60.2% seed(5)
preds<-predict(model, x=x_test)

species.predictions<-apply(preds,1,which.max)
species.predictions<-as.factor(ifelse(species.predictions == 1, "LT",
                                      ifelse(species.predictions ==2, "LWF", "SMB")))
confusionMatrix(species.predictions,as.factor(test$species))


# Leaky relu vs relu seems to make no difference in generalizability

## Bayesian Optimization of the CNN
keras_fit <- function(filter1,kernel,maxpool,regrate,learn,batchsize){
  model <- keras_model_sequential()
  model%>%
    layer_conv_1d(filters = filter1, kernel_size = kernel, strides = 1, input_shape = c(249, 1),padding="same",kernel_regularizer = regularizer_l2(regrate))%>%
    layer_activation_relu()%>%
    layer_batch_normalization()%>%
   layer_conv_1d(filters = filter1,kernel_size = kernel, strides = 1,padding="same",kernel_regularizer = regularizer_l2(regrate))%>%
    layer_activation_relu()%>%
    layer_batch_normalization()%>%
    layer_max_pooling_1d(pool_size = maxpool,padding="same")%>%
    layer_conv_1d(filters=filter1*2,kernel_size=kernel, strides = 1,padding="same",kernel_regularizer = regularizer_l2(regrate))%>%
    layer_activation_relu()%>%
    layer_batch_normalization()%>%
   layer_conv_1d(filters=filter1*2,kernel_size=kernel, strides = 1,padding="same",kernel_regularizer = regularizer_l2(regrate))%>%
    layer_activation_relu()%>%
    layer_max_pooling_1d(pool_size = maxpool,padding="same")%>%
    layer_batch_normalization()%>%
    layer_flatten()%>%
    layer_dense(units = 3, activation = 'softmax')
  
  model %>% compile(
    loss = loss_categorical_crossentropy,
    optimizer = optimizer_adam(learn),
    metrics = c('accuracy')
  )
  
  callbacks <- list( callback_early_stopping(monitor = "val_loss",patience = 100,
                                             min_delta=0.01,restore_best_weights = T))
  
  cnn_history <- model %>% fit(
    x_train_S, dummy_y_train_S,
    batch_size = batchsize,
    epochs = 1000,
    validation_data = list(x_validate_S,dummy_y_validate_S),
    callbacks = callbacks
  )
  
  result <- list(Score = -min(cnn_history$metrics$val_loss), 
                 Pred = 0)
  
  return(result)
  
}

## Define the search boundaries
search_bound_keras <- list(filter1=c(2L,35L),
                           kernel=c(2L,60L),
                           maxpool=c(2L,60L),
                           regrate=c(0.0001,0.1),
                           learn=c(1e-5,1e-2),
                           batchsize=c(250L,1000L))

search_grid_keras <- data.frame(filter1=floor(runif(2,2,6)),
                                kernel=floor(runif(2,2,25)),
                                maxpool=floor(runif(2,2,8)),
                                regrate=runif(2,0.0001,0.1),
                                learn=runif(2,1e-5,1e-2),
                                batchsize=floor(runif(2,250,1001)))
head(search_grid_keras)

## Run the Bayesian Optimization
#set.seed(1)
bayes_keras <- rBayesianOptimization::BayesianOptimization(FUN = keras_fit, bounds = search_bound_keras, init_points = 0, init_grid_dt = search_grid_keras , n_iter = 5, acq = "ucb",kappa=2.9)

# $Best_Par
# filter1       kernel        learn    batchsize 
# 3.000000e+00 7.000000e+00 7.334392e-03 7.950000e+02 


# Fit the best model
set_random_seed(55)
model <- keras_model_sequential()
model%>%
  layer_conv_1d(filters = bayes_keras$Best_Par[1], kernel_size = bayes_keras$Best_Par[2], strides = 1, input_shape = c(249, 1),padding="same")%>%
  layer_activation_relu()%>%
  layer_batch_normalization()%>%
 # layer_conv_1d(filters = bayes_keras$Best_Par[1],kernel_size = bayes_keras$Best_Par[2], strides = 1,padding="same")%>%
  layer_activation_relu()%>%
  layer_batch_normalization()%>%
  layer_max_pooling_1d(pool_size = bayes_keras$Best_Par[3],padding="same")%>%
  layer_conv_1d(filters=bayes_keras$Best_Par[1]*2,kernel_size=bayes_keras$Best_Par[2], strides = 1,padding="same")%>%
  layer_activation_relu()%>%
  layer_batch_normalization()%>%
#  layer_conv_1d(filters=bayes_keras$Best_Par[1]*2,kernel_size=bayes_keras$Best_Par[2], strides = 1,padding="same")%>%
  layer_activation_relu()%>%
  layer_max_pooling_1d(pool_size = bayes_keras$Best_Par[3],padding="same")%>%
  layer_batch_normalization()%>%
  layer_flatten()%>%
  layer_dense(units = 3, activation = 'softmax')

model %>% compile(
  loss = loss_categorical_crossentropy,
  optimizer = optimizer_adam(bayes_keras$Best_Par[4]),
  metrics = c('accuracy')
)

callbacks <- list( callback_early_stopping(monitor = "val_loss",patience = 25,
                                           min_delta=0.01,restore_best_weights = F))

cnn_history <- model %>% fit(
  x_train_S, dummy_y_train_S,
  batch_size = bayes_keras$Best_Par[5],
  epochs = 100,
  validation_data = list(x_validate_S,dummy_y_validate_S),
  callbacks = callbacks
)


evaluate(model, x_test, dummy_y_test) #63.6 seed 25
preds<-predict(model, x=x_test)

species.predictions<-apply(preds,1,which.max)
species.predictions<-as.factor(ifelse(species.predictions == 1, "LT",
                                      ifelse(species.predictions ==2, "LWF", "SMB")))
confusionMatrix(species.predictions,as.factor(test$species))


# $Best_Par
# filter1       kernel        learn    batchsize 
# 3.000000e+00 7.000000e+00 7.334392e-03 7.950000e+02 

##### ALL Below is trying the feature extractions
set_random_seed(15)
model <- keras_model_sequential()
model%>%
  layer_conv_1d(filters = 9, kernel_size = 9, strides = 1, input_shape = c(249, 1))%>%
  layer_activation_relu()%>%
  layer_batch_normalization()%>%
  layer_conv_1d(filters = 9,kernel_size = 9, strides = 1)%>%
  layer_activation_relu()%>%
  layer_batch_normalization()%>%
  layer_max_pooling_1d(pool_size = c(2))%>%
  layer_conv_1d(filters=11,kernel_size=9, strides = 1)%>%
  layer_activation_relu()%>%
  layer_batch_normalization()%>%
  layer_conv_1d(filters=11,kernel_size=9, strides = 1)%>%
  layer_activation_relu()%>%
  layer_max_pooling_1d(pool_size = 2)%>%
  layer_batch_normalization()%>%
  layer_flatten()%>%
  layer_dense(units = 3, activation = 'softmax')

model %>% compile(
  loss = loss_categorical_crossentropy,
  optimizer = optimizer_adam(0.00385245),
  metrics = c('accuracy')
)

callbacks <- list( callback_early_stopping(monitor = "val_loss",patience = 25,
                                           min_delta=0.01,restore_best_weights = F))

cnn_history <- model %>% fit(
  x_train_S, dummy_y_train_S,
  batch_size = 500,
  epochs = 100,
  validation_split = 0.3,
  callbacks = callbacks
)

evaluate(model, x_test, dummy_y_test) #62.7 seed15

summary(model)


feature_extractor <- keras_model(
  inputs = model$inputs,
  outputs = get_layer(model, index=14)$output,
)

features <- feature_extractor(x_test)



dim(features)
features.array<-as.array(features,dim=c(5355,20,11))

channel1 <- reshape::melt(as.matrix(features.array[,,1]))
colnames(channel1) <- c("data", "feature", "value") 

channel2 <- reshape::melt(as.matrix(features.array[,,2]))
colnames(channel2) <- c("data", "feature", "value") 

channel3 <- reshape::melt(as.matrix(features.array[,,3]))
colnames(channel3) <- c("data", "feature", "value") 

channel4 <- reshape::melt(as.matrix(features.array[,,4]))
colnames(channel4) <- c("data", "feature", "value") 

channel5 <- reshape::melt(as.matrix(features.array[,,5]))
colnames(channel5) <- c("data", "feature", "value") 

channel6 <- reshape::melt(as.matrix(features.array[,,6]))
colnames(channel6) <- c("data", "feature", "value") 

channel7 <- reshape::melt(as.matrix(features.array[,,7]))
colnames(channel7) <- c("data", "feature", "value") 

channel8 <- reshape::melt(as.matrix(features.array[,,8]))
colnames(channel8) <- c("data", "feature", "value") 

channel9 <- reshape::melt(as.matrix(features.array[,,9]))
colnames(channel9) <- c("data", "feature", "value") 

channel10 <- reshape::melt(as.matrix(features.array[,,10]))
colnames(channel10) <- c("data", "feature", "value") 

channel11 <- reshape::melt(as.matrix(features.array[,,11]))
colnames(channel11) <- c("data", "feature", "value") 



ggplot()+
  geom_line(data=channel1,aes(x=feature,y=value,group=data),alpha=0.2)

ggplot()+
  geom_line(data=channel2,aes(x=feature,y=value,group=data),alpha=0.2)

ggplot()+
  geom_line(data=channel3,aes(x=feature,y=value,group=data),alpha=0.2)

ggplot()+
  geom_line(data=channel4,aes(x=feature,y=value,group=data),alpha=0.2)

ggplot()+
  geom_line(data=channel5,aes(x=feature,y=value,group=data),alpha=0.2)

ggplot()+
  geom_line(data=channel6,aes(x=feature,y=value,group=data),alpha=0.2)

ggplot()+
  geom_line(data=channel7,aes(x=feature,y=value,group=data),alpha=0.2)

ggplot()+
  geom_line(data=channel8,aes(x=feature,y=value,group=data),alpha=0.2)

ggplot()+
  geom_line(data=channel9,aes(x=feature,y=value,group=data),alpha=0.2)

ggplot()+
  geom_line(data=channel10,aes(x=feature,y=value,group=data),alpha=0.2)

ggplot()+
  geom_line(data=channel11,aes(x=feature,y=value,group=data),alpha=0.2)



## top 10 in channel 1

max1<-which(features.array[,,1] >= tail(sort(features.array[,,1]), n=100)[1], arr.ind=TRUE)

max1.mm<-features.array[max1[,1],,1]
channel1 <- reshape::melt(max1.mm)
colnames(channel1) <- c("data", "feature", "value") 

ggplot()+
  geom_line(data=channel1,aes(x=feature,y=value,group=data),alpha=0.2)+
  scale_y_continuous(limits=c(-5,45))

## top 10 in channel 2
max2<-which(features.array[,,2] >= tail(sort(features.array[,,2]), n=100)[1], arr.ind=TRUE)

max2.mm<-features.array[max2[,1],,2]
channel2 <- reshape::melt(max2.mm)
colnames(channel2) <- c("data", "feature", "value") 

ggplot()+
  geom_line(data=channel2,aes(x=feature,y=value,group=data),alpha=0.2)+
  scale_y_continuous(limits=c(-5,45))

## top 10 in channel 3
max3<-which(features.array[,,3] >= tail(sort(features.array[,,3]), n=100)[1], arr.ind=TRUE)

max3.mm<-features.array[max3[,1],,3]
channel3 <- reshape::melt(max3.mm)
colnames(channel3) <- c("data", "feature", "value") 

ggplot()+
  geom_line(data=channel3,aes(x=feature,y=value,group=data),alpha=0.2)+
  scale_y_continuous(limits=c(-5,45))

## top 10 in channel 4
max4<-which(features.array[,,4] >= tail(sort(features.array[,,4]), n=100)[1], arr.ind=TRUE)

max4.mm<-features.array[max4[,1],,4]
channel4 <- reshape::melt(max4.mm)
colnames(channel4) <- c("data", "feature", "value") 

ggplot()+
  geom_line(data=channel4,aes(x=feature,y=value,group=data),alpha=0.2)+
  scale_y_continuous(limits=c(-5,45))

## top 10 in channel 5
max5<-which(features.array[,,5] >= tail(sort(features.array[,,5]), n=100)[1], arr.ind=TRUE)

max5.mm<-features.array[max5[,1],,5]
channel5 <- reshape::melt(max5.mm)
colnames(channel5) <- c("data", "feature", "value") 

ggplot()+
  geom_line(data=channel5,aes(x=feature,y=value,group=data),alpha=0.2)+
  scale_y_continuous(limits=c(-5,45))

## top 10 in channel 6
max6<-which(features.array[,,6] >= tail(sort(features.array[,,6]), n=100)[1], arr.ind=TRUE)

max6.mm<-features.array[max6[,1],,6]
channel6 <- reshape::melt(max6.mm)
colnames(channel6) <- c("data", "feature", "value") 

ggplot()+
  geom_line(data=channel6,aes(x=feature,y=value,group=data),alpha=0.2)+
  scale_y_continuous(limits=c(-5,45))

## top 10 in channel 7
max7<-which(features.array[,,7] >= tail(sort(features.array[,,7]), n=100)[1], arr.ind=TRUE)

max7.mm<-features.array[max7[,1],,7]
channel7 <- reshape::melt(max7.mm)
colnames(channel7) <- c("data", "feature", "value") 

ggplot()+
  geom_line(data=channel7,aes(x=feature,y=value,group=data),alpha=0.2)+
  scale_y_continuous(limits=c(-5,45))

## top 10 in channel 8
max8<-which(features.array[,,8] >= tail(sort(features.array[,,8]), n=100)[1], arr.ind=TRUE)

max8.mm<-features.array[max8[,1],,8]
channel8 <- reshape::melt(max8.mm)
colnames(channel8) <- c("data", "feature", "value") 

ggplot()+
  geom_line(data=channel8,aes(x=feature,y=value,group=data),alpha=0.2)+
  scale_y_continuous(limits=c(-5,45))

## top 10 in channel 9
max9<-which(features.array[,,9] >= tail(sort(features.array[,,9]), n=100)[1], arr.ind=TRUE)

max9.mm<-features.array[max9[,1],,9]
channel9 <- reshape::melt(max9.mm)
colnames(channel9) <- c("data", "feature", "value") 

ggplot()+
  geom_line(data=channel9,aes(x=feature,y=value,group=data),alpha=0.2)+
  scale_y_continuous(limits=c(-5,45))

## top 10 in channel 10
max10<-which(features.array[,,10] >= tail(sort(features.array[,,10]), n=100)[1], arr.ind=TRUE)

max10.mm<-features.array[max10[,1],,10]
channel10 <- reshape::melt(max10.mm)
colnames(channel10) <- c("data", "feature", "value") 

ggplot()+
  geom_line(data=channel10,aes(x=feature,y=value,group=data),alpha=0.2)+
  scale_y_continuous(limits=c(-5,45))

## top 10 in channel 11
max11<-which(features.array[,,11] >= tail(sort(features.array[,,11]), n=100)[1], arr.ind=TRUE)

max11.mm<-features.array[max11[,1],,11]
channel11 <- reshape::melt(max11.mm)
colnames(channel11) <- c("data", "feature", "value") 

ggplot()+
  geom_line(data=channel11,aes(x=feature,y=value,group=data),alpha=0.2)+
  scale_y_continuous(limits=c(-5,45))



## Extract weights of the final layer
summary(model)
finalweights<-keras::get_weights(model)[[25]]

LT.weights<-finalweights[,1]
LT.weights<-matrix(LT.weights,nrow=8,byrow = T)  

LWF.weights<-finalweights[,2]
LWF.weights<-matrix(LWF.weights,nrow=8,byrow = T) 

SMB.weights<-finalweights[,3]
SMB.weights<-matrix(SMB.weights,nrow=8,byrow = T) 

heatmap(LT.weights,Rowv=NA,Colv=NA)
heatmap(LWF.weights,Rowv=NA,Colv=NA)
heatmap(SMB.weights,Rowv=NA,Colv=NA)

dim(finalweights)
