### updates since mar14/2024
### data leakage problem fixed by grouping using fishNum instead of Region_name


# Load in the libraries
library(dplyr)
library(tidymodels)
library(vip)
library(keras)
library(rBayesianOptimization)
library(caret)
library(tensorflow)
library(kernelshap)
library(shapviz)
library(str2str)


###### Data cleaning ######
# Load the data
load("processed_AnalysisData_no200.Rdata")

# Make the name easier to type
processed_data<-processed_data_no200

# Look at the structure of individuals
processed_data%>%group_by(spCode,fishNum,Region_name)%>%dplyr::count()

# Create unique region per fish
processed_data$Region<-interaction(processed_data$fishNum,processed_data$Region_name)

# Remove individuals with missing transducers
processed_data<-processed_data%>%filter(is.na(F100)==F)

# also remove individual LWF23018 (only two pings)
processed_data<-processed_data%>%filter(fishNum!="LWF23018")

# remove the 90kHZ and 90.5kHZ columns
processed_data<-processed_data%>%select(-F90)
processed_data<-processed_data%>%select(-F90.5)

# only keep LT (81), and SMB (316)
processed_data<-processed_data%>%filter(spCode == "81" |spCode == "316")

processed_data$species<-ifelse(processed_data$spCode==81, "LT", 
                               ifelse(processed_data$spCode == 91, "LWF", "SMB"))

# remove the one ping that has a VERY low TS
processed_data<-processed_data%>%filter(F100>-1000)
glimpse(processed_data)



###### Training/test data split ######
set.seed(490)
split<-group_initial_split(processed_data,group=fishNum,strata = species, prop=0.65)
train<-training(split)
val_test<-testing(split)
split2<-group_initial_split(val_test,group=fishNum,strata = species, prop=0.5)
validate<-training(split2)
test<-testing(split2)

train%>%group_by(species)%>%dplyr::count()
validate%>%group_by(species)%>%dplyr::count()
test%>%group_by(species)%>%dplyr::count()

train<-train[order(train$species),]
validate<-validate[order(validate$species),]
test<-test[order(test$species),]

train<-train%>%select(F45:F170,Region_name,species,totalLength)
train[,1:249]<-exp((train[,1:249]+10*log10(450/train$totalLength))/10)

validate<-validate%>%select(F45:F170,Region_name,species,totalLength)
validate[,1:249]<-exp((validate[,1:249]+10*log10(450/validate$totalLength))/10)

test<-test%>%select(F45:F170,Region_name,species,totalLength)
test[,1:249]<-exp((test[,1:249]+10*log10(450/test$totalLength))/10)

head(train)
head(validate)
head(test)


###### Training data ######

# Creating a listing variable within each group so that we can split groups longer than 5 into groups of 5
train_grps<-train%>%group_by(Region_name)%>%mutate(grp=rep(1:ceiling(n()/5), each=5, length.out=n()))%>%ungroup()
head(train_grps)

# splitting into lists 
listgrps_train<-train_grps%>%group_split(Region_name,grp)

# keeping only lists that are of length 5
listgrps_train<-listgrps_train[sapply(listgrps_train, nrow) >= 5]

listgrps_train2<-map(listgrps_train, ~ (.x %>% select(1:249)))

# each dataframe in the list to a matrix
x_data_train<-lapply(listgrps_train2, as.matrix)

# Flatten into a 3D array
x_data_train<-lm2a(x_data_train,dim.order=c(3,1,2))

# Check dims
dim(x_data_train)

# Selecting the y data training
y_data_train<-vector()

for(i in 1:dim(x_data_train)[1]){
  a <-listgrps_train[[i]]%>%select(species)
  y_data_train[i]<-a[1,]
}

# Unlist
y_data_train<-unlist(y_data_train)

# Balance the classes
summary(factor(y_data_train)) 

y_train<-NA
y_train[y_data_train=="LT"]<-0
y_train[y_data_train=="SMB"]<-1
summary(y_train)
dummy_y_train<-to_categorical(y_train, num_classes = 2)
dim(dummy_y_train)


###### Validating Data ######

# Creating a listing variable within each group so that we can split groups longer than 5 into groups of 5
validate_grps<-validate%>%group_by(Region_name)%>%mutate(grp=rep(1:ceiling(n()/5), each=5, length.out=n()))%>%ungroup()
head(validate_grps)

# splitting into lists 
listgrps_validate<-validate_grps%>%group_split(Region_name,grp)

# keeping only lists that are of length 5
listgrps_validate<-listgrps_validate[sapply(listgrps_validate, nrow) >= 5]

listgrps_validate2<-map(listgrps_validate, ~ (.x %>% select(1:249)))

# each dataframe in the list to a matrix
x_data_validate<-lapply(listgrps_validate2, as.matrix)

# Flatten into a 3D array
x_data_validate<-lm2a(x_data_validate,dim.order=c(3,1,2))

# Check dims
dim(x_data_validate)

# Selecting the y data
y_data_validate<-vector()

for(i in 1:dim(x_data_validate)[1]){
  a <-listgrps_validate[[i]]%>%select(species)
  y_data_validate[i]<-a[1,]
}

# Unlist
y_data_validate<-unlist(y_data_validate)

# Balance the classes
summary(factor(y_data_validate)) 

y_validate<-NA
y_validate[y_data_validate=="LT"]<-0
y_validate[y_data_validate=="SMB"]<-1
summary(y_validate)
dummy_y_validate<-to_categorical(y_validate, num_classes = 2)
dim(dummy_y_validate)


###### Testing Data ######

# Creating a listing variable within each group so that we can split groups longer than 5 into groups of 5
test_grps<-test%>%group_by(Region_name)%>%mutate(grp=rep(1:ceiling(n()/5), each=5, length.out=n()))%>%ungroup()
head(test_grps)

# splitting into lists 
listgrps_test<-test_grps%>%group_split(Region_name,grp)

# keeping only lists that are of length 5
listgrps_test<-listgrps_test[sapply(listgrps_test, nrow) >= 5]

listgrps_test2<-map(listgrps_test, ~ (.x %>% select(1:249)))

# each dataframe in the list to a matrix
x_data_test<-lapply(listgrps_test2, as.matrix)

# Flatten into a 3D array
x_data_test<-lm2a(x_data_test,dim.order=c(3,1,2))

# Check dims
dim(x_data_test)

# Selecting the y data
y_data_test<-vector()

for(i in 1:dim(x_data_test)[1]){
  a <-listgrps_test[[i]]%>%select(species)
  y_data_test[i]<-a[1,]
}

# Unlist
y_data_test<-unlist(y_data_test)

# Balance the classes
summary(factor(y_data_test)) 

y_test<-NA
y_test[y_data_test=="LT"]<-0
y_test[y_data_test=="SMB"]<-1
summary(y_test)
dummy_y_test<-to_categorical(y_test, num_classes = 2)
dim(dummy_y_test)


####### shuffle ######
set.seed(250)
x<-sample(1:nrow(x_data_train))
x_data_train_S= x_data_train[x,, ] 
dummy_y_train_S= dummy_y_train[x, ] 

set.seed(250)
x<-sample(1:nrow(x_data_validate))
x_data_validate_S= x_data_validate[x,, ] 
dummy_y_validate_S= dummy_y_validate[x, ]



# -----------------------Model Fit Structure-----------------------
set_random_seed(8)
rnn = keras_model_sequential() # initialize model
## our input layer
rnn %>%
  layer_lstm(input_shape=c(5,249), units = 249, activation = "relu",
             return_sequences = TRUE) %>% # rnn layer, input_shape = # timepoints, units can > frequencies
  layer_lstm(units = 200, activation = "relu") %>%
  layer_dense(units = 130, activation = "relu", activity_regularizer =  regularizer_l2(0.01)) %>%
  layer_dropout(rate = 0.25) %>%
  layer_dense(units = 60, activation = "relu", activity_regularizer =  regularizer_l2(0.01)) %>%
  layer_dropout(rate = 0.25) %>%
  layer_dense(units = 15, activation = "relu", activity_regularizer =  regularizer_l2(0.01)) %>%
  layer_dropout(rate = 0.25) %>%
  layer_dense(units = 2, activation = 'sigmoid')
# look at our model architecture
summary(rnn)
rnn %>% compile(
  loss = loss_binary_crossentropy,
  optimizer = optimizer_adam(),
  metrics = c('accuracy', tf$keras$metrics$AUC())
)

# will go through all and keep the optimal value through restore_best_weights
# patience = how long it run until find optimal
callbacks <- list(callback_early_stopping(monitor = "val_loss",patience = 25,min_delta=0.01,restore_best_weights = T))

history <- rnn %>% fit(
  x_data_train_S, dummy_y_train_S,
  batch_size = 1000,# need to be optimized
  epochs = 5, # need to be increased
  validation_data = list(x_data_validate_S,dummy_y_validate_S),
  callbacks = callbacks,
  class_weight = list("0"=1,"1"=2)) # deal with unbalanced data set

# plot(history)
evaluate(rnn, x_data_test, dummy_y_test)
# l1= 0.005, l2= 0.005 = 0.70
# now try add in some drop out.

preds<-predict(rnn, x=x_data_test)

species.predictions<-apply(preds,1,which.max)
species.predictions<-as.factor(ifelse(species.predictions == 1, "LT",
                                      "SMB"))
confusionMatrix(species.predictions,as.factor(y_data_test))


# -----------------------Optimization-----------------------
## one hidden layer optimization------------------------
## Bayesian optimization, run the model 50 times, optimizing units and neurons

# Try more:
# --different regularization L1, L2
# --increase num layer
# --recurrent_regularization in lstm layer

# ------------------------Starts Below-----------------------


# to run over and over again, build, compile, run history
keras_fit <- function(units1,neuron1,neuron2){
  #,neuron3,neuron4
  set_random_seed(15)
  model <- keras_model_sequential()
  model %>%
    layer_lstm(input_shape=c(5,249),units = units1,activation = "relu") %>%
    layer_dense(units=neuron1,activation = "relu")%>%
    layer_dense(units=neuron2,activation = "relu")%>%
    # layer_dense(units=neuron3,activation = "relu")%>%
    # layer_dense(units=neuron4,activation = "relu")%>%
    layer_dense(units = 2, activation = 'sigmoid') 
  
  
  model %>% compile(
    loss = loss_binary_crossentropy,
    optimizer = optimizer_adam(),
    metrics = c('accuracy', tf$keras$metrics$AUC())
  )
  
  callbacks <- list(callback_early_stopping(monitor = "val_loss",patience = 25,min_delta=0.001,restore_best_weights = T))
  
  history <- model %>% fit(
    x_data_train_S, dummy_y_train_S,
    batch_size = 1000, 
    epochs = 300, # change to higher, e.g. 300
    validation_data = list(x_data_validate_S,dummy_y_validate_S),
    callbacks = callbacks,
    class_weight = list("0"=1,"1"=2))
  
  
  result <- list(Score = max(history$metrics[[6]]), 
                 Pred = 0)
  
  return(result)
  
}

# do this for any parameter you want to optimize
search_bound_keras <- list(units1=c(2L,500L),
                           neuron1=c(249L,500L),
                           neuron2=c(2L,249L))
# ,
# neuron2=c(2L,500L),
# neuron3=c(2L,500L),
# neuron4=c(2L,500L)

# the more parameter you want to optimize the more...
search_grid_keras <- data.frame(units1=floor(runif(5,2,500)),
                                neuron1=floor(runif(5,249,500)),
                                neuron2=floor(runif(5,2,249)))
# ,
# neuron2=floor(runif(5,2,500)),
# neuron3=floor(runif(5,2,500)),
# neuron4=floor(runif(5,2,500))
head(search_grid_keras)


bayes_opt_rnn <- rBayesianOptimization::BayesianOptimization(FUN = keras_fit, bounds = search_bound_keras, init_points = 0, init_grid_dt = search_grid_keras, n_iter = 5, acq = "ucb")
# n_iter = how many times you run bayesoptim, 10 is too much
# acq = "ei


# ggplot()+
#   geom_point(data=bayes_opt_rnn$History,aes(x=neuron1,y=units1,col=Value))

# Pulling the best parameter from the Bayesian optimization
set_random_seed(15)
model <- keras_model_sequential()
model %>%
  layer_lstm(input_shape=c(5,249),units = bayes_opt_rnn$Best_Par[1],
             activation = "relu") %>%
  layer_dense(units=bayes_opt_rnn$Best_Par[2],activation = "relu")%>%
  layer_dense(units=bayes_opt_rnn$Best_Par[3],activation = "relu")%>%
  # layer_dense(units=bayes_opt_rnn$Best_Par[4],activation = "relu")%>%
  # layer_dense(units=bayes_opt_rnn$Best_Par[5],activation = "relu")%>%
  layer_dense(units = 2, activation = 'sigmoid') 

model %>% compile(
  loss = loss_binary_crossentropy,
  optimizer = optimizer_adam(),
  metrics = c('accuracy',tf$keras$metrics$AUC())
)
callbacks <- list(callback_early_stopping(monitor = "val_loss",patience = 25,min_delta=0.001,restore_best_weights = T))

history <- model %>% fit(
  x_data_train_S, dummy_y_train_S,
  batch_size = 1000, # need to be optimized
  epochs = 300, # need increase, 5 just testing, maybe 300
  validation_data = list(x_data_validate_S,dummy_y_validate_S),
  callbacks = callbacks,
  class_weight = list("0"=1,"1"=2))



plot(history)
evaluate(model, x_data_test, dummy_y_test) 

preds<-predict(model, x=x_data_test)

species.predictions<-apply(preds,1,which.max)
species.predictions<-as.factor(ifelse(species.predictions == 1, "LT",
                                      "SMB"))
confusionMatrix(species.predictions,as.factor(y_data_test))
# units = 224 neuron 1 = 249; balanced acc = 0.70; AUC = 0.826

# units=459 neuron1=500 acc=0.7212714 auc=0.7486744
# +activation_regularizer_l2: units=411 neuron1=10 reg1=0.1 acc=0.7057865 AUC=0.7288269
# +1dense_layer: units=479 neuron1=379 neuron2=325 acc=0.7057865 auc=0.7411598
# +2dense_layer: units=2 neuron1=466 neuron2=197 neuron3=129 acc=0.7131214 auc=0.7367606
# 0.7082314 0.7341782 (decreasing output dim in each layer)
# 0.7090465 0.7330543 (found neuron1, units1 first)
# +3dense_layer: units=184 neuron1=416 neuron2=431 neuron3=388 acc=0.7098615 0.7414647 