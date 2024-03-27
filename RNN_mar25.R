library(dplyr)
library(tidymodels)
library(vip)
library(keras)
library(rBayesianOptimization)
library(caret)
library(tensorflow)
library(str2str)
library(pROC)
library(abind)

#### RNN ####
# Load the data
load("processed_AnalysisData_no200.Rdata")

# make the name easier to type
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

# transform and standardize data
processed_data_TS <- processed_data %>% 
  select(52:300)
processed_data_TS<-processed_data_TS+10*log10(450/processed_data$totalLength)
processed_data_TS<-exp(processed_data_TS/10)
processed_data_TS<-as.data.frame(processed_data_TS%>%scale())

# adding back necessary/important columns
processed_data_TS$species<-processed_data$species
processed_data_TS$Region<-processed_data$Region
processed_data_TS$fishNum<-processed_data$fishNum


# 10% of data in test set, rest in training set
# group by fishNum makes sure same fish doesn't appear in both testing and training
set.seed(15)
split<-group_initial_split(processed_data_TS,group=fishNum,strata = species, prop=0.9)
train<-training(split)
test<-testing(split)


### Training Data ###

# Creating a listing variable within each group so that we can split groups longer than 5 into groups of 5
train_grps<-train%>%group_by(Region)%>%mutate(grp=rep(1:ceiling(n()/5), each=5, length.out=n()))%>%ungroup()
head(train_grps)

# splitting into lists 
listgrps_train<-train_grps%>%group_split(Region,grp)

# keeping only lists that are of length 5
listgrps_train<-listgrps_train[sapply(listgrps_train, nrow) >= 5]

# select frequencies only
listgrps_train2<-map(listgrps_train, ~ (.x %>% select(c(1:249))))

# each dataframe in the list to a matrix
x_data_train<-lapply(listgrps_train2, as.matrix)

# Flatten into a 3D array
x_data_train<-lm2a(x_data_train,dim.order=c(3,1,2))

# Check dims
dim(x_data_train)

# Selecting the y data
y_data_train<-vector()

for(i in 1:dim(x_data_train)[1]){
  a <-listgrps_train[[i]]%>%select(species)
  y_data_train[i]<-a[1,]
}

# Unlist
y_data_train<-unlist(y_data_train)

# Dummy code this
y_train<-NA
y_train[y_data_train=="LT"]<-0
y_train[y_data_train=="SMB"]<-1
summary(y_train)
dummy_y_train<-to_categorical(y_train, num_classes = 2)
dim(dummy_y_train)



## Getting fish ID so that it is not repeated across folds
fishID_train<-vector()

for(i in 1:dim(x_data_train)[1]){
  a <-listgrps_train[[i]]%>%select(fishNum)
  fishID_train[i]<-a[1,]
}

# Unlist
fishID_train<-unlist(fishID_train)


### Testing Data ###

# Creating a listing variable within each group so that we can split groups longer than 5 into groups of 5
test_grps<-test%>%group_by(Region)%>%mutate(grp=rep(1:ceiling(n()/5), each=5, length.out=n()))%>%ungroup()
head(test_grps)

# splitting into lists 
listgrps_test<-test_grps%>%group_split(Region,grp)

# keeping only lists that are of length 5
listgrps_test<-listgrps_test[sapply(listgrps_test, nrow) >= 5]

# keeping only frequencies
listgrps_test2<-map(listgrps_test, ~ (.x %>% select(c(1:249))))

# each data frame in the list to a matrix
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

# Dummy code this
y_test<-NA
y_test[y_data_test=="LT"]<-0
y_test[y_data_test=="SMB"]<-1
summary(y_test)
dummy_y_test<-to_categorical(y_test, num_classes = 2)
dim(dummy_y_test)



## shuffle data ##
set.seed(15)
shuffle_index_train<-sample(1:dim(x_data_train)[1],dim(x_data_train)[1])
x_data_train<-x_data_train[shuffle_index_train,,]
dummy_y_train<-dummy_y_train[shuffle_index_train,]
fishID_train<-fishID_train[shuffle_index_train]


# K fold validation, but with no repeat of fish across, preventing data leakage
set.seed(15)
folds<-groupKFold(fishID_train,k=5)

# create grid of parameter space we want to search
regrate<-c(1e-6,1e-5,1e-4)
droprate=c(0,0.1,0.15)
droprate2=c(0,0.1,0.15) # only needed in 3 hidden layer if want different drop rates
lstmunits<-c(256,128,64)
neuron1<-c(256,128,64)
neuron2=c(64,32,16)
neuron3=c(16,8,4) # only needed in 3 hidden layers

# expand the grid so that every possible combination of the above parameters is present. 
# creating every possible combination to test
grid.search.full<-expand.grid(regrate=regrate,droprate=droprate,droprate2=droprate2,lstmunits=lstmunits,neuron1=neuron1,neuron2=neuron2,neuron3=neuron3)

# randomly select 20 of these models to fit. 
set.seed(15)
# x<-sample(1:45,20,replace=F) # 45 = size of grid search (num of row)
x<-sample(1:nrow(grid.search.full),20,replace=F)
grid.search.subset<-grid.search.full[x,]

val_loss<-matrix(nrow=20,ncol=5)
best_epoch_loss<-matrix(nrow=20,ncol=5)
val_auc<-matrix(nrow=20,ncol=5)
best_epoch_auc<-matrix(nrow=20,ncol=5)

## finding the optimal parameters ##
for(i in 1:20){ # go through subset of the different parameter that was randomly selected
  for(fold in 1:5){ # cross-fold validation for each combination
    x_train_set<-x_data_train[folds[[fold]],,]
    y_train_set<-dummy_y_train[folds[[fold]],]
    
    x_val_set<-x_data_train[-folds[[fold]],,]
    y_val_set<-dummy_y_train[-folds[[fold]],]
    
    set_random_seed(15)
    rnn = keras_model_sequential() # initialize model
    # our input layer
    rnn %>%
      layer_lstm(input_shape=c(5,249),units = grid.search.subset$lstmunits[i]) %>%
      layer_activation_leaky_relu()%>%
      layer_batch_normalization()%>%
      layer_dense(units = grid.search.subset$neuron1[i],activity_regularizer = regularizer_l2(l=grid.search.subset$regrate[i])) %>%
      layer_activation_leaky_relu()%>%
      layer_dropout(rate = grid.search.subset$droprate[i])%>%
      layer_dense(units = grid.search.subset$neuron2[i],activity_regularizer = regularizer_l2(l=grid.search.subset$regrate[i])) %>%
      layer_activation_leaky_relu()%>%
      layer_dropout(rate = grid.search.subset$droprate2[i])%>%
      layer_dense(units = grid.search.subset$neuron3[i],activity_regularizer = regularizer_l2(l=grid.search.subset$regrate[i])) %>%
      layer_activation_leaky_relu()%>%
      layer_dense(units = 2, activation = 'sigmoid')
    
    rnn %>% compile(
      loss = loss_binary_crossentropy,
      optimizer = optimizer_adam(3e-4),
      metrics = c('accuracy', tf$keras$metrics$AUC()))
    
    history <- rnn %>% fit(
      x_train_set, y_train_set,
      batch_size = 1000, 
      epochs = 50,
      validation_data = list(x_val_set,y_val_set),
      class_weight = list("0"=1,"1"=2))
    
    
    val_loss[i,fold]<-min(history$metrics$val_loss)
    best_epoch_loss[i,fold]<-which(history$metrics$val_loss==min(history$metrics$val_loss))[1]
    val_auc[i,fold]<-max(history$metrics$val_auc)
    best_epoch_auc[i,fold]<-which(history$metrics$val_auc==max(history$metrics$val_auc))
    print(i)
    print(fold) 
  }
}

## find best parameters ##

# Using validation loss

# find the lowest validation loss (will not be used if using AUC)
which(val_loss==min(val_loss),arr.ind = T)
lowest_val_loss=which(val_loss==min(val_loss),arr.ind = T) # name so can call on it later
# val_loss[5,4]
# best_epoch[5,4]
# the overall lowest validation loss was 0.418, but it occurred in epoch 29/30, i.e. the model could have improved more.

val_loss[lowest_val_loss] # index must be what was outputted from prev line that finds the lowest val loss
best_epoch_loss[lowest_val_loss]

# find best mean val loss
which(rowMeans(val_loss)==min(rowMeans(val_loss)))
best_mean_val_loss=which(rowMeans(val_loss)==min(rowMeans(val_loss)))
mean(val_loss[best_mean_val_loss[1],])
mean(best_epoch_loss[best_mean_val_loss[1],])
val_loss[best_mean_val_loss]      
best_epoch_loss[best_mean_val_loss]


# Using validation AUC

# find highest validation AUC
highest_val_auc=which(val_auc==max(val_auc),arr.ind = T) # name so can call on it later
val_auc[highest_val_auc] # index must be what was outputted from prev line that finds the lowest val loss
best_epoch_auc[highest_val_auc]

# find best mean val auc
best_mean_val_auc=which(rowMeans(val_auc)==max(rowMeans(val_auc)))
mean(val_auc[best_mean_val_auc[1],])
mean(best_epoch_auc[best_mean_val_auc[1],])
val_auc[best_mean_val_auc]      
best_epoch_auc[best_mean_val_auc]


## training the best model ##
# now fit the best model and evaluate test results (once for each fold)
fold=5 # change this to 1,2,3,4,5 for each fold
x_train_set<-x_data_train[folds[[fold]],,]
y_train_set<-dummy_y_train[folds[[fold]],]

x_val_set<-x_data_train[-folds[[fold]],,]
y_val_set<-dummy_y_train[-folds[[fold]],]


# below need to be extracted and inputted as values so only need to change this line every time we have new optimal values
# 3 hidden layers would need additional neuron3 (and droprate2 if want) variable
best_param=tibble(regrate=0.00001, droprate=0.15, droprate2=0.15,
                  lstmunits=256, neuron1=256, neuron2=16,neuron3=8)

set_random_seed(15)
rnn = keras_model_sequential() # initialize model
# our input layer
rnn %>%
  layer_lstm(input_shape=c(5,249),units = best_param$lstmunits) %>%
  layer_activation_leaky_relu()%>%
  layer_batch_normalization()%>%
  layer_dense(units = best_param$neuron1,activity_regularizer = regularizer_l2(l=best_param$regrate)) %>%
  layer_activation_leaky_relu()%>%
  layer_dropout(rate = best_param$droprate)%>%
  layer_dense(units = best_param$neuron2,activity_regularizer = regularizer_l2(l=best_param$regrate)) %>%
  layer_activation_leaky_relu()%>%
  layer_dense(units = 2, activation = 'sigmoid')

rnn %>% compile(
  loss = loss_binary_crossentropy,
  optimizer = optimizer_adam(3e-4),
  metrics = c('accuracy', tf$keras$metrics$AUC()))

history <- rnn %>% fit(
  x_train_set, y_train_set,
  batch_size = 1000, 
  epochs = 50,
  validation_data = list(x_val_set,y_val_set),
  class_weight = list("0"=1,"1"=2))

## evaluating the best model on test data ##
# evaluate performance on test data
evaluate(rnn, x_data_test, dummy_y_test) 

# extract test data classifications
preds<-predict(rnn, x=x_data_test)

species.predictions<-apply(preds,1,which.max)
species.predictions<-as.factor(ifelse(species.predictions == 1, "0",
                                      "1"))
confusionMatrix(species.predictions,as.factor(y_test))
