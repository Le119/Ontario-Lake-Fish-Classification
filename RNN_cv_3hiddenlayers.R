# Load in the libraries
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
  select(52:300) # only contains the target strengths
processed_data_TS<-processed_data_TS+10*log10(450/processed_data$totalLength)
processed_data_TS<-exp(processed_data_TS/10)
processed_data_TS<-as.data.frame(processed_data_TS%>%scale())

# adding back necessary/important columns
processed_data_TS$species<-processed_data$species
processed_data_TS$Region<-processed_data$Region
processed_data_TS$fishNum<-processed_data$fishNum

# split into lake trout and bass datasets
trout<-processed_data_TS%>%filter(species=="LT")

# Creating a listing variable within each group so that we can split groups longer than 5 into groups of 5
trout<-trout%>%group_by(Region)%>%mutate(grp=rep(1:ceiling(n()/5), each=5, length.out=n()))%>%ungroup()
head(trout$grp)

# splitting into lists 
listgrps_trout<-trout%>%group_split(Region,grp)

# keeping only lists that are of length 5
listgrps_trout<-listgrps_trout[sapply(listgrps_trout, nrow) >= 5]

# Keeping only the frequency data
listgrps_trout2<-map(listgrps_trout, ~ (.x %>% select(1:249)))

# each dataframe in the list to a matrix
x_trout<-lapply(listgrps_trout2, as.matrix)

# Flatten into a 3D array
x_trout<-lm2a(x_trout,dim.order=c(3,1,2))

# Check dims
dim(x_trout)

# Create y data (0 if LT), in y_trout all is 0
y_trout<-rep(0,dim(x_trout)[1])

# 10% of data in test set, rest in training set
# n_test<-floor(dim(x_trout)[1])/10
n_test<-floor(dim(x_trout)[1]/10)
set.seed(15)
trout_test_index<-sample(1:n_test,replace = F)

trout_test<-x_trout[c(trout_test_index),,]
trout_train<-x_trout[-trout_test_index,,]

y_trout_test<-y_trout[trout_test_index]
y_trout_train<-y_trout[-trout_test_index]

set.seed(15)
folds_trout<-sample(1:5,dim(trout_train)[1],replace=T)

# bass
bass<-processed_data_TS%>%filter(species=="SMB")

# Creating a listing variable within each group so that we can split groups longer than 5 into groups of 5
bass<-bass%>%group_by(Region)%>%mutate(grp=rep(1:ceiling(n()/5), each=5, length.out=n()))%>%ungroup()
head(bass$grp)

# splitting into lists 
listgrps_bass<-bass%>%group_split(Region,grp)

# keeping only lists that are of length 5
listgrps_bass<-listgrps_bass[sapply(listgrps_bass, nrow) >= 5]

# Keeping only the frequency data
listgrps_bass2<-map(listgrps_bass, ~ (.x %>% select(1:249)))

# each dataframe in the list to a matrix
x_bass<-lapply(listgrps_bass2, as.matrix)

# Flatten into a 3D array
x_bass<-lm2a(x_bass,dim.order=c(3,1,2))

# Check dims
dim(x_bass)

# Create y data (1 if LT)
y_bass<-rep(1,dim(x_bass)[1])

# 10% of data in test set, rest in training set
n_test<-floor(dim(x_bass)[1]/10)
set.seed(15)
bass_test_index<-sample(1:n_test,replace = F)

bass_test<-x_bass[c(bass_test_index),,]
bass_train<-x_bass[-bass_test_index,,]

y_bass_test<-y_bass[bass_test_index]
y_bass_train<-y_bass[-bass_test_index]

set.seed(15)
folds_bass<-sample(1:5,dim(bass_train)[1],replace=T)

# Join training dataset and shuffle
x_train<-abind(trout_train,bass_train,along=1)
dim(x_train)

folds<-c(folds_trout,folds_bass)##############

y_train<-c(y_trout_train,y_bass_train)

set.seed(15)
shuffle_index_train<-sample(1:dim(x_train)[1],dim(x_train)[1])
x_train<-x_train[shuffle_index_train,,]
folds<-folds[shuffle_index_train]
y_train<-y_train[shuffle_index_train]

# create dummy y variable
dummy_y_train<-to_categorical(y_train, num_classes = 2)
dim(dummy_y_train)

# create test data
x_test<-abind(trout_test,bass_test,along=1)
dim(x_test)
y_test<-c(y_trout_test,y_bass_test)

# create dummy y variable
dummy_y_test<-to_categorical(y_test, num_classes = 2)
dim(dummy_y_test)

# create grid of parameter space we want to search
regrate<-c(1e-6,1e-5,1e-4)
droprate=c(0,0.1,0.15)
droprate2=c(0,0.1,0.15)
lstmunits<-c(256,128,64)
neuron1<-c(256,128,64)
neuron2=c(64,32,16)
neuron3=c(16,8,4)

# expand the grid so that every possible combination of the above parameters is present. 
# creating every possible combination to test
grid.search.full<-expand.grid(regrate=regrate,droprate=droprate,droprate2=droprate2,
                              lstmunits=lstmunits,neuron1=neuron1,neuron2=neuron2,neuron3=neuron3)

# randomly select 20 of these models to fit. 
set.seed(15)
# x<-sample(1:45,20,replace=F) # 45 = size of grid search (num of row)
x<-sample(1:nrow(grid.search.full),20,replace=F)
grid.search.subset<-grid.search.full[x,]

val_loss<-matrix(nrow=20,ncol=5)
best_epoch<-matrix(nrow=20,ncol=5)

for(i in 1:20){ # go through subset of the different parameter that was randomly selected
  for(fold in 1:5){ # cross-fold validation for each combination
    fold_index<-which(folds==fold)
    x_train_set<-x_train[-fold_index,,]
    y_train_set<-dummy_y_train[-fold_index,]
    
    x_val_set<-x_train[fold_index,,]
    y_val_set<-dummy_y_train[fold_index,]
    
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
      metrics = c('accuracy'))
    
    history <- rnn %>% fit(
      x_train_set, y_train_set,
      batch_size = 1000, 
      epochs = 50,
      validation_data = list(x_val_set,y_val_set),
      class_weight = list("0"=1,"1"=3))
    
    
    val_loss[i,fold]<-min(history$metrics$val_loss)
    best_epoch[i,fold]<-which(history$metrics$val_loss==min(history$metrics$val_loss))
    
    print(i)
    print(fold) 
  }
}

# find the lowest validation loss 
which(val_loss==min(val_loss),arr.ind = T)
lowest_val_loss=which(val_loss==min(val_loss),arr.ind = T) # name so can call on it later
# val_loss[5,4]
# best_epoch[5,4]
# the overall lowest validation loss was 0.418, but it occurred in epoch 29/30, i.e. the model could have improved more.

val_loss[lowest_val_loss] # row num must be what was outputted from prev line that finds the lowest val loss
best_epoch[lowest_val_loss]

# find best mean val loss
which(rowMeans(val_loss)==min(rowMeans(val_loss)))
best_mean_val_loss=which(rowMeans(val_loss)==min(rowMeans(val_loss)))
mean(val_loss[best_mean_val_loss[1],])
mean(best_epoch[best_mean_val_loss[1],])
val_loss[best_mean_val_loss]      
best_epoch[best_mean_val_loss]

# now fit the best model and evaluate test results (once for each fold)
fold=5 # change this to 1,2,3,4,5
fold_index<-which(folds==fold)
x_train_set<-x_train[-fold_index,,]
y_train_set<-dummy_y_train[-fold_index,]

x_val_set<-x_train[fold_index,,]
y_val_set<-dummy_y_train[fold_index,]

# below need to be extracted and inputted as values so only need to change this line everytime we have new optimal values
# 3 hidden layers would need additional neuron3 variable
best_param=tibble(regrate=r, droprate=d, droprate2=d2,
                  lstmunits=u, neuron1=n1, neuron2=n2, neuron3=n3)

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
  layer_dropout(rate = best_param$droprate2)%>%
  layer_dense(units = best_param$neuron3,activity_regularizer = regularizer_l2(l=best_param$regrate)) %>%
  layer_activation_leaky_relu()%>%
  layer_dense(units = 2, activation = 'sigmoid')

rnn %>% compile(
  loss = loss_binary_crossentropy,
  optimizer = optimizer_adam(3e-4),
  metrics = c('accuracy'))

history <- rnn %>% fit(
  x_train_set, y_train_set,
  batch_size = 1000, 
  epochs = 50,
  validation_data = list(x_val_set,y_val_set),
  class_weight = list("0"=1,"1"=3))

# evaluate performance on test data
evaluate(rnn, x_test, dummy_y_test) 

# extract test data classifications
preds<-predict(rnn, x=x_test)

species.predictions<-apply(preds,1,which.max)
species.predictions<-as.factor(ifelse(species.predictions == 1, "0",
                                      "1"))
confusionMatrix(species.predictions,as.factor(y_test))
