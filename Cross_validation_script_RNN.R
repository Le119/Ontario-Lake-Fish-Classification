## Late Trout vs Small mouth bass 

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
library(pROC)

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

#### Train/Test Set Dataset ##
set.seed(73)
split<-group_initial_split(processed_data,group=fishNum,strata = species, prop=0.9)
train<-training(split)
test<-testing(split)

train%>%group_by(species)%>%count() # 2.14 x more LT
test%>%group_by(species)%>%count() # 2.17

train$fishNum<-as.factor(train$fishNum)

# Creating the cross validation groups. 
set.seed(15)
train <- groupdata2::fold(train, k = 5, cat_col = 'species', id_col = 'fishNum')
train$`.folds`<-as.numeric(train$`.folds`)

train<-train%>%select(F45:F170,Region,species,totalLength,.folds)
train[,1:249]<-exp((train[,1:249]+10*log10(450/train$totalLength))/10)

test<-test%>%select(F45:F170,Region,species,totalLength)
test[,1:249]<-exp((test[,1:249]+10*log10(450/test$totalLength))/10)

head(train)
head(test)

# Training Data

# Creating a listing variable within each group so that we can split groups longer than 5 into groups of 5
train_grps<-train%>%group_by(Region)%>%mutate(grp=rep(1:ceiling(n()/5), each=5, length.out=n()))%>%ungroup()
head(train_grps)

# splitting into lists 
listgrps_train<-train_grps%>%group_split(Region,grp)

# keeping only lists that are of length 5
listgrps_train<-listgrps_train[sapply(listgrps_train, nrow) >= 5]

listgrps_train2<-map(listgrps_train, ~ (.x %>% select(c(1:249,253))))

# each dataframe in the list to a matrix
x_data_train<-lapply(listgrps_train2, as.matrix)

# Flatten into a 3D array
x_data_train<-lm2a(x_data_train,dim.order=c(3,1,2))

# Check dims
dim(x_data_train)
# 
# # get the fold for each of the 5795 training data matrices
# listgrps_folds<-map(listgrps_train, ~ (.x %>% select(253)))
# fold<-vector()
# for(i in 1:5795){
#   fold[i]<-mean(as.numeric(listgrps_folds[[i]]$.folds))
# }

# Selecting the y data
y_data_train<-vector()

for(i in 1:dim(x_data_train)[1]){
  a <-listgrps_train[[i]]%>%select(species)
  y_data_train[i]<-a[1,]
}

# Unlist
y_data_train<-unlist(y_data_train)

# Check the number of each class
summary(factor(y_data_train)) 

# Dummy code this
y_train<-NA
y_train[y_data_train=="LT"]<-0
y_train[y_data_train=="SMB"]<-1
summary(y_train)
dummy_y_train<-to_categorical(y_train, num_classes = 2)
dim(dummy_y_train)


# create test data
# Testing Data

# Creating a listing variable within each group so that we can split groups longer than 5 into groups of 5
test_grps<-test%>%group_by(Region)%>%mutate(grp=rep(1:ceiling(n()/5), each=5, length.out=n()))%>%ungroup()
head(test_grps)

# splitting into lists 
listgrps_test<-test_grps%>%group_split(Region,grp)

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

for(i in 1:684){
  a <-listgrps_test[[i]]%>%select(species)
  y_data_test[i]<-a[1,]
}

# Unlist
y_data_test<-unlist(y_data_test)

# Check the number of each class
summary(factor(y_data_test))

# Dummy code
y_test<-NA
y_test[y_data_test=="LT"]<-0
y_test[y_data_test=="SMB"]<-1
summary(y_test)
dummy_y_test<-to_categorical(y_test, num_classes = 2)
dim(dummy_y_test)


#shuffle
set.seed(250)
x<-sample(1:nrow(x_data_train))
x_data_train_S= x_data_train[x,, ]
dummy_y_train_S= dummy_y_train[x, ] 

# add shuffled fold to y data
dummy_y_train_S<-cbind(dummy_y_train_S,(x_data_train_S[,1,250]))

# create grid of parameter space we want to search
regrate<-c(1e-6,1e-5,1e-4)
lstmunits<-c(256,128,64)
neuron1<-c(256,128,64,32,16)

# expand the grid so that every possible combination of the above parameters is present. 
grid.search.full<-expand.grid(regrate=regrate,lstmunits=lstmunits,neuron1=neuron1)

# randomly select 20 of these models to fit. 
set.seed(15)
x<-sample(1:45,20,replace=F)
grid.search.subset<-grid.search.full[x,]

val_loss<-matrix(nrow=20,ncol=5)
best_epoch<-matrix(nrow=20,ncol=5)

for(i in 1:20){
for(fold in 1:5){
  x_train_set<-x_data_train_S[x_data_train_S[,1,250] != fold,,1:249]
  y_train_set<-dummy_y_train_S[dummy_y_train_S[,3]!=fold,1:2]
  
  x_val_set<-x_data_train_S[x_data_train_S[,1,250] == fold,,1:249]
  y_val_set<-dummy_y_train_S[dummy_y_train_S[,3]==fold,1:2]
  
  cw<-summary(as.factor(y_train_set[,1]))[2]/summary(as.factor(y_train_set[,1]))[1]

  set_random_seed(15)
  rnn = keras_model_sequential() # initialize model
  # our input layer
  rnn %>%
    layer_lstm(input_shape=c(5,249),units = lstmunits[i]) %>%
    layer_activation_leaky_relu()%>%
    layer_batch_normalization()%>%
    layer_dense(units=neuron1[i],activity_regularizer = regularizer_l2(regrate[i]))%>%
    layer_activation_leaky_relu()%>%
    layer_dense(units = 2, activation = 'sigmoid')
  
  rnn %>% compile(
    loss = loss_binary_crossentropy,
    optimizer = optimizer_adam(3e-4),
    metrics = c('accuracy')
  )
  
  history <- rnn %>% fit(
    x_train_set, y_train_set,
    batch_size = 250, 
    epochs = 30,
    validation_data = list(x_val_set,y_val_set),
    class_weight = list("0"=1,"1"=cw))
  
  
  val_loss[i,fold]<-min(history$metrics$val_loss)
  best_epoch[i,fold]<-which(history$metrics$val_loss==min(history$metrics$val_loss))
  
  print(i)
  print(fold) 
}}

mean(val_loss)
best_epoch

# find the lowest validation loss 
which(val_loss==min(val_loss),arr.ind = T)
val_loss[5,4]
best_epoch[5,4]
# the overall lowest validation loss was 0.418, but it occurred in epoch 29/30, i.e. the model could have improved more.

# best mean val loss
which(rowMeans(val_loss)==min(rowMeans(val_loss)))
mean(val_loss[7,])
mean(best_epoch[7,])
val_loss[7,]      
best_epoch[7,]

# Link to upload results
# https://utoronto-my.sharepoint.com/:x:/g/personal/jessica_leivesley_utoronto_ca/EW_I626HN9dNvLk8iUoEPRkBxKyMVkkLUD_wwHSU7kqkag?e=P51CZi
