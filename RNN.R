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

#### Train/Test Set####
set.seed(73)
split<-group_initial_split(processed_data,group=Region,strata = species, prop=0.65)
train<-training(split)
val_test<-testing(split)
split2<-group_initial_split(val_test,group=Region,strata = species, prop=0.65)
validate<-training(split2)
test<-testing(split2)

train%>%group_by(species)%>%dplyr::count()
validate%>%group_by(species)%>%dplyr::count()
test%>%group_by(species)%>%dplyr::count()

train<-train[order(train$species),]
validate<-validate[order(validate$species),]
test<-test[order(test$species),]

train<-train%>%select(F45:F170,Region,species,totalLength)
train[,1:249]<-exp((train[,1:249]+10*log10(450/train$totalLength))/10)

validate<-validate%>%select(F45:F170,Region,species,totalLength)
validate[,1:249]<-exp((validate[,1:249]+10*log10(450/validate$totalLength))/10)

test<-test%>%select(F45:F170,Region,species,totalLength)
test[,1:249]<-exp((test[,1:249]+10*log10(450/test$totalLength))/10)

head(train)
head(validate)
head(test)


# Training Data

# Creating a listing variable within each group so that we can split groups longer than 5 into groups of 5
train_grps<-train%>%group_by(Region)%>%mutate(grp=rep(1:ceiling(n()/5), each=5, length.out=n()))%>%ungroup()
head(train_grps)

# splitting into lists 
listgrps_train<-train_grps%>%group_split(Region,grp)

# keeping only lists that are of length 5
listgrps_train<-listgrps_train[sapply(listgrps_train, nrow) >= 5]

# ## Balance the classes
# ls_sp<-vector()
# for(i in 1:4493){
# ls_sp[i]<-listgrps_train[[i]][1,251]}
# 
# ls_sp<-unlist(ls_sp)
# 
# listgrps_train<-listgrps_train[order(c(ls_sp))]
# summary(as.factor(ls_sp))
# # 3042 LT and 1451 SMB, so need to remove 1591 LT
# set.seed(5)
# rem<-sample(c(rep(0,1591),rep(1,1451)),3042)
# rem2<-rep(1,1451)
# 
# keep<-which(c(rem,rem2)==1)
# 
# listgrps_train<-listgrps_train[c(keep)]
listgrps_train2<-map(listgrps_train, ~ (.x %>% select(1:249)))

# each dataframe in the list to a matrix
x_data_train<-lapply(listgrps_train2, as.matrix)

# Flatten into a 3D array
x_data_train<-lm2a(x_data_train,dim.order=c(3,1,2))

# Check dims
dim(x_data_train)


# Selecting the y data
y_data_train<-vector()

for(i in 1:4493){
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

# Validating Data

# Creating a listing variable within each group so that we can split groups longer than 5 into groups of 5
validate_grps<-validate%>%group_by(Region)%>%mutate(grp=rep(1:ceiling(n()/5), each=5, length.out=n()))%>%ungroup()
head(validate_grps)

# splitting into lists 
listgrps_validate<-validate_grps%>%group_split(Region,grp)

# keeping only lists that are of length 5
listgrps_validate<-listgrps_validate[sapply(listgrps_validate, nrow) >= 5]

# ## Balance the classes
# ls_sp<-vector()
# for(i in 1:1585){
#   ls_sp[i]<-listgrps_validate[[i]][1,251]}
# 
# ls_sp<-unlist(ls_sp)
# 
# listgrps_validate<-listgrps_validate[order(c(ls_sp))]
# 
# summary(factor(ls_sp))
# # 1068 LT and 517 SMB, so need to remove 551 LT
# set.seed(5)
# rem<-sample(c(rep(0,551),rep(1,517)),1068)
# rem2<-rep(1,517)
# 
# keep<-which(c(rem,rem2)==1)
# 
# listgrps_validate<-listgrps_validate[c(keep)]
listgrps_validate2<-map(listgrps_validate, ~ (.x %>% select(1:249)))

# each dataframe in the list to a matrix
x_data_validate<-lapply(listgrps_validate2, as.matrix)

# Flatten into a 3D array
x_data_validate<-lm2a(x_data_validate,dim.order=c(3,1,2))

# Check dims
dim(x_data_validate)


# Selecting the y data
y_data_validate<-vector()

for(i in 1:1585){
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

# Testing Data

# Creating a listing variable within each group so that we can split groups longer than 5 into groups of 5
test_grps<-test%>%group_by(Region)%>%mutate(grp=rep(1:ceiling(n()/5), each=5, length.out=n()))%>%ungroup()
head(test_grps)

# splitting into lists 
listgrps_test<-test_grps%>%group_split(Region,grp)

# keeping only lists that are of length 5
listgrps_test<-listgrps_test[sapply(listgrps_test, nrow) >= 5]

# ## Balance the classes
# ls_sp<-vector()
# for(i in 1:831){
#   ls_sp[i]<-listgrps_test[[i]][1,251]}
# 
# ls_sp<-unlist(ls_sp)
# 
# listgrps_test<-listgrps_test[order(c(ls_sp))]
# 
# summary(factor(ls_sp))
# # 555 LT and 276 SMB, so need to remove 279 LT
# set.seed(5)
# rem<-sample(c(rep(0,279),rep(1,276)),555)
# rem2<-rep(1,276)
# 
# keep<-which(c(rem,rem2)==1)
# 
# listgrps_test<-listgrps_test[c(keep)]

listgrps_test2<-map(listgrps_test, ~ (.x %>% select(1:249)))

# each dataframe in the list to a matrix
x_data_test<-lapply(listgrps_test2, as.matrix)

# Flatten into a 3D array
x_data_test<-lm2a(x_data_test,dim.order=c(3,1,2))

# Check dims
dim(x_data_test)


# Selecting the y data
y_data_test<-vector()

for(i in 1:831){
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

#shuffle
set.seed(250)
x<-sample(1:nrow(x_data_train))
x_data_train_S= x_data_train[x,, ] 
dummy_y_train_S= dummy_y_train[x, ] 

set.seed(250)
x<-sample(1:nrow(x_data_validate))
x_data_validate_S= x_data_validate[x,, ] 
dummy_y_validate_S= dummy_y_validate[x, ] 



set_random_seed(32)
rnn = keras_model_sequential() # initialize model
# our input layer
rnn %>%
  layer_lstm(input_shape=c(5,249),units = 249,activation = "relu") %>%
  layer_dense(units=150)%>%
  layer_dense(units=75)%>%
  layer_dense(units=38)%>%
  layer_dense(units=19)%>%
  layer_dense(units = 2, activation = 'sigmoid')
# look at our model architecture
summary(rnn)
rnn %>% compile(
  loss = loss_binary_crossentropy,
  optimizer = optimizer_adam(),
  metrics = c('accuracy', tf$keras$metrics$AUC())
)

callbacks <- list(callback_early_stopping(monitor = "val_loss",patience = 25,min_delta=0.01,restore_best_weights = T))

history <- rnn %>% fit(
  x_data_train_S, dummy_y_train_S,
  batch_size = 1000, 
  epochs = 5,
  validation_data = list(x_data_validate_S,dummy_y_validate_S),
  callbacks = callbacks,
  class_weight = list("0"=1,"1"=2))

plot(history)
evaluate(rnn, x_data_test, dummy_y_test) 
# l1= 0.005, l2= 0.005 = 0.70 
# now try add in some drop out. 

preds<-predict(rnn, x=x_data_test)

species.predictions<-apply(preds,1,which.max)
species.predictions<-as.factor(ifelse(species.predictions == 1, "LT",
                                      "SMB"))
confusionMatrix(species.predictions,as.factor(y_data_test))

# one hidden layer opt ------------------------
keras_fit <- function(units1,neuron1){
  set_random_seed(15)
  model <- keras_model_sequential()
  model %>%
    layer_lstm(input_shape=c(5,249),units = units1,activation = "relu") %>%
    layer_dense(units=neuron1)%>%
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
    epochs = 300,
    validation_data = list(x_data_validate_S,dummy_y_validate_S),
    callbacks = callbacks,
    class_weight = list("0"=1,"1"=2))
  
  
  result <- list(Score = max(history$metrics[[6]]), 
                 Pred = 0)
  
  return(result)
  
}



search_bound_keras <- list(units1=c(2L,500L),
                           neuron1=c(2L,500L))


search_grid_keras <- data.frame(units1=floor(runif(5,2,500)),
                                neuron1=floor(runif(5,2,500)))

head(search_grid_keras)

bayes_opt_rnn <- rBayesianOptimization::BayesianOptimization(FUN = keras_fit, bounds = search_bound_keras, init_points = 0, init_grid_dt = search_grid_keras , n_iter = 10, acq = "ucb")

ggplot()+
  geom_point(data=bayes_opt_rnn$History,aes(x=neuron1,y=units1,col=Value))

set_random_seed(15)
model <- keras_model_sequential()
model %>%
  layer_lstm(input_shape=c(5,249),units = bayes_opt_rnn$Best_Par[1],activation = "relu") %>%
  layer_dense(units=bayes_opt_rnn$Best_Par[2])%>%
  layer_dense(units = 2, activation = 'sigmoid') 

model %>% compile(
  loss = loss_binary_crossentropy,
  optimizer = optimizer_adam(),
  metrics = c('accuracy',tf$keras$metrics$AUC())
)
callbacks <- list(callback_early_stopping(monitor = "val_loss",patience = 25,min_delta=0.001,restore_best_weights = T))

history <- model %>% fit(
  x_data_train_S, dummy_y_train_S,
  batch_size = 1000, 
  epochs = 300,
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
