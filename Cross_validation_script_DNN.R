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


## Set up the Data ##
load("processed_AnalysisData_no200.Rdata")
processed_data<-processed_data_no200

processed_data%>%group_by(spCode,fishNum)%>%count()

# Remove individuals with missing transducers
processed_data<-processed_data%>%filter(is.na(F100)==F)

processed_data<-processed_data%>%select(-F90)
processed_data<-processed_data%>%select(-F90.5)

processed_data<-processed_data%>%filter(spCode == "81" |spCode == "316")

processed_data$species<-ifelse(processed_data$spCode==81, "LT","SMB")

processed_data<-processed_data%>%filter(is.na(aspectAngle)==F & is.na(Angle_major_axis)==F)

processed_data<-processed_data%>%filter(F100>-1000)

## length overlap?
processed_data%>%group_by(fishNum,species)%>%summarise(length=mean(totalLength))%>%group_by(species)%>%summarise(meanlength=mean(length),sdlength=sd(length),minlength=min(length),maxlength=max(length))

## range overlap?
processed_data%>%group_by(fishNum,species)%>%summarise(TR=mean(Target_range))%>%group_by(species)%>%summarise(meanTR=mean(TR),sdTR=sd(TR),minTR=min(TR),maxTR=max(TR))

processed_data%>%group_by(fishNum,species)%>%summarise(TR=mean(Target_range))%>%
  ggplot()+
  geom_density(aes(x=TR,group=species,fill=species),alpha=0.7)+
  xlab("Target range")

#### Train/Test Set Dataset ##
set.seed(73)
split<-group_initial_split(processed_data,group=fishNum,strata = species, prop=0.85)
train<-training(split)
test<-testing(split)

train%>%group_by(species)%>%count() # 2.14 x more LT
test%>%group_by(species)%>%count() # 2.17

train$fishNum<-as.factor(train$fishNum)

# Creating the cross validation groups. 
set.seed(15)
train.cv <- groupdata2::fold(train, k = 5, cat_col = 'species', id_col = 'fishNum')
train.cv$`.folds`

# Check distribution of the 5 folds
fold1<-train.cv%>%filter(.folds==1)

ggplot()+
  geom_density(data=train.cv,aes(x=exp((F45+10*log10(450/train$totalLength))/10),group=.folds,fill=.folds),alpha=0.2)+
  facet_wrap(~species)

train$y<-NA
train$y[train$species=="LT"]<-0
train$y[train$species=="SMB"]<-1
summary(train$y)
dummy_y_train<-to_categorical(train$y, num_classes = 2)


test$y<-NA
test$y[test$species=="LT"]<-0
test$y[test$species=="SMB"]<-1
summary(test$y)
dummy_y_test<-to_categorical(test$y, num_classes = 2)

x_train <- train %>% 
  select(c(21,23,24,52:300))
x_train[,4:252]<-x_train[,4:252]+10*log10(450/train$totalLength)
x_train[,4:252]<-exp(x_train[,4:252]/10)
x_train<-x_train%>%scale()
x_train<-as.matrix(x_train)

xmean<-attributes(x_train)$`scaled:center`
xsd<-attributes(x_train)$`scaled:scale`

x_test <- test %>% 
  select(c(21,23,24,52:300))
x_test[,4:252]<-x_test[,4:252]+10*log10(450/test$totalLength)
x_test[,4:252]<-exp(x_test[,4:252]/10)
x_test<-x_test%>%scale(xmean,xsd)
x_test<-as.matrix(x_test)

# add folds to x matrix
x_train<-cbind(x_train,train.cv$`.folds`)
dummy_y_train<-cbind(dummy_y_train,train.cv$.folds)

## Shuffle training data
set.seed(250)
x<-sample(1:nrow(x_train))
x_train_S= x_train[x, ] 
dummy_y_train_S= dummy_y_train[x, ] 

## RNN GROUP - here is where I tell the model what parameters I would like it to try. For the 3 hidden layer model, I have three different regularisation rates, 4 different drop outs, and different numbers of neurons for each hidden layer
## three layer random search
regrate<-c(1e-6,1e-5,1e-4)
dropout<-c(0,.1,.15,.2)
neuron1<-c(128,96,64)
neuron2<-c(64,48,32)
neuron3<-c(32,24,16,8)

# expand the grid so that every possible combination of the above parameters is present. 
grid.search.full<-expand.grid(regrate=regrate,dropout=dropout,neuron1=neuron1,neuron2=neuron2,neuron3=neuron3)

# randomly select 20 of these models to fit. 
set.seed(15)
x<-sample(1:324,20,replace=F)
grid.search.subset<-grid.search.full[x,]


val_loss<-matrix(nrow=20,ncol=5)
best_epoch<-matrix(nrow=20,ncol=5)

for(i in 1:20){
for(fold in 1:5){
  x_train_set<-x_train_S[x_train_S[,253] != fold,]
  y_train_set<-dummy_y_train_S[dummy_y_train_S[,3]!=fold,]
  
  x_val_set<-x_train_S[x_train_S[,253] == fold,]
  y_val_set<-dummy_y_train_S[dummy_y_train_S[,3]==fold,]
  
  cw<-summary(as.factor(y_train_set[,1]))[2]/summary(as.factor(y_train_set[,1]))[1]
  ## You will need to edit the model here depending on the number of layers you are fitting
  set_random_seed(15)
  model1 <- keras_model_sequential()
  model1 %>%
    layer_dense(units = grid.search.subset$neuron1[i], input_shape = c(252),activity_regularizer = regularizer_l2(l=grid.search.subset$regrate[i])) %>%
    layer_activation_leaky_relu()%>%
    layer_dropout(grid.search.subset$dropout[i])%>%
    layer_dense(units = grid.search.subset$neuron2[i],activity_regularizer = regularizer_l2(l=grid.search.subset$regrate[i])) %>%
    layer_activation_leaky_relu()%>%
    layer_dropout(grid.search.subset$dropout[i])%>%
    layer_dense(units = grid.search.subset$neuron3[i],activity_regularizer = regularizer_l2(l=grid.search.subset$regrate[i])) %>%
    layer_activation_leaky_relu()%>%
    layer_dense(units = 2, activation = "sigmoid")

  model1 %>% compile(
    loss = 'binary_crossentropy',
    optimizer =  optimizer_adam(3e-4),
    metrics = c('accuracy'))

  history <- model1 %>% fit(
    x_train_set[,c(1:252)], y_train_set[,c(1:2)],
  batch_size = 500, 
  epochs = 200,
  validation_data = list(x_val_set[,c(1:252)],y_val_set[,c(1:2)]),
  class_weight = list("0"=1,"1"=cw))
  
  val_loss[i,fold]<-min(history$metrics$val_loss)
  best_epoch[i,fold]<-which(history$metrics$val_loss==min(history$metrics$val_loss))
  
  print(i)
  print(fold)
}}

# find the lowest validation loss 
which(val_loss==min(val_loss),arr.ind = T)
val_loss[5,4]
best_epoch[5,4]
 # the overall lowest validation loss was 0.418, but it occurred in epoch 29/30, i.e. the model could have improved more.

grid.search.subset[7,]

# best mean val loss
which(rowMeans(val_loss)==min(rowMeans(val_loss)))
mean(val_loss[7,])
mean(best_epoch[7,])
val_loss[7,]      
best_epoch[7,]

# Fit each fold in turn and add the test results to the spreadsheet
fold = 1
  x_train_set<-x_train_S[x_train_S[,253] != fold,]
  y_train_set<-dummy_y_train_S[dummy_y_train_S[,3]!=fold,]
  
  x_val_set<-x_train_S[x_train_S[,253] == fold,]
  y_val_set<-dummy_y_train_S[dummy_y_train_S[,3]==fold,]
  
  cw<-summary(as.factor(y_train_set[,1]))[2]/summary(as.factor(y_train_set[,1]))[1]
  
  set_random_seed(15)
  model1 <- keras_model_sequential()
  model1 %>%
  layer_dense(units = 96, input_shape = c(252),activity_regularizer = regularizer_l2(l=1e-4)) %>%
  layer_activation_leaky_relu()%>%
  layer_dropout(0.2)%>%
  layer_dense(units = 32,activity_regularizer = regularizer_l2(l=1e-4)) %>%
  layer_activation_leaky_relu()%>%
  layer_dropout(0.2)%>%
  layer_dense(units = 24 ,activity_regularizer = regularizer_l2(l=1e-4)) %>%
  layer_activation_leaky_relu()%>%
  layer_dense(units = 2, activation = "sigmoid")
  
  model1 %>% compile(
    loss = 'binary_crossentropy',
    optimizer =  optimizer_adam(3e-4),
  metrics = c('accuracy'))


history <- model1 %>% fit(
  x_train_set[,c(1:252)], y_train_set[,c(1:2)],
  batch_size = 500, 
  epochs = 170,
  validation_data = list(x_val_set[,c(1:252)],y_val_set[,c(1:2)]),
  class_weight = list("0"=1,"1"=2.1))

plot(history)
which(history$metrics$val_loss==min(history$metrics$val_loss))

evaluate(model1, x_test, dummy_y_test) 

preds<-predict(model1, x=x_test)

species.predictions<-apply(preds,1,which.max)
species.predictions<-as.factor(ifelse(species.predictions == 1, "LT",
                                      "SMB"))
confusionMatrix(species.predictions,as.factor(test$species))


pred_fun <- function(mod, X) predict(mod, data.matrix(X), batch_size = 1e4, verbose = FALSE)
ks <- kernelshap(model1, x_train_S[1:100,c(1:252)], bg_X = x_train_S[101:150,c(1:252)], pred_fun = pred_fun)  
shaps <- shapviz(ks)
sv_importance(shaps, show_numbers = TRUE)
sv_importance(shaps, kind = "beeswarm")
imp<-sv_importance(shaps, kind = "no")

frequencies<-rownames(imp)
frequencies<-as.numeric(gsub('F','',frequencies))
imp<-cbind.data.frame(imp,frequencies)
ggplot()+
  geom_line(data=imp,aes(x=frequencies,y=Class_1))+
  xlab("Frequency")+
  ylab("Mean SHAP")+
  theme_classic()+
  ggtitle("Lake Trout")+
  theme(text=element_text(size=16))

ggplot()+
  geom_line(data=imp,aes(x=frequencies,y=Class_2))+
  xlab("Frequency")+
  ylab("Mean SHAP")+
  theme_classic()+
  ggtitle("Smallmouth Bass")+
  theme(text=element_text(size=16))






##----------- Four Hidden Layer Search
regrate<-c(1e-6,1e-5,1e-4)
dropout<-c(0,0.1,0.15,0.2)
neuron1<-c(128,96,64)
neuron2<-c(64,48,32)
neuron3<-c(32,24,16)
neuron4<-c(16,8,4)

grid.search.full<-expand.grid(regrate=regrate,dropout=dropout,neuron1=neuron1,neuron2=neuron2,neuron3=neuron3,neuron4=neuron4)

# randomly select 20 of these models to fit. 
set.seed(15)
x<-sample(1:324,20,replace=F)
grid.search.subset<-grid.search.full[x,]


val_loss<-matrix(nrow=20,ncol=5)
best_epoch<-matrix(nrow=20,ncol=5)

for(i in 1:20){
  for(fold in 1:5){
    x_train_set<-x_train_S[x_train_S[,253] != fold,]
    y_train_set<-dummy_y_train_S[dummy_y_train_S[,3]!=fold,]
    
    x_val_set<-x_train_S[x_train_S[,253] == fold,]
    y_val_set<-dummy_y_train_S[dummy_y_train_S[,3]==fold,]
    
    cw<-summary(as.factor(y_train_set[,1]))[2]/summary(as.factor(y_train_set[,1]))[1]
    
    set_random_seed(15)
    model1 <- keras_model_sequential()
    model1 %>%
      layer_dense(units = grid.search.subset$neuron1[i], input_shape = c(252),activity_regularizer = regularizer_l2(l=grid.search.subset$regrate[i])) %>%
      layer_activation_leaky_relu()%>%
      layer_dropout(grid.search.subset$dropout[i])%>%
      layer_dense(units = grid.search.subset$neuron2[i],activity_regularizer = regularizer_l2(l=grid.search.subset$regrate[i])) %>%
      layer_activation_leaky_relu()%>%
      layer_dropout(grid.search.subset$dropout[i])%>%
      layer_dense(units = grid.search.subset$neuron3[i],activity_regularizer = regularizer_l2(l=grid.search.subset$regrate[i])) %>%
      layer_activation_leaky_relu()%>%
      layer_dropout(grid.search.subset$dropout[i])%>%
      layer_dense(units = grid.search.subset$neuron4[i],activity_regularizer = regularizer_l2(l=grid.search.subset$regrate[i])) %>%
      layer_activation_leaky_relu()%>%
      layer_dense(units = 2, activation = "sigmoid")
    
    model1 %>% compile(
      loss = 'binary_crossentropy',
      optimizer =  optimizer_adam(3e-4),
      metrics = c('accuracy'))
    
    history <- model1 %>% fit(
      x_train_set[,c(1:252)], y_train_set[,c(1:2)],
      batch_size = 500, 
      epochs = 100,
      validation_data = list(x_val_set[,c(1:252)],y_val_set[,c(1:2)]),
      class_weight = list("0"=1,"1"=cw))
    
    val_loss[i,fold]<-min(history$metrics$val_loss)
    best_epoch[i,fold]<-which(history$metrics$val_loss==min(history$metrics$val_loss))
  }}

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

grid.search.subset


fold = 5
x_train_set<-x_train_S[x_train_S[,253] != fold,]
y_train_set<-dummy_y_train_S[dummy_y_train_S[,3]!=fold,]

x_val_set<-x_train_S[x_train_S[,253] == fold,]
y_val_set<-dummy_y_train_S[dummy_y_train_S[,3]==fold,]

cw<-summary(as.factor(y_train_set[,1]))[2]/summary(as.factor(y_train_set[,1]))[1]

set_random_seed(15)
model1 <- keras_model_sequential()
model1 %>%
  layer_dense(units = 128, input_shape = c(252),activity_regularizer = regularizer_l2(l=1e-4)) %>%
  layer_activation_leaky_relu()%>%
  layer_dropout(0.2)%>%
  layer_dense(units = 32,activity_regularizer = regularizer_l2(l=1e-4)) %>%
  layer_activation_leaky_relu()%>%
  layer_dropout(0.2)%>%
  layer_dense(units = 32 ,activity_regularizer = regularizer_l2(l=1e-4)) %>%
  layer_activation_leaky_relu()%>%
  layer_dropout(0.2)%>%
  layer_dense(units = 16 ,activity_regularizer = regularizer_l2(l=1e-4)) %>%
  layer_activation_leaky_relu()%>%
  layer_dense(units = 2, activation = "sigmoid")

model1 %>% compile(
  loss = 'binary_crossentropy',
  optimizer =  optimizer_adam(3e-4),
  metrics = c('accuracy'))


history <- model1 %>% fit(
  x_train_set[,c(1:252)], y_train_set[,c(1:2)],
  batch_size = 500, 
  epochs = 40,
  validation_data = list(x_val_set[,c(1:252)],y_val_set[,c(1:2)]),
  class_weight = list("0"=1,"1"=cw))

plot(history)
which(history$metrics$val_loss==min(history$metrics$val_loss))

evaluate(model1, x_test, dummy_y_test) 

preds<-predict(model1, x=x_test)

species.predictions<-apply(preds,1,which.max)
species.predictions<-as.factor(ifelse(species.predictions == 1, "LT",
                                      "SMB"))
confusionMatrix(species.predictions,as.factor(test$species))
