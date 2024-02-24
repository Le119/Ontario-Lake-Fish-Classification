library(dplyr)
library(tidyr)
library(keras)
library(rBayesianOptimization)
library(tidymodels)
library(caret)

load("processed_AnalysisData_no200.Rdata")

processed_data<-processed_data_no200

processed_data%>%group_by(spCode,fishNum)%>%count()

# Remove individuals with missing transducers
processed_data<-processed_data%>%filter(is.na(F100)==F)

# also remove individual LWF23018 (only two pings)
processed_data<-processed_data%>%filter(fishNum!="LWF23018")

processed_data<-processed_data%>%select(-F90)
processed_data<-processed_data%>%select(-F90.5)

processed_data<-processed_data%>%filter(spCode == "81"|spCode == "316")

processed_data$species<-ifelse(processed_data$spCode==81, "LT", "SMB")

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
train$y[train$species=="SMB"]<-1
summary(train$y)
dummy_y_train<-to_categorical(train$y, num_classes = 2)

test$y<-NA
test$y[test$species=="LT"]<-0
test$y[test$species=="SMB"]<-1
summary(test$y)
dummy_y_test<-to_categorical(test$y, num_classes = 2)

x_train <- train %>% 
  select(c(52:300))
x_train<-x_train+10*log10(450/train$totalLength)
x_train<-exp(x_train/10)
x_train<-x_train%>%scale()
x_train<-as.matrix(x_train)

xmean<-attributes(x_train)$`scaled:center`
xsd<-attributes(x_train)$`scaled:scale`

x_test <- test %>% 
  select(c(52:300))
x_test<-x_test+10*log10(450/test$totalLength)
x_test<-exp(x_test/10)
x_test<-x_test%>%scale(xmean,xsd)
x_test<-as.matrix(x_test)

## Shuffle training data
set.seed(250)
x<-sample(1:nrow(x_train))
x_train_S= x_train[x, ] 
dummy_y_train_S= dummy_y_train[x, ] 

set.seed(250)
x<-sample(1:nrow(x_test))
x_test_S= x_test[x, ] 
dummy_y_test_S= dummy_y_test[x,] 

# x_train <- array_reshape(x_train_S, c(nrow(x_train_S), 1, 249, 1))
# x_test <- array_reshape(x_test_S, c(nrow(x_test_S), 1, 249, 1))
input_shape <- c(249,1)


#---------------------------------------------

# Define a simple residual block function
simple_res_block <- function(input, filters, kernel_size, stride = 1) {
  # Adjust input shape
  part <- input %>%
    layer_conv_1d(filters = filters, kernel_size = 1, strides = stride, padding = "same") %>%
    layer_batch_normalization()
  
  # First convolutional layer in the block
  part <- part %>%
    layer_conv_1d(filters = filters, kernel_size = kernel_size, padding = "same") %>%
    layer_batch_normalization() %>%
    layer_activation_relu()
  
  # Second convolutional layer in the block
  part <- part %>%
    layer_conv_1d(filters = filters, kernel_size = kernel_size, padding = "same") %>%
    layer_batch_normalization()
  
  # Add the input (residual connection) to the output of the block
  output <- layer_add(list(input, part)) %>%
    layer_activation_relu()

  return(output)
}

# Start building the model
input <- layer_input(shape = c(249, 1))

# Initial convolutional layer
x <- input %>%
  layer_conv_1d(filters = 16, kernel_size = 3, strides = 1, padding = "same") %>%
  layer_activation_relu() %>%
  layer_batch_normalization()

# Add simple residual blocks
x <- simple_res_block(x, filters = 16, kernel_size = 3)
x <- simple_res_block(x, filters = 16, kernel_size = 3)

# Final layers
x <- x %>%
  layer_max_pooling_1d(pool_size = 2) %>%
  layer_flatten() %>%
  layer_dense(units = 2, activation = 'sigmoid')

# Create the model
model <- keras_model(inputs = input, outputs = x)

# Compile the model
model %>% compile(
  loss = loss_categorical_crossentropy,
  optimizer = optimizer_adam(),
  metrics = c('accuracy')
)

# Model summary
summary(model)

#---------------------------------------------------

# Compile model
model %>% compile(
  loss = loss_categorical_crossentropy,
  optimizer = optimizer_adam(),
  metrics = c('accuracy')
)

cnn_history <- model %>% fit(
  x_train_S, dummy_y_train_S,
  batch_size = 1000,
  epochs = 100,
  validation_split = 0.4
)


evaluate(model, x_test, dummy_y_test)
preds<-predict(model, x=x_test)

species.predictions<-apply(preds,1,which.max)
species.predictions<-as.factor(ifelse(species.predictions == 1, "LT", "SMB"))
confusionMatrix(species.predictions,as.factor(test$species))
