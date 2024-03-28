## Jess ResNet, based on Lewei & Rita's work

## Load in the necessary libraries
library(dplyr)
library(tidyr)
library(keras)
library(rBayesianOptimization)
library(tidymodels)
library(caret)
library(tensorflow)
library(torchvision)
library(magrittr) # conditional piping with %>%


load("processed_AnalysisData_no200.Rdata")

processed_data <- processed_data_no200

processed_data%>%group_by(spCode,fishNum)%>%count()

processed_data <- processed_data %>% filter(is.na(F100) == F)
processed_data <- processed_data %>% filter(fishNum != "LWF23018")
processed_data <- processed_data %>% select(-F90, -F90.5)
processed_data <- processed_data %>% filter(spCode == "81" | spCode == "316")
processed_data$species <- ifelse(processed_data$spCode == 81, "LT", "SMB")
processed_data <- processed_data %>% filter(is.na(aspectAngle) == F & is.na(Angle_major_axis) == F & is.na(Angle_minor_axis) == F)
processed_data <- processed_data %>% filter(F100 > -1000)

set.seed(73)
split <- group_initial_split(processed_data, group = fishNum, strata = species, prop = 0.9)
train <- training(split)
test <- testing(split)

train%>%group_by(species)%>%count()
test%>%group_by(species)%>%count()

train <- slice_sample(train, n = 8533, by = species)
test <- slice_sample(test, n = 601, by = species)

train%>%group_by(species)%>%count()
test%>%group_by(species)%>%count()

# Shuffle data
set.seed(250)
train_folds <- groupKFold(train$fishNum,k=5)
train_indices <- sample(1:nrow(train))
train <- train[train_indices, ] 

set.seed(250)
test_indices <- sample(1:nrow(test))
test <- test[test_indices, ] 

train$y <- ifelse(train$species == "LT", 0, 1)
dummy_y_train <- to_categorical(train$y, num_classes = 2)
test$y <- ifelse(test$species == "LT", 0, 1)
dummy_y_test <- to_categorical(test$y, num_classes = 2)


x_train <- train %>% select(52:300)
x_train<-x_train+10*log10(450/train$totalLength)
x_train<-exp(x_train/10)
x_train<-x_train%>%scale()
x_train<-as.matrix(x_train)

xmean<-attributes(x_train)$`scaled:center`
xsd<-attributes(x_train)$`scaled:scale`
x_test <- test %>% select(52:300)
x_test<-x_test+10*log10(450/test$totalLength)
x_test<-exp(x_test/10)
x_test<-x_test%>%scale(xmean,xsd)
x_test<-as.matrix(x_test)

# functions for adding layers conditionally

conv_activation_layer <- function(input_layer, filters, kernel_size, leaky_relu) {
  if (!leaky_relu) {
    # conv layer with ReLU activation
    output_layer <- input_layer %>% 
      layer_conv_1d(filters = filters, kernel_size = kernel_size, 
                    activation = 'relu', padding = 'same', strides = 1)
  } else {
    # conv layer followed by Leaky ReLU
    output_layer <- input_layer %>% 
      layer_conv_1d(filters = filters, kernel_size = kernel_size, 
                    padding = 'same', strides = 1) %>%
      layer_activation_leaky_relu()
  }
  return(output_layer)
}

add_batch_normalization <- function(input_layer, batch_normalization) {
  if (batch_normalization) {
    output_layer <- input_layer %>% layer_batch_normalization()
  } else {
    output_layer <- input_layer
  }
  return(output_layer)
}


# create grid of parameter space we want to search
filters <- c(8, 16, 32)
kernel_size <- c(3, 5, 7)
leaky_relu <- c(T, F)
batch_normalization <- c(T, F)

# expand the grid so that every possible combination of the above parameters is present. 
# creating every possible combination to test
grid.search.full<-expand.grid(filters = filters, kernel_size = kernel_size, 
                              leaky_relu = leaky_relu, 
                              batch_normalization = batch_normalization)

set.seed(15)
# x<-sample(1:45,20,replace=F) # 45 = size of grid search (num of row)
x<-sample(1:nrow(grid.search.full),20,replace=F)
grid.search.subset<-grid.search.full[x,]

### need to change the grid and implement in for loop

val_loss<-matrix(nrow=20,ncol=5)
best_epoch<-matrix(nrow=20,ncol=5)


for (i in 1:20){
  for (fold in 1:5){
    x_train_set <- x_train[train_folds[[fold]],]
    y_train_set <- dummy_y_train[train_folds[[fold]],]
    
    x_val_set<-x_train[-train_folds[[fold]],]
    y_val_set<-dummy_y_train[-train_folds[[fold]],]
    
    input_shape <- c(249,1)
    set_random_seed(15)
    inputs <- layer_input(shape = input_shape)
    
    block_1_output <- inputs %>%
      conv_activation_layer(filters = grid.search.subset$filters[i], 
                            kernel_size = grid.search.subset$kernel_size[i], 
                            leaky_relu = grid.search.subset$leaky_relu[i]) # self-defined function with optional leaky relu (or relu)
    
    # Adjust block_2_output to include the first convolutional layer for block_2
    block_2_prep <- block_1_output %>%
      conv_activation_layer(filters = grid.search.subset$filters[i], 
                            kernel_size = grid.search.subset$kernel_size[i], 
                            leaky_relu = grid.search.subset$leaky_relu[i]) %>%
      add_batch_normalization(batch_normalization = grid.search.subset$batch_normalization[i])
    
    block_2_output <- block_2_prep %>%
      conv_activation_layer(filters = grid.search.subset$filters[i], 
                            kernel_size = grid.search.subset$kernel_size[i], 
                            leaky_relu = grid.search.subset$leaky_relu[i]) %>%
      add_batch_normalization(batch_normalization = grid.search.subset$batch_normalization[i]) %>%
      layer_add(block_1_output)
    
    # Introduce a skip from block_1_output to block_3_output
    block_3_output <- block_2_output %>%
      conv_activation_layer(filters = grid.search.subset$filters[i], 
                            kernel_size = grid.search.subset$kernel_size[i], 
                            leaky_relu = grid.search.subset$leaky_relu[i]) %>%
      conv_activation_layer(filters = grid.search.subset$filters[i], 
                            kernel_size = grid.search.subset$kernel_size[i], 
                            leaky_relu = grid.search.subset$leaky_relu[i]) %>%
      add_batch_normalization(batch_normalization = grid.search.subset$batch_normalization[i]) %>%
      layer_add(block_2_output) %>%
      layer_add(block_1_output) # Adding block_1_output as a skip to block_3_output
    
    # Continue from block_3_output to block_4_output
    block_4_output <- block_3_output %>%
      conv_activation_layer(filters = grid.search.subset$filters[i], 
                            kernel_size = grid.search.subset$kernel_size[i], 
                            leaky_relu = grid.search.subset$leaky_relu[i]) %>%
      conv_activation_layer(filters = grid.search.subset$filters[i], 
                            kernel_size = grid.search.subset$kernel_size[i], 
                            leaky_relu = grid.search.subset$leaky_relu[i]) %>%
      add_batch_normalization(batch_normalization = grid.search.subset$batch_normalization[i]) %>%
      layer_add(block_3_output)
    
    # Introduce a skip from block_3_output to block_5_output
    block_5_output <- block_4_output %>%
      conv_activation_layer(filters = grid.search.subset$filters[i], 
                            kernel_size = grid.search.subset$kernel_size[i], 
                            leaky_relu = grid.search.subset$leaky_relu[i]) %>%
      conv_activation_layer(filters = grid.search.subset$filters[i], 
                            kernel_size = grid.search.subset$kernel_size[i], 
                            leaky_relu = grid.search.subset$leaky_relu[i]) %>%
      add_batch_normalization(batch_normalization = grid.search.subset$batch_normalization[i]) %>%
      layer_add(block_4_output) %>%
      layer_add(block_3_output) # Adding block_3_output as a skip to block_5_output
    
    outputs <- block_5_output %>%
      conv_activation_layer(filters = grid.search.subset$filters[i], 
                            kernel_size = grid.search.subset$kernel_size[i], 
                            leaky_relu = grid.search.subset$leaky_relu[i]) %>%
      conv_activation_layer(filters = grid.search.subset$filters[i], 
                            kernel_size = grid.search.subset$kernel_size[i], 
                            leaky_relu = grid.search.subset$leaky_relu[i]) %>%
      add_batch_normalization(batch_normalization = grid.search.subset$batch_normalization[i]) %>%
      layer_max_pooling_1d(pool_size = 2) %>%
      layer_flatten() %>%
      layer_dense(2, activation="sigmoid")
    
    model <- keras_model(inputs, outputs)
    #model
    #plot(model,show_shapes = T)
    
    model %>% compile(
      optimizer = optimizer_adam(),
      loss = loss_categorical_crossentropy,
      metrics = c("accuracy")
    )
    
    # Fit model (just resnet)
    resnet_history <- model %>% fit(
      x_train_set, y_train_set,
      batch_size = 1000,
      epochs = 75,
      validation_data = list(x_val_set, y_val_set)
    )
    
    val_loss[i,fold]<-min(resnet_history$metrics$val_loss)
    best_epoch[i,fold]<-which(resnet_history$metrics$val_loss==min(resnet_history$metrics$val_loss))
    
    print(i)
    print(fold)
    
  }
}

print(val_loss)
print(best_epoch)

fold = 1
fold_index <- which(train$folds == fold)
x_train_set <- x_train[-fold_index,]
y_train_set <- dummy_y_train[-fold_index,]

x_val_set <- x_train[fold_index,]
y_val_set <- dummy_y_train[fold_index,]

# run up to here

# below need to be extracted and inputted as values so only need to change this line everytime we have new optimal values
best_param=tibble(filters = 16, kernel_size = 3, leaky_relu = T, batch_normalization = T)

### NEED TO add a function with if else structure for adding conditional layers for leaky and bn

input_shape <- c(249,1)
set_random_seed(15)
inputs <- layer_input(shape = input_shape)

block_1_output <- inputs %>%
  conv_activation_layer(filters = best_param$filters, 
                        kernel_size = best_param$kernel_size, 
                        leaky_relu = best_param$leaky_relu)

# Adjust block_2_output to include the first convolutional layer for block_2
block_2_prep <- block_1_output %>%
  conv_activation_layer(filters = best_param$filters, 
                        kernel_size = best_param$kernel_size, 
                        leaky_relu = best_param$leaky_relu) %>%
  add_batch_normalization(batch_normalization = best_param$batch_normalization)

block_2_output <- block_2_prep %>%
  conv_activation_layer(filters = best_param$filters, 
                        kernel_size = best_param$kernel_size, 
                        leaky_relu = best_param$leaky_relu) %>%
  add_batch_normalization(batch_normalization = best_param$batch_normalization) %>%
  layer_add(block_1_output)

# Introduce a skip from block_1_output to block_3_output
block_3_output <- block_2_output %>%
  conv_activation_layer(filters = best_param$filters, 
                        kernel_size = best_param$kernel_size, 
                        leaky_relu = best_param$leaky_relu) %>%
  conv_activation_layer(filters = best_param$filters, 
                        kernel_size = best_param$kernel_size, 
                        leaky_relu = best_param$leaky_relu) %>%
  add_batch_normalization(batch_normalization = best_param$batch_normalization) %>%
  layer_add(block_2_output) %>%
  layer_add(block_1_output) # Adding block_1_output as a skip to block_3_output

# Continue from block_3_output to block_4_output
block_4_output <- block_3_output %>%
  conv_activation_layer(filters = best_param$filters, 
                        kernel_size = best_param$kernel_size, 
                        leaky_relu = best_param$leaky_relu) %>%
  conv_activation_layer(filters = best_param$filters, 
                        kernel_size = best_param$kernel_size, 
                        leaky_relu = best_param$leaky_relu) %>%
  add_batch_normalization(batch_normalization = best_param$batch_normalization) %>%
  layer_add(block_3_output)

# Introduce a skip from block_3_output to block_5_output
block_5_output <- block_4_output %>%
  conv_activation_layer(filters = best_param$filters, 
                        kernel_size = best_param$kernel_size, 
                        leaky_relu = best_param$leaky_relu) %>%
  conv_activation_layer(filters = best_param$filters, 
                        kernel_size = best_param$kernel_size, 
                        leaky_relu = best_param$leaky_relu) %>%
  add_batch_normalization(batch_normalization = best_param$batch_normalization) %>%
  layer_add(block_4_output) %>%
  layer_add(block_3_output) # Adding block_3_output as a skip to block_5_output

outputs <- block_5_output %>%
  conv_activation_layer(filters = best_param$filters, 
                        kernel_size = best_param$kernel_size, 
                        leaky_relu = best_param$leaky_relu) %>%
  conv_activation_layer(filters = best_param$filters, 
                        kernel_size = best_param$kernel_size, 
                        leaky_relu = best_param$leaky_relu) %>%
  add_batch_normalization(batch_normalization = best_param$batch_normalization) %>%
  layer_max_pooling_1d(pool_size = 2) %>%
  layer_flatten() %>%
  layer_dense(2, activation="sigmoid")

model <- keras_model(inputs, outputs)
#model
#plot(model,show_shapes = T)

model %>% compile(
  optimizer = optimizer_adam(),
  loss = loss_categorical_crossentropy,
  metrics = c("accuracy")
)

# Fit model (just resnet)
resnet_history <- model %>% fit(
  x_train_set, y_train_set,
  batch_size = 1000,
  epochs = 75,
  validation_data = list(x_val_set, y_val_set)
)



plot(resnet_history)
evaluate(model, (x_test), dummy_y_test)
preds<-predict(model, x=x_test)

species.predictions<-apply(preds,1,which.max)
species.predictions<-as.factor(ifelse(species.predictions == 1, "LT", "SMB"))
confusionMatrix(species.predictions,as.factor(test$species))

