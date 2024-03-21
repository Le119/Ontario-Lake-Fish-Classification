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
split <- group_initial_split(processed_data, group = fishNum, strata = species, prop = 0.7)
train <- training(split)
val_test <- testing(split)
split2 <- group_initial_split(val_test, group = fishNum, strata = species, prop = 0.5)
validate <- training(split2)
test <- testing(split2)

train%>%group_by(species)%>%count()
validate%>%group_by(species)%>%count()
test%>%group_by(species)%>%count()

train <- slice_sample(train, n = 6378, by = species)
validate <- slice_sample(validate, n = 1382, by = species)
test <- slice_sample(test, n = 1374, by = species)

train%>%group_by(species)%>%count()
validate%>%group_by(species)%>%count()
test%>%group_by(species)%>%count()

train$y <- ifelse(train$species == "LT", 0, 1)
dummy_y_train <- to_categorical(train$y, num_classes = 2)
test$y <- ifelse(test$species == "LT", 0, 1)
dummy_y_test <- to_categorical(test$y, num_classes = 2)
validate$y <- ifelse(validate$species == "LT", 0, 1)
dummy_y_val <- to_categorical(validate$y, num_classes = 2)


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

x_validate <- validate %>% select(52:300)
x_validate<-x_validate+10*log10(450/validate$totalLength)
x_validate<-exp(x_validate/10)
x_validate<-x_validate%>%scale(xmean,xsd)
x_validate<-as.matrix(x_validate)

# Shuffle training data
set.seed(250)
train_indices <- sample(1:nrow(x_train))
x_train <- x_train[train_indices, ] 
dummy_y_train <- dummy_y_train[train_indices, ] 

set.seed(250)
val_indices <- sample(1:nrow(x_validate))
x_validate <- x_validate[val_indices, ] 
dummy_y_val <- dummy_y_val[val_indices, ]


input_shape <- c(249,1)
set_random_seed(15)
inputs <- layer_input(shape = input_shape)

block_1_output <- inputs %>%
  layer_conv_1d(filters = 16, kernel_size = 3, activation = "relu", padding = "same", strides = 1)

# Adjust block_2_output to include the first convolutional layer for block_2
block_2_prep <- block_1_output %>%
  layer_conv_1d(16, 3, activation = "relu", padding = "same")

block_2_output <- block_2_prep %>%
  layer_conv_1d(16, 3, activation = "relu", padding = "same") %>%
  layer_add(block_1_output)

# Introduce a skip from block_1_output to block_3_output
block_3_output <- block_2_output %>%
  layer_conv_1d(16, 3, activation = "relu", padding = "same") %>%
  layer_conv_1d(16, 3, activation = "relu", padding = "same") %>%
  layer_add(block_2_output) %>%
  layer_add(block_1_output) # Adding block_1_output as a skip to block_3_output

# Continue from block_3_output to block_4_output
block_4_output <- block_3_output %>%
  layer_conv_1d(16, 3, activation = "relu", padding = "same") %>%
  layer_conv_1d(16, 3, activation = "relu", padding = "same") %>%
  layer_add(block_3_output)

# Introduce a skip from block_3_output to block_5_output
block_5_output <- block_4_output %>%
  layer_conv_1d(16, 3, activation = "relu", padding = "same") %>%
  layer_conv_1d(16, 3, activation = "relu", padding = "same") %>%
  layer_add(block_4_output) %>%
  layer_add(block_3_output) # Adding block_3_output as a skip to block_5_output

outputs <- block_5_output %>%
  layer_conv_1d(16, 3, activation = "relu", padding = "same") %>%
  layer_conv_1d(16, 3, activation = "relu", padding = "same") %>%
  layer_max_pooling_1d(pool_size = 2) %>%
  layer_flatten() %>%
  layer_dense(2, activation="sigmoid")

model <- keras_model(inputs, outputs)
model
#plot(model,show_shapes = T)

model %>% compile(
  optimizer = optimizer_adam(),
  loss = loss_categorical_crossentropy,
  metrics = c("accuracy")
)

# Fit model (just resnet)
resnet_history <- model %>% fit(
  x_train,dummy_y_train,
  batch_size = 1000,
  epochs = 100,
  validation_data = list(x_validate, dummy_y_val)
)

plot(resnet_history)
evaluate(model, (x_test), dummy_y_test)
preds<-predict(model, x=x_test)

species.predictions<-apply(preds,1,which.max)
species.predictions<-as.factor(ifelse(species.predictions == 1, "LT", "SMB"))
confusionMatrix(species.predictions,as.factor(test$species))
