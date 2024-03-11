library(dplyr)
library(tidyr)
library(keras)
library(rBayesianOptimization)
library(tidymodels)
library(caret)

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

# Prepare the metadata for input
metadata_train <- train %>% select(Angle_minor_axis, Angle_major_axis, aspectAngle) %>% as.matrix()
metadata_test <- test %>% select(Angle_minor_axis, Angle_major_axis, aspectAngle) %>% as.matrix()
metadata_val <- validate %>% select(Angle_minor_axis, Angle_major_axis, aspectAngle) %>% as.matrix()

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

x_val <- validate %>% select(52:300)
x_val<-x_val+10*log10(450/validate$totalLength)
x_val<-exp(x_val/10)
x_val<-x_val%>%scale(xmean,xsd)
x_val<-as.matrix(x_val)



# Shuffle training data
set.seed(250)
train_indices <- sample(1:nrow(x_train))
x_train <- x_train[train_indices, ] 
metadata_train <- metadata_train[train_indices, ]
dummy_y_train <- dummy_y_train[train_indices, ] 

set.seed(250)
val_indices <- sample(1:nrow(x_val))
x_val <- x_val[val_indices, ] 
metadata_val <- metadata_val[val_indices, ]
dummy_y_val <- dummy_y_val[val_indices, ]

set.seed(250)
test_indices <- sample(1:nrow(x_test))
x_test <- x_test[test_indices, ] 
metadata_test <- metadata_test[test_indices, ]
dummy_y_test <- dummy_y_test[test_indices, ]

input_shape <- c(249, 1)

# Define a simple residual block function
simple_res_block <- function(input, filters, kernel_size, stride = 1) {
  part <- input %>%
    layer_conv_1d(filters = filters, kernel_size = kernel_size, strides = stride, padding = "same") %>%
    layer_batch_normalization() %>%
    layer_conv_1d(filters = filters, kernel_size = kernel_size, strides = stride, padding = "same") %>%
    layer_batch_normalization() %>%
    layer_activation_relu() %>%
    layer_conv_1d(filters = filters, kernel_size = kernel_size, strides = stride, padding = "same") %>%
    layer_batch_normalization()
  output <- layer_add(list(input, part)) %>%
    layer_activation_relu()
  return(output)
}

# Input layers for CNN and metadata
input_cnn <- layer_input(shape = input_shape)
input_meta <- layer_input(shape = c(3))

# ResNet layers
x <- input_cnn %>%
  layer_conv_1d(filters = 16, kernel_size = 3, strides = 1, padding = "same") %>%
  layer_activation_relu() %>%
  layer_batch_normalization() %>%
  simple_res_block(filters = 16, kernel_size = 3) %>%
  simple_res_block(filters = 16, kernel_size = 3) %>%
  simple_res_block(filters = 16, kernel_size = 3) %>%
  simple_res_block(filters = 16, kernel_size = 3) %>%
  layer_max_pooling_1d(pool_size = 2) %>%
  layer_flatten()

# Combine ResNet output with metadata
combined <- layer_concatenate(list(x, input_meta)) %>%
  layer_dense(units = 1024, activation = 'relu') %>%
  layer_dense(units = 1024, activation = 'relu') %>%
  layer_dense(units = 1024, activation = 'relu')

output <- combined %>%
  layer_dense(units = 2, activation = 'sigmoid')

model <- keras_model(inputs = list(input_cnn, input_meta), outputs = output)

summary(model)

model %>% compile(
  loss = loss_categorical_crossentropy,
  optimizer = optimizer_adam(),
  metrics = c('accuracy')
)

# Fit model with metadata
cnn_history <- model %>% fit(
  list(x_train, metadata_train), dummy_y_train,
  batch_size = 1000,
  epochs = 100,
  validation_data = list(x_val, metadata_val) ####
)

# Evaluate and predict with metadata
evaluate(model, list(x_test, metadata_test), dummy_y_test)
preds <- predict(model, list(x_test, metadata_test))

species_predictions <- apply(preds, 1, which.max) - 1
species_predictions <- ifelse(species_predictions == 0, "LT", "SMB")
confusionMatrix(factor(species_predictions), factor(test$species))