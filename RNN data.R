load("~/Downloads/processed_AnalysisData_no200.Rdata")
# ls()
# libraries
library(dplyr)
library(tidyr)
library(str2str)
library(keras)
library(caret)
library(lubridate)
library(tidymodels)
library(vip)
library(rBayesianOptimization)
library(tensorflow)


## Custom Function
split_at_gap <- function(data, max_gap = 60, shortest_track = 0) {
  # Number of tracks
  n_tracks <- length(unique(data$ID))
  
  # Save old ID and reinitialise ID column
  data$ID_old <- as.factor(data$ID)
  data$ID <- character(nrow(data))
  
  # Loop over tracks (i.e., over IDs)
  for(i_track in 1:n_tracks) {
    # Indices for this track
    ind_this_track <- which(data$ID_old == unique(data$ID_old)[i_track])
    track_length <- length(ind_this_track)
    
    # Time intervals in min
    dtimes <- difftime(data$time[ind_this_track[-1]], 
                       data$time[ind_this_track[-track_length]],
                       units = "secs")
    
    # Indices of gaps longer than max_gap
    ind_gap <- c(0, which(dtimes > max_gap), track_length)
    
    # Create new ID based on split track
    subtrack_ID <- rep(1:(length(ind_gap) - 1), diff(ind_gap))
    data$ID[ind_this_track] <- paste0(data$ID_old[ind_this_track], "-", subtrack_ID)
  }
  
  # Only keep sub-tracks longer than some duration
  track_lengths <- sapply(unique(data$ID), function(id) {
    ind <- which(data$ID == id)
    difftime(data$time[ind[length(ind)]], data$time[ind[1]], units = "sec")
  })
  ID_keep <- names(track_lengths)[which(track_lengths >= shortest_track)]
  data <- subset(data, ID %in% ID_keep)
  
  return(data)
}

# Load the data
load("processed_AnalysisData_no200.Rdata")

# make the name easier to type
processed_data<-processed_data_no200

# Look at the structure of individuals
processed_data%>%group_by(spCode,fishNum)%>%count()

# Remove individuals with missing transducers
processed_data<-processed_data%>%filter(is.na(F100)==F)

# also remove individual LWF23018 (only two pings)
processed_data<-processed_data%>%filter(fishNum!="LWF23018")

# remove the 90kHZ and 90.5kHZ columns
processed_data<-processed_data%>%select(-F90)
processed_data<-processed_data%>%select(-F90.5)

# only keep LT (81), LWF (91), and SMB (316)
processed_data<-processed_data%>%filter(spCode == "81" |spCode == "91"|spCode == "316")

processed_data$species<-ifelse(processed_data$spCode==81, "LT", 
                               ifelse(processed_data$spCode == 91, "LWF", "SMB"))

# remove the one ping that has a VERY low TS
processed_data<-processed_data%>%filter(F100>-1000)

# function for converting time
convert_to_datetime <- function(time_str) {
  return(as.POSIXct(time_str, format = "%H:%M:%OS"))
}

# Modify the Ping_time column to wanted format
processed_data <- processed_data %>%
  mutate(Ping_time = convert_to_datetime(Ping_time))

#### Train/Test Set####
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


### Train Data
# Group together and summarise any pings that occur less than 0.5secs apart. 
data_0.5sec_train <- train%>%
  select(fishNum,Ping_time,F45:F170,species)%>% #,F45:F170
  mutate(ID= fishNum) %>% 
  group_by(ID,time = floor_date(Ping_time, unit = "0.5 sec")) %>%
  summarise(across(everything(), list(max))) %>%
  ungroup() 

data_0.5sec_train[,5:253]<-exp(data_0.5sec_train[,5:253]/10)

# any data that occurs consecutively is given the same grouping. If there is a 0.5sec gap then it becomes a new group. I've specified here there shortest track should be 5.
data_0.5sec_train_grps<-split_at_gap(data = data_0.5sec_train, 
             max_gap = 0.5, 
             shortest_track = 5)


# Creating a listing variable within each group so that we can split groups longer than 5 into groups of 5
data_0.5sec_train_grps<-data_0.5sec_train_grps%>%group_by(ID)%>%mutate(grp=rep(1:ceiling(n()/5), each=5, length.out=n()))%>%ungroup()

# splitting into lists 
listgrps_train<-data_0.5sec_train_grps%>%group_split(ID,grp)

# keeping only lists that are of length 5
listgrps_train<-listgrps_train[sapply(listgrps_train, nrow) >= 5]

# Selecting the x data
x_data_train<-list()

for(i in 1:1497){
  x_data_train[[i]]<-listgrps_train[[i]]%>%select(F45_1:F170_1)
}

# each dataframe in the list to a matrix
x_data_train<-lapply(x_data_train, as.matrix)

# Flatten into a 3D array
x_data_train<-lm2a(x_data_train,dim.order=c(3,1,2))

# Check dims
dim(x_data_train)

# Selecting the y data
y_data_train<-vector()

for(i in 1:1497){
a <-listgrps_train[[i]]%>%select(species_1)
y_data_train[i]<-a[1,]
}

# Unlist
y_data_train<-unlist(y_data_train)

# Balance the classes
summary(factor(y_data_train))

# Balance the classes
summary(factor(y_data_train)) #need 277 

# remove 535 LT & 131 LWF
set.seed(5)
rem<-sample(c(rep(0,535),rep(1,277)),812)
rem2<-sample(c(rep(0,131),rep(1,277)),408)
remove<-which(c(rem,rem2)==0)
x_data_train<-x_data_train[-c(remove),,]
y_data_train<-y_data_train[-remove]

y_train<-NA
y_train[y_data_train=="LT"]<-0
y_train[y_data_train=="LWF"]<-1
y_train[y_data_train=="SMB"]<-2
summary(y_train)
dummy_y_train<-to_categorical(y_train, num_classes = 3)

# Shuffle data
set.seed(250)
x<-sample(1:nrow(x_data_train))
x_data_train_S= x_data_train[x, ,] 
dummy_y_train_S= dummy_y_train[x, ] 

summary(as.factor(y_data_train))

### validate Data
# Group together and summarise any pings that occur less than 0.5secs apart. 
data_0.5sec_validate <- validate%>%
  select(fishNum,Ping_time,F45:F170,species)%>% #,F45:F170
  mutate(ID= fishNum) %>% 
  group_by(ID,time = floor_date(Ping_time, unit = "0.5 sec")) %>%
  summarise(across(everything(), list(max))) %>%
  ungroup() 

data_0.5sec_validate[,5:253]<-exp(data_0.5sec_validate[,5:253]/10)

# any data that occurs consecutively is given the same grouping. If there is a 0.5sec gap then it becomes a new group. I've specified here there shortest track should be 5.
data_0.5sec_validate_grps<-split_at_gap(data = data_0.5sec_validate, 
                                     max_gap = 0.5, 
                                     shortest_track = 5)


# Creating a listing variable within each group so that we can split groups longer than 5 into groups of 5
data_0.5sec_validate_grps<-data_0.5sec_validate_grps%>%group_by(ID)%>%mutate(grp=rep(1:ceiling(n()/5), each=5, length.out=n()))%>%ungroup()

# splitting into lists 
listgrps_validate<-data_0.5sec_validate_grps%>%group_split(ID,grp)

# keeping only lists that are of length 5
listgrps_validate<-listgrps_validate[sapply(listgrps_validate, nrow) >= 5]

# Selecting the x data
x_data_validate<-list()

for(i in 1:378){
  x_data_validate[[i]]<-listgrps_validate[[i]]%>%select(F45_1:F170_1)
}

# each dataframe in the list to a matrix
x_data_validate<-lapply(x_data_validate, as.matrix)

# Flatten into a 3D array
x_data_validate<-lm2a(x_data_validate,dim.order=c(3,1,2))

# Check dims
dim(x_data_validate)

# Selecting the y data
y_data_validate<-vector()

for(i in 1:378){
  a <-listgrps_validate[[i]]%>%select(species_1)
  y_data_validate[i]<-a[1,]
}

# Unlist
y_data_validate<-unlist(y_data_validate)

# Balance the classes
summary(factor(y_data_validate)) #need 87 

# remove 110 LT & 7 SMB
set.seed(5)
rem<-sample(c(rep(0,110),rep(1,87)),197)
rem2<-rep(1,87)
rem3<-sample(c(rep(0,7),rep(1,87)),94)
remove<-which(c(rem,rem2,rem3)==0)
x_data_validate<-x_data_validate[-c(remove),,]
y_data_validate<-y_data_validate[-remove]


y_validate<-NA
y_validate[y_data_validate=="LT"]<-0
y_validate[y_data_validate=="LWF"]<-1
y_validate[y_data_validate=="SMB"]<-2
summary(y_validate)


dummy_y_validate<-to_categorical(y_validate, num_classes = 3)

# Shuffle data
set.seed(250)
x<-sample(1:nrow(x_data_validate))
x_data_validate_S= x_data_validate[x, ,] 
dummy_y_validate_S= dummy_y_validate[x, ] 

summary(as.factor(y_data_validate))


### test Data
# Group together and summarise any pings that occur less than 0.5secs apart. 
data_0.5sec_test <- test%>%
  select(fishNum,Ping_time,F45:F170,species)%>% #,F45:F170
  mutate(ID= fishNum) %>% 
  group_by(ID,time = floor_date(Ping_time, unit = "0.5 sec")) %>%
  summarise(across(everything(), list(max))) %>%
  ungroup() 

data_0.5sec_test[,5:253]<-exp(data_0.5sec_test[,5:253]/10)

# any data that occurs consecutively is given the same grouping. If there is a 0.5sec gap then it becomes a new group. I've specified here there shortest track should be 5.
data_0.5sec_test_grps<-split_at_gap(data = data_0.5sec_test, 
                                        max_gap = 0.5, 
                                        shortest_track = 5)


# Creating a listing variable within each group so that we can split groups longer than 5 into groups of 5
data_0.5sec_test_grps<-data_0.5sec_test_grps%>%group_by(ID)%>%mutate(grp=rep(1:ceiling(n()/5), each=5, length.out=n()))%>%ungroup()

# splitting into lists 
listgrps_test<-data_0.5sec_test_grps%>%group_split(ID,grp)

# keeping only lists that are of length 5
listgrps_test<-listgrps_test[sapply(listgrps_test, nrow) >= 5]

# Selecting the x data
x_data_test<-list()

for(i in 1:386){
  x_data_test[[i]]<-listgrps_test[[i]]%>%select(F45_1:F170_1)
}

# each dataframe in the list to a matrix
x_data_test<-lapply(x_data_test, as.matrix)

# Flatten into a 3D array
x_data_test<-lm2a(x_data_test,dim.order=c(3,1,2))

# Check dims
dim(x_data_test)

# Selecting the y data
y_data_test<-vector()

for(i in 1:386){
  a <-listgrps_test[[i]]%>%select(species_1)
  y_data_test[i]<-a[1,]
}

# Unlist
y_data_test<-unlist(y_data_test)

# Balance the classes
summary(factor(y_data_test)) #need 39 

# remove 225 LT & 44 SMB
set.seed(5)
rem<-sample(c(rep(0,225),rep(1,39)),264)
rem2<-rep(1,39)
rem3<-sample(c(rep(0,44),rep(1,39)),83)
remove<-which(c(rem,rem2,rem3)==0)
x_data_test<-x_data_test[-c(remove),,]
y_data_test<-y_data_test[-remove]


y_test<-NA
y_test[y_data_test=="LT"]<-0
y_test[y_data_test=="LWF"]<-1
y_test[y_data_test=="SMB"]<-2
summary(y_test)
dummy_y_test<-to_categorical(y_test, num_classes = 3)

# Shuffle data
set.seed(250)
x<-sample(1:nrow(x_data_test))
x_data_test_S= x_data_test[x, ,] 
dummy_y_test_S= dummy_y_test[x, ] 

summary(as.factor(y_data_test))


rnn <- keras_model_sequential() 

# Add LSTM layers with dropout
rnn %>%
  layer_lstm(input_shape=c(5, 249), units=200, dropout=0.2, recurrent_dropout=0.2, return_sequences=TRUE) %>%
  layer_lstm(units=200, dropout=0.2, recurrent_dropout=0.2) %>%
  layer_dense(units=80, activation='relu') %>%
  layer_dense(units=3, activation='softmax')

# Compile the model
rnn %>% compile(
  loss = loss_categorical_crossentropy,
  optimizer = optimizer_adam(),
  metrics = c('accuracy')
)

# Fit the model with early stopping
history <- rnn %>% fit(
  x_data_train_S, dummy_y_train_S,
  batch_size = 100, 
  epochs = 40,
  validation_data = list(x_data_validate_S, dummy_y_validate_S),
  callbacks = list(callback_early_stopping(patience=5))
)

# Evaluate the model
evaluate(rnn, x_data_test_S, dummy_y_test_S) 

# Plot training history
plot(history)


preds<-predict(rnn, x=x_data_test_S)

species.predictions<-apply(preds,1,which.max)
species.predictions<-as.factor(ifelse(species.predictions == 1, "LT",
                                      ifelse(species.predictions ==2, "LWF", "SMB")))
confusionMatrix(species.predictions,as.factor(y_data_test))

