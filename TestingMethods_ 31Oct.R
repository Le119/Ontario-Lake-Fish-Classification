## Run a random forest on the cleaned data. This just has clean LT data and all LWF data
library(dplyr)
library(randomForest)
library(caret)
library(foreach)
library(doParallel)
library(ranger)
library(tidymodels)
library(vip)

## Read in the data
load("~/Library/CloudStorage/OneDrive-UniversityofToronto/Documents/PostDoc/EATFAT/processed_AnalysisData.Rdata")

processed_data%>%group_by(spCode,fishNum)%>%count()

# Remove individuals with missing transducers
processed_data<-processed_data%>%filter(is.na(F100)==F)

# also remove individual LWF23018 (only two pings)
processed_data<-processed_data%>%filter(fishNum!="LWF23018")

# remove the individual with -9x10^38 TS
processed_data<-processed_data[-36830,]

## Lake Trout Filtering
## LT 009, 010, 012, 014 (although looks more like a dead fish), 016, 017 (6/10 condition), 021, 23007, 23009, 23013, 23012, 23011, 23010, 23005, 23004, 23003, 23002 are all good and were behaving normally.

# LT019 and LT23008 were dead the whole time. Remove. 
processed_data<-processed_data%>%filter(fishNum!="LT019")
processed_data<-processed_data%>%filter(fishNum!="LT23008")

# LT015, 018, and 23018 were bad on the way down and then good on retrieval - will filter the first parts of the data

# LT015 was at a very shallow depth and on quadrant for the majority of pinging, then seemed to move to the correct depth. Keep only those pings
processed_data <- processed_data[!(processed_data$fishNum == "LT015" & processed_data$Target_true_depth > 15.5 ) ,]

# LT018 was rough on attachment too, also has two rythmic changes in depth - going to remove the times the fish was above 15.5m as this will get rid of the time that fish was being dragged potentially and the first part of the timeseries where the fish was rough
processed_data <- processed_data[!(processed_data$fishNum == "LT018" & processed_data$Target_true_depth > 15.5 ) ,]

# LT23018 was rough on attachment, barely alive, but "suprisingly okay" coming back - no clear indication of when it got "okay" in the data so remove whole fish
processed_data<-processed_data%>%filter(fishNum!="LT23018")


# LT013, LT017, and LT23001 went down okay but died sometime down there. 

# LT013 was almost dead on release as well as dead on retrival - remove all. 
processed_data<-processed_data%>%filter(fishNum!="LT013")

# LT23001 looks like there are barely any salvagable pings - remove all
processed_data<-processed_data%>%filter(fishNum!="LT23001")


## Cleaned data: 
processed_data%>%filter(spCode==81)%>%group_by(fishNum)%>%count()
# 21 fish, 22,792 pings

## Looking at cleaned LT data
print(processed_data%>%filter(spCode==81)%>%group_by(fishNum)%>%summarise(TL=mean(totalLength)),n=21)



# PROCESS AND Transform DATA
F45index <- which(names(processed_data) == "F45")
f1=seq(from=45,to=89.5,by=0.5) #first group of frequencies
f2=seq(from=90,to=170,by=0.5) #second group of frequencies
f3=seq(from=173,to=260,by=0.5) #third group of frequencies
listf1=seq(from=F45index,to=(F45index-1)+length(f1),by=1) #columns identifying the first group of frequencies
listf2=seq(from=F45index+length(f1),to=(F45index-1)+length(f1)+length(f2),by=1) #columns identifying the second group of frequencies
listf3=seq(from=F45index+length(f1)+length(f2),to=(F45index-1)+length(f1)+length(f2)+length(f3),by=1) #columns identifying the third group of frequencies
f1mar1=45
f1mar2=89.5
f2mar1=91
f2mar2=170
f3mar1=173
f3mar2=260

# filter
f1inc=f1>=(f1mar1)&f1<=(f1mar2) #list of frequencies to keep in the first group
f2inc=f2>=(f2mar1)&f2<=(f2mar2) #list of frequencies to keep in the second group
f3inc=f3>=(f3mar1)&f3<=(f3mar2) #list of frequencies to keep in the third group
freqs=c(f1[f1inc],f2[f2inc],f3[f3inc]) #kept frequencies


X1=exp(processed_data[,listf1[f1inc]]/10) #ftransform to acoustic backscatter
X2=exp(processed_data[,listf2[f2inc]]/10)
X3=exp(processed_data[,listf3[f3inc]]/10)

# create unique region
processed_data$unique.region<-interaction(processed_data$fishNum,processed_data$Region_name)

y <- as.factor(processed_data$spCode)
region<-processed_data$unique.region
fishNum<-processed_data$fishNum


# make test and train as a dataset
data = cbind.data.frame(X1,X2,X3,y,fishNum,region)

set.seed(123)
split<-group_initial_split(data,group=fishNum,strata = y)
train<-training(split)
test<-testing(split)

test%>%group_by(y)%>%count()
train%>%group_by(y)%>%count()

# make the design balanced across both species
test<-test%>%filter(y=="81"&row_number()<3222|y=="91")
train<-train%>%filter(y=="81"&row_number()<9537|y=="91")

set.seed(123)
cv_folds <- train%>%group_vfold_cv(v = 5,group=fishNum,strata=y)

# Set up optimization code (will tune the number of trees and the mtry)
model_rf <- 
  rand_forest(mtry = tune(), trees = tune(), min_n = 1) %>% 
  set_engine("ranger", importance = "impurity") %>% 
  set_mode("classification")


# Model code
wkfl_rf <- 
  workflow() %>% 
  add_formula(y ~ F45+F45.5+F46+F46.5+F47+F47.5+F48+F48.5+F49+F49.5+F50+F50.5+F51+F51.5+F52+F52.5+F53+F53.5+F54+F54.5+F55+F55.5+F56+F56.5+F57+F57.5+F58+F58.5+F59+F59.5+F60+F60.5+F61+F61.5+F62+F62.5+F63+F63.5+F64+F64.5+F65+F65.5+F66+F66.5+F67+F67.5+F68+F68.5+F69+F69.5+F70+F70.5+F71+F71.5+F72+F72.5+F73+F73.5+F74+F74.5+F75+F75.5+F76+F76.5+F77+F77.5+F78+F78.5+F79+F79.5+F80+F80.5+F81+F81.5+F82+F82.5+F83+F83.5+F84+F84.5+F85+F85.5+F86+F86.5+F87+F87.5+F88+F88.5+F89+F89.5+F91+F91.5+F92+F92.5+F93+F93.5+F94+F94.5+F95+F95.5+F96+F96.5+F97+F97.5+F98+F98.5+F99+F99.5+F100+F100.5+F101+F101.5+F102+F102.5+F103+F103.5+F104+F104.5+F105+F105.5+F106+F106.5+F107+F107.5+F108+F108.5+F109+F109.5+F110+F110.5+F111+F111.5+F112+F112.5+F113+F113.5+F114+F114.5+F115+F115.5+F116+F116.5+F117+F117.5+F118+F118.5+F119+F119.5+F120+F120.5+F121+F121.5+F122+F122.5+F123+F123.5+F124+F124.5+F125+F125.5+F126+F126.5+F127+F127.5+F128+F128.5+F129+F129.5+F130+F130.5+F131+F131.5+F132+F132.5+F133+F133.5+F134+F134.5+F135+F135.5+F136+F136.5+F137+F137.5+F138+F138.5+F139+F139.5+F140+F140.5+F141+F141.5+F142+F142.5+F143+F143.5+F144+F144.5+F145+F145.5+F146+F146.5+F147+F147.5+F148+F148.5+F149+F149.5+F150+F150.5+F151+F151.5+F152+F152.5+F153+F153.5+F154+F154.5+F155+F155.5+F156+F156.5+F157+F157.5+F158+F158.5+F159+F159.5+F160+F160.5+F161+F161.5+F162+F162.5+F163+F163.5+F164+F164.5+F165+F165.5+F166+F166.5+F167+F167.5+F168+F168.5+F169+F169.5+F170+F173+F173.5+F174+F174.5+F175+F175.5+F176+F176.5+F177+F177.5+F178+F178.5+F179+F179.5+F180+F180.5+F181+F181.5+F182+F182.5+F183+F183.5+F184+F184.5+F185+F185.5+F186+F186.5+F187+F187.5+F188+F188.5+F189+F189.5+F190+F190.5+F191+F191.5+F192+F192.5+F193+F193.5+F194+F194.5+F195+F195.5+F196+F196.5+F197+F197.5+F198+F198.5+F199+F199.5+F200+F200.5+F201+F201.5+F202+F202.5+F203+F203.5+F204+F204.5+F205+F205.5+F206+F206.5+F207+F207.5+F208+F208.5+F209+F209.5+F210+F210.5+F211+F211.5+F212+F212.5+F213+F213.5+F214+F214.5+F215+F215.5+F216+F216.5+F217+F217.5+F218+F218.5+F219+F219.5+F220+F220.5+F221+F221.5+F222+F222.5+F223+F223.5+F224+F224.5+F225+F225.5+F226+F226.5+F227+F227.5+F228+F228.5+F229+F229.5+F230+F230.5+F231+F231.5+F232+F232.5+F233+F233.5+F234+F234.5+F235+F235.5+F236+F236.5+F237+F237.5+F238+F238.5+F239+F239.5+F240+F240.5+F241+F241.5+F242+F242.5+F243+F243.5+F244+F244.5+F245+F245.5+F246+F246.5+F247+F247.5+F248+F248.5+F249+F249.5+F250+F250.5+F251+F251.5+F252+F252.5+F253+F253.5+F254+F254.5+F255+F255.5+F256+F256.5+F257+F257.5+F258+F258.5+F259+F259.5+F260) %>% 
  add_model(model_rf)

params <- 
  wkfl_rf %>% 
  extract_parameter_set_dials() %>% 
  update(mtry = mtry(c(2, 50)),trees = trees(c(1,25)))
# set metrics to keep
my_metrics <- metric_set(accuracy, sens, spec, precision)
# accuracy is the percent of predictions that are correct
# precision is the number of correctly identified members of a class divided by the number of times that class was predicted
#Sensitivity (SN) is calculated as the number of correct positive predictions divided by the total number of positives
#Specificity (true negative rate) is the probability of a negative test result, conditioned on the individual truly being negative.


# Run the Bayesian Optimization on mtry and ntree
doParallel::registerDoParallel(4)
set.seed(123)
rf_fit <- tune_bayes(
  wkfl_rf,
  resamples = cv_folds,
  initial = 5,
  metrics = my_metrics,
  param_info = params,
  iter=5,
  control = control_bayes(verbose = TRUE,parallel_over="resamples", no_improve=15) ) # don't save prediction (imho)
doParallel::stopImplicitCluster()


rf <- ranger(y ~ .,mtry=25,ntree=500, data = train[,1:425])
pred.rf <- predict(rf, data = test[,1:425])
table(test$y, pred.rf$predictions)



## Gaussian Mixture Model
## Using columns 1:60 works fine, but 1:90 I get an error, trying to narrow down where it occurs! 
# Columns 61:85 works, 61:90 works. 
# Now trying G = 5:10 for 1:90, perhaps there is too much to decompose into 5 components.
# 1:90 runs with G 5:10, but number of components is 9 and 10 so probably want to increase that with the full dataset. Test set error = 28%.(This is on data that has been transformed into acoustic backscatter coefficient but not corrected for size in anyway!)
# Next would be to run with all frequencies, use cross validation, look up if there is a way to get variable importance, run on size standardised data. 

library(mclust)
class <- factor(train$y)
table(class)

X <- train[,1:90]
head(X)
modDA <- MclustDA(X, class, G=5:10)
summary(modDA)

newclass<-test$y
X.test<-test[,1:90]
summary(modDA,newdata=X.test,newclass=newclass)
