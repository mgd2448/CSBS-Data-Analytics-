rm(list=ls())
setwd("C:/Users/jmun/OneDrive/Desktop/CSBS/Data")

library(lmtest)
library(caret)
library(dplyr)
library(MASS)
library(rattle)
library(randomForest)

options(scipen = 100)
memory.limit(size=30000)

#Loading data sets into R#
train_data = read.csv("Train_Model_Sample.csv", header=TRUE, sep=',', stringsAsFactors = TRUE)
test_data = read.csv("Test_Model_Sample.csv", header=TRUE, sep=',', stringsAsFactors = TRUE)
full_data = read.csv("Full_Model_Data.csv", header=TRUE, sep=',', stringsAsFactors = TRUE)

full_data<-filter(full_data, CB==1)
train_data<-filter(train_data, CB==1)
test_data<-filter(test_data, CB==1)

train_data$FintechPartnership <- as.factor(train_data$FintechPartnership)
train_data$minority <- as.factor(train_data$minority)
train_data$lmiindicator<- as.factor(train_data$lmiindicator)
train_data$ruralurbanindicator<- as.factor(train_data$ruralurbanindicator)

test_data$FintechPartnership <- as.factor(test_data$FintechPartnership)
test_data$minority <- as.factor(test_data$minority)
test_data$lmiindicator<- as.factor(test_data$lmiindicator)
test_data$ruralurbanindicator<- as.factor(test_data$ruralurbanindicator)

full_data$FintechPartnership <- as.factor(full_data$FintechPartnership)
full_data$minority <- as.factor(full_data$minority)
full_data$lmiindicator<- as.factor(full_data$lmiindicator)
full_data$ruralurbanindicator<- as.factor(full_data$ruralurbanindicator)


####Omitting character and nonrelevant columns from all data sets, I took out CB since it's all 1
train_data = subset(train_data, select=-c(forgivenessamount,Sum_Approval.Branch, Sum_Jobs.Branch,CERT, Zip, loannumber, NameFull, 
                                          City, borrowerstate, borrowercity, originatinglender, originatinglendercity, 
                                          originatinglenderstate, naicscode, Unbanked..90.PCT.CI., Has.bank.account..90.PCT.CI., 
                                          Has.bank.account, Right_State, NaicsCode2, Size, dateapproved, Right_cert, CB, 
                                          HCAsset,Stcnty))
train_data<-na.omit(train_data) 

test_data = subset(test_data, select=-c(forgivenessamount, Sum_Approval.Branch, Sum_Jobs.Branch,CERT, Zip, loannumber,NameFull, City, borrowerstate, borrowercity, originatinglender, originatinglendercity, originatinglenderstate,
                                          naicscode, Unbanked..90.PCT.CI., Has.bank.account,Has.bank.account..90.PCT.CI., Right_State, NaicsCode2, Size, dateapproved, Right_cert, CB, HCAsset,Stcnty))
test_data<-na.omit(test_data) 


full_data = subset(full_data, select=-c(forgivenessamount, Sum_Approval.Branch, Sum_Jobs.Branch,CERT, Zip, loannumber,NameFull, City, borrowerstate, borrowercity, originatinglender, originatinglendercity, originatinglenderstate,
                                        naicscode, Unbanked..90.PCT.CI., Has.bank.account,Has.bank.account..90.PCT.CI., Right_State, NaicsCode2, Size, dateapproved, Right_cert, CB, HCAsset,Stcnty))
full_data<-na.omit(full_data) 


############################################################
#################MINORITY PREDICTION MODEL##################
############################################################
train_data_minority <- subset(train_data, select = -c(lmiindicator))
test_data_minority <- subset(test_data, select = -c(lmiindicator))
full_data_minority <- subset(full_data, select = -c(lmiindicator))


############LOGISTIC MODEL WITH FINTECH COLUMN################
set.seed(5082)
logisticmodel <-glm(formula=minority~., family = binomial, data = train_data_minority)
summary(logisticmodel)
yhat<-predict(logisticmodel, test_data_minority, type="response")
yhat<-ifelse(yhat >.5, "1", "0")
confusion<-table(Predicted=yhat, Actual=test_data_minority$minority)
confusion
#Accuracy rate
(confusion[1]+confusion[4])/sum(confusion) #Logistic model provides accuracy rate: 61.83



###########TESTING ASSUMPTIONS#################

###Assumption 1: Linearity of the Logit####
interaction <- full_data_minority$Asset * log(full_data_minority$Asset)
multi<-glm(minority~interaction, family = binomial, data= full_data_minority) ###Statistically statistic! 
summary(multi)

interaction <- full_data_minority$LoanToAsset * log(full_data_minority$LoanToAsset)
multi<-glm(minority~interaction, family = binomial, data= full_data_minority) ###Statistically statistic! 
summary(multi)

interaction <- full_data_minority$CoreRatio * log(full_data_minority$CoreRatio)
multi<-glm(minority~interaction, family = binomial, data= full_data_minority) ###Statistically statistic! 
summary(multi)

interaction <- full_data_minority$Office_Count * log(full_data_minority$Office_Count)
multi<-glm(minority~interaction, family = binomial, data= full_data_minority) ###Statistically statistic! 
summary(multi)

interaction <- full_data_minority$Unique_Metros * log(full_data_minority$Unique_Metros)
multi<-glm(minority~interaction, family = binomial, data= full_data_minority) ###Statistically statistic! 
summary(multi)

interaction <- full_data_minority$State_Count * log(full_data_minority$State_Count)
multi<-glm(minority~interaction, family = binomial, data= full_data_minority) ###Statistically statistic! 
summary(multi)

interaction <- full_data_minority$currentapprovalamount * log(full_data_minority$currentapprovalamount)
multi<-glm(minority~interaction, family = binomial, data= full_data_minority) ###Statistically statistic! 
summary(multi)

interaction <- full_data_minority$jobsreported * log(full_data_minority$jobsreported)
multi<-glm(minority~interaction, family = binomial, data= full_data_minority) ###Statistically statistic! 
summary(multi)


interaction <- full_data_minority$Number.of.Households..1000s. * log(full_data_minority$Number.of.Households..1000s.)
multi<-glm(minority~interaction, family = binomial, data= full_data_minority) ###Statistically statistic! 
summary(multi)

interaction <- full_data_minority$Unbanked * log(full_data_minority$Unbanked)
multi<-glm(minority~interaction, family = binomial, data= full_data_minority) ###Statistically statistic! 
summary(multi)


#############Assumption 2: Multicolinearity#############
car::vif(logisticmodel) #Looks good! Highest value is 3.54



#############Assumption 3: Lack of strongly influencial outliers#############
library(broom)
library(tidyverse)
library(ggplot2)

modelResults <- augment(logisticmodel) %>% mutate(index = 1:n())
ggplot(modelResults, aes(index, .std.resid))+geom_point(aes(color='red'))


#############Assumption 4: Independence of Errors#############
plot(logisticmodel, which=3) 


###################RANGER MINORITY###################################
library(ranger)


## Initialize empty accuracy list 
accuracy <- c()

## Define start time and iterate through list of tree amounts 
Sys.time()
set.seed(5082)
for (i in c(100, 200, 300, 400, 500)){
  ra.fb <- ranger(minority~.,
                  train_data_minority,
                  num.trees = i,
                  write.forest = TRUE,
                  importance = 'impurity')
  pred <- predict(ra.fb, test_data_minority)
  accuracy <- append(accuracy, mean(pred$predictions==test_data_minority$minority))
}
accuracy
Sys.time() 
which.max(accuracy)


set.seed(5082)
ra.fb <- ranger(minority~.,
                train_data_minority,
                num.trees = 300,
                write.forest = TRUE,
                importance = 'impurity')
pred <- predict(ra.fb, test_data_minority)
print(mean(pred$predictions==test_data_minority$minority))

ra.fb$variable.importance






############################################################
#################LMI PREDICTION MODEL##################
############################################################

train_data_lmi <- subset(train_data, select = -c(minority))
test_data_lmi <- subset(test_data, select = -c(minority))

############LOGISTIC MODEL WITH FINTECH COLUMN################
set.seed(5082)
logisticmodel <-glm(formula=lmiindicator~., family = binomial, data = train_data_lmi)
summary(logisticmodel)
yhat<-predict(logisticmodel, test_data_lmi, type="response")
yhat<-ifelse(yhat >.5, "1", "0")
confusion<-table(Predicted=yhat, Actual=test_data_lmi$lmiindicator)
confusion
#Accuracy rate
(confusion[1]+confusion[4])/sum(confusion) #Logistic model provides accuracy rate: 69.89%







###################RANGER LMI###################################
library(ranger)


## Initialize empty accuracy list 
accuracy <- c()

## Define start time and iterate through list of tree amounts 
Sys.time()
set.seed(5082)
for (i in c(100, 200, 300, 400, 500)){
  ra.fb <- ranger(lmiindicator~.,
                  train_data_lmi,
                  num.trees = i,
                  write.forest = TRUE,
                  importance = 'impurity')
  pred <- predict(ra.fb, test_data_lmi)
  accuracy <- append(accuracy, mean(pred$predictions==test_data_lmi$lmiindicator))
}
accuracy
Sys.time() 
which.max(accuracy)


set.seed(5082)
ra.fb <- ranger(lmiindicator~.,
                train_data_lmi,
                num.trees = 100,
                write.forest = TRUE,
                importance = 'impurity')
pred <- predict(ra.fb, test_data_lmi)
print(mean(pred$predictions==test_data_lmi$lmiindicator))
ranger::importance(ra.fb)
ra.fb$variable.importance









