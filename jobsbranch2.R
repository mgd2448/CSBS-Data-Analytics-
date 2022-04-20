# Increase memory limit to accommodate the big dataset
memory.limit(size=30000)

setwd("C:/Users/jmun/OneDrive/Desktop/CSBS/Data")
rm(list=ls())
library(tidyverse)
library(lmtest)
library(mltools)
library(broom)
library(ggplot2)
library(dplyr)

# Full Dataset
data = read.csv("Full_Model_Data.csv", stringsAsFactors=TRUE)

# Convert to dataframe
df = as.data.frame(data)

# Update column name
df <- rename(df, Number_Households = Number.of.Households..1000s.)

# Replacing N/As with median of forgivenessAmount 
df$forgivenessamount[is.na(df$forgivenessamount)] <- median(df$forgivenessamount, na.rm=TRUE)
df$jobsreported[is.na(df$jobsreported)] <- median(df$jobsreported, na.rm=TRUE)

# Check if NAs are still present
summary(df$jobsreported)
summary(df$forgivenessamount) #no NA


###### CONTINUOUS VARIABLE EVALUATION #####
# scale continuous vars
df$Sum_Jobs.Branch <- scale(df$Sum_Jobs.Branch, center=T, scale=T)
df$Asset <- scale(df$Asset, center=T, scale=T)
df$LoanToAsset <- scale(df$LoanToAsset, center=T, scale=T)
df$Office_Count <- scale(df$Office_Count, center=T, scale=T)
df$CoreRatio <- scale(df$CoreRatio, center=T, scale=T)
df$State_Count <- scale(df$State_Count, center=T, scale=T)
df$Unbanked <- scale(df$Unbanked, center=T, scale=T)
df$Number_Households <- scale(df$Number_Households, center=T, scale=T)
df$jobsreported <- scale(df$jobsreported, center=T, scale=T)
df$forgivenessamount <- scale(df$forgivenessamount, center=T, scale=T)
df$currentapprovalamount <- scale(df$currentapprovalamount, center=T, scale=T)

# Remove outliers
boxplot(df$currentapprovalamount, plot=FALSE)
outliers <- boxplot(df$currentapprovalamount, plot=FALSE)$out
x <- df
x <- x[-which(x$currentapprovalamount %in% outliers),]

# Check correlation
cont=subset(x, select = c(Asset,LoanToAsset, Office_Count, CoreRatio,State_Count,currentapprovalamount,Unbanked,Number_Households, jobsreported, forgivenessamount))
cor(cont)

# Plot correlation amongst continuous vars
library(corrplot)
corrplot(cor(cont[, unlist(lapply(cont, is.numeric))]))

#Including anything above absolute value of .4
#keep just one column 
#LoanToAsset & Office_Count: -0.726461556 - offices considered an asset, so bigger banks with have smaller loan to asset ratio 
#LoanToAsset & Unique_Metros: -0.752436641 - offices considered an asset, so bigger banks with have smaller loan to asset ratio 
#LoanToAsset & State_Count: -0.752294093 # - offices considered an asset, so bigger banks with have smaller loan to asset ratio 

#just keep state_count - remove Unique_metros 
#Office_Count & Unique_Metros: 0.98022037    
#Office_Count & State_Count: 0.98269340  
#State_Count & Unique_Metros: 0.98461548 -- State_count and Unique_Metros are likely highly correlated          

# CONCLUSION: 
#remove HCasset
#remove Unique_Metros 
#remove Has.Bank.account
x <- subset(x, select = -c(HCAsset, Unique_Metros, Has.bank.account))

# Categorical Vars:
'''
#str(df)
 $ loannumber                  : num  1003248809 1030537104 1034918801 1047347102 1049948508 ...
 $ CERT                        : int  10359 10359 10359 10359 10359 10359 10359 10359 10359 10359 ...
 $ NameFull                    : Factor w/ 329 levels "1ST FINANCIAL BANK USA",..: 149 149 149 149 149 149 149 149 149 149 ...
 $ City    (about city not the name)  : Factor w/ 249 levels "ABINGTON","ALBUQUERQUE",..: 87 87 87 87 87 87 87 87 87 87 ...
 $ Zip                         : int  79042 79042 79042 79042 79042 79042 79042 79042 79042 79042 ...
 $ Stcnty                      : int  48437 48437 48437 48437 48437 48437 48437 48437 48437 48437 ...
 $ CB                          : int  0 0 0 0 0 0 0 0 0 0 ...
 $ Size                        : Factor w/ 3 levels "","L","S": 1 1 1 1 1 1 1 1 1 1 ...
 $ dateapproved                : Factor w/ 237 levels "2020-04-03","2020-04-04",..: 192 7 192 7 150 192 18 207 7 7 ...
 $ borrowercity                : Factor w/ 30474 levels "#2D",", Brooklyn",..: 472 20350 11324 3639 6917 17770 6918 6629 473 473 ...
 $ borrowerstate               : Factor w/ 57 levels "","AK","AL","AR",..: 49 49 49 49 49 49 49 49 49 49 ...
bhy
 $ ruralurbanindicator         : int  1 1 1 1 1 1 1 0 1 1 ...
 $ lmiindicator                : int  0 0 0 0 0 0 0 1 0 0 ...
 $ minority                    : int  1 0 1 0 1 1 0 1 0 0 ...
 $ FintechPartnership          : int  0 0 0 0 0 0 0 0 0 0 ...
 $ Unbanked..90.PCT.CI.        : Factor w/ 46 levels "(1.7,3.2)","(1.9,3.4)",..: 44 44 44 44 44 44 44 44 44 44 ...
 $ Has.bank.account..90.PCT.CI.: Factor w/ 46 levels "(85,88.2)","(85.4,87.9)",..: 10 10 10 10 10 10 10 10 10 10 ...
 $ Right_State                 : Factor w/ 46 levels "AL","AR","AZ",..: 40 40 40 40 40 40 40 40 40 40 ...
 $ NaicsCode2                  : int  81 62 11 72 61 48 23 48 72 44 ...
 $ Right_cert                  : int  10359 10359 10359 10359 10359 10359 10359 10359 10359 10359 ...
'''

# Drop more irrelevant cols
x <- subset(x, select = -c(loannumber, NameFull, City, Stcnty, Size, dateapproved, borrowercity,
                             originatinglender, originatinglendercity, Unbanked..90.PCT.CI., Has.bank.account..90.PCT.CI.,
                             Right_cert))
dim(x) #824372     24


# Identify the categorical variables with direct effects on Sum_Jobs.Branch
set.seed(1)
lmcat1 <- lm(Sum_Jobs.Branch~CERT, data = x)
summary(lmcat1) # Adjusted R-squared:  0.4145   

lmcat2 <- lm(Sum_Jobs.Branch~Zip, data = x)
summary(lmcat2)# Adjusted R-squared: 0.1235 

lmcat3 <- lm(Sum_Jobs.Branch~CB, data = x) 
summary(lmcat3)# Adjusted R-squared: 0.6468  

lmcat4 <- lm(Sum_Jobs.Branch~borrowerstate, data = x)
summary(lmcat4)# Adjusted R-squared: 0.06052         

lmcat5 <- lm(Sum_Jobs.Branch~originatinglenderstate, data = x)
summary(lmcat5) # Adjusted R-squared: 0.8354 

lmcat6 <- lm(Sum_Jobs.Branch~naicscode, data = df)
summary(lmcat6) # Adjusted R-squared: 0.0001238 

lmcat7 <- lm(Sum_Jobs.Branch~ruralurbanindicator, data = df)
summary(lmcat7) # Adjusted R-squared: 0.005283  

lmcat8 <- lm(Sum_Jobs.Branch~lmiindicator, data = df)
summary(lmcat8)# Adjusted R-squared: 0.003832        

lmcat9 <- lm(Sum_Jobs.Branch~ minority, data = df)
summary(lmcat9)# Adjusted R-squared: 0.005463  

lmcat10 <- lm(Sum_Jobs.Branch~ FintechPartnership, data = df)
summary(lmcat10)# Adjusted R-squared: 0.6613  


#Dummy code all categorical variables with adjusted R squared above .6
x$cb_1 <- ifelse(x$CB == 1, 1, 0)
x$cb_0 <- ifelse(x$CB == 0, 1, 0)

x$FintechPartnership_1 <- ifelse(x$FintechPartnership == 1, 1, 0)
x$FintechPartnership_0 <- ifelse(x$FintechPartnership == 0, 1, 0)


# lmmodel1 without originatingstate
lmmodel1 = lm(Sum_Jobs.Branch ~ Asset + LoanToAsset + Office_Count + CoreRatio + State_Count + 
                currentapprovalamount + Unbanked + Number_Households + jobsreported + forgivenessamount + 
                cb_1 + cb_0 + FintechPartnership_1 + FintechPartnership_0, data=x)
summary(lmmodel1)
# cb_0, FintechPartnership_0: Insignificant
# Adjusted R-squared:  0.7863  

# lmmodel2: log(Sum_Jobs.Branch) & Drop insignificant vars from lmmodel1
lmmodel2 = lm(log(Sum_Jobs.Branch) ~ Asset + LoanToAsset + Office_Count + CoreRatio + State_Count + 
                currentapprovalamount + Unbanked + Number_Households + jobsreported + forgivenessamount + 
                cb_1 + FintechPartnership_1, data=x)
summary(lmmodel2)
# currentapprovalamount, FintechPartnership_1: Insignificant


# lmmodel3: Drop insignificant vars from lmmodel2
lmmodel3 = lm(log(Sum_Jobs.Branch) ~ Asset + LoanToAsset + Office_Count + CoreRatio + State_Count + 
                Unbanked + Number_Households + jobsreported + log(forgivenessamount) + 
                cb_1, data=x)
summary(lmmodel3)
# Adjusted R-squared:  0.4852 

# scaling does not resolve skewness
# check all vars' skewness
# reg: check model assumptions
# logreg: need to check assumptions
# tree, svm make no prob assumptions

# r-sq: reg is gd
# RSS: reduce 
# tree: tells u vars of importance 
# gini impurity redux: substitute metrics




# Verify lmmodel2
anova(lmmodel2)

# Testing assumptions of linear reg
# 1. Plot residuals (linearity and additivity of the relationship between dependent and independent variables)
plot(lmmodel2, which = 1)
# 

# 2. Plot QQ (normality of the error distribution)
plot(lmmodel2, which = 2)
# residuals not normally distributed

# 3. Test heteroscedasticity
bptest(lmmodel2, data=df, studentize=FALSE) 
# BP = 450058, df = 11, p-value < 0.00000000000000022 
# Highly heteroscedastic

# 4. Test multicollinearity of variables
library(car)
vif(lmmodel2)
'''
                     Asset                LoanToAsset               Office_Count                  CoreRatio                State_Count 
                  1.053528                   2.221296                  23.695721                   1.166538                  26.129395 
log(currentapprovalamount)                   Unbanked          Number_Households               jobsreported          forgivenessamount 
                  1.532594                   1.203590                   1.082074                   1.831377                   2.010047 
                      cb_1       FintechPartnership_1 
                 16.063896                  16.353686
'''

#create vector of VIF values
vif_values <- vif(lmmodel2)
#create horizontal bar chart to display each VIF value
barplot(vif_values, main = "VIF Values", horiz = TRUE, col = "steelblue")
#add vertical line at 5
abline(v = 5, lwd = 3, lty = 2)


# lmmodel2: Drop vars with high multicollinearity from lmmodel2 and log($$)
lmmodel3 = lm(Sum_Jobs.Branch~ Asset + LoanToAsset + Office_Count + CoreRatio + State_Count + 
                log(currentapprovalamount) + Unbanked + Number_Households + jobsreported + forgivenessamount + 
                cb_1 + FintechPartnership_1, data=df)
summary(lmmodel3)
# Adjusted R-squared:  0.4248  


### AdaBoost ###
install.packages("adabag")
library(adabag)
df$Sum_Jobs.Branch <- as.factor(df$Sum_Jobs.Branch)
adaboost<-boosting(Sum_Jobs.Branch~Asset + LoanToAsset + Office_Count + CoreRatio + State_Count + 
                     Unbanked + Number_Households + jobsreported + forgivenessamount + 
                     cb_1 + FintechPartnership_1, data=df, boos=TRUE, mfinal=20,coeflearn='Breiman')
summary(adaboost)
# takes forever couldnt finish running





