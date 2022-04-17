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
attach(df)

# Replacing N/As with median of forgivenessAmount 
df$forgivenessamount[is.na(df$forgivenessamount)] <- median(df$forgivenessamount, na.rm=TRUE)
df$jobsreported[is.na(df$jobsreported)] <- median(df$jobsreported, na.rm=TRUE)

# Check if NAs are still present
summary(df$jobsreported) #no NA
summary(df$forgivenessamount) #no NA


###### CONTINUOUS VARIABLE EVALUATION #####
cont=subset(df, select = c(Asset,LoanToAsset, Office_Count, CoreRatio,State_Count,currentapprovalamount,Unbanked,Number_Households, jobsreported, forgivenessamount))

# scale continuous vars
scale(cont, scale=TRUE, center=TRUE)
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
df <- subset(df, select = -c(HCAsset, Unique_Metros, Has.bank.account))

# Categorical Vars:
'''
#str(df)
 $ loannumber                  : num  1003248809 1030537104 1034918801 1047347102 1049948508 ...
 $ CERT                        : int  10359 10359 10359 10359 10359 10359 10359 10359 10359 10359 ...
 $ NameFull                    : Factor w/ 329 levels "1ST FINANCIAL BANK USA",..: 149 149 149 149 149 149 149 149 149 149 ...
 $ City                        : Factor w/ 249 levels "ABINGTON","ALBUQUERQUE",..: 87 87 87 87 87 87 87 87 87 87 ...
 $ Zip                         : int  79042 79042 79042 79042 79042 79042 79042 79042 79042 79042 ...
 $ Stcnty                      : int  48437 48437 48437 48437 48437 48437 48437 48437 48437 48437 ...
 $ CB                          : int  0 0 0 0 0 0 0 0 0 0 ...
 $ Size                        : Factor w/ 3 levels "","L","S": 1 1 1 1 1 1 1 1 1 1 ...
 $ dateapproved                : Factor w/ 237 levels "2020-04-03","2020-04-04",..: 192 7 192 7 150 192 18 207 7 7 ...
 $ borrowercity                : Factor w/ 30474 levels "#2D",", Brooklyn",..: 472 20350 11324 3639 6917 17770 6918 6629 473 473 ...
 $ borrowerstate               : Factor w/ 57 levels "","AK","AL","AR",..: 49 49 49 49 49 49 49 49 49 49 ...
 $ originatinglender           : Factor w/ 331 levels "1st Financial Bank USA",..: 149 149 149 149 149 149 149 149 149 149 ...
 $ originatinglendercity       : Factor w/ 250 levels "ABINGTON","ALBUQUERQUE",..: 90 90 90 90 90 90 90 90 90 90 ...
 $ originatinglenderstate      : Factor w/ 46 levels "AL","AR","AZ",..: 40 40 40 40 40 40 40 40 40 40 ...
 $ naicscode                   : num  813110 621320 112111 722515 611511 ...
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
df <- subset(df, select = -c(loannumber, NameFull, City, Stcnty, Size, dateapproved, borrowercity,
                             originatinglender, originatinglendercity, Unbanked..90.PCT.CI., Has.bank.account..90.PCT.CI.,
                             Right_cert))
dim(df) #824372     24


# Identify the categorical variables with direct effects on Sum_Approval.Branch
set.seed(1)
lmcat1 <- lm(Sum_Approval.Branch~CERT, data = df)
summary(lmcat1) # Adjusted R-squared:  0.4197  

lmcat2 <- lm(Sum_Approval.Branch~Zip, data = df)
summary(lmcat2)# Adjusted R-squared: 0.129  

lmcat3 <- lm(Sum_Approval.Branch~CB, data = df) 
summary(lmcat3)# Adjusted R-squared: 0.6715 

lmcat4 <- lm(Sum_Approval.Branch~borrowerstate, data = df) 
summary(lmcat4)# Adjusted R-squared: 0.06141        

lmcat5 <- lm(Sum_Approval.Branch~originatinglenderstate, data = df)
summary(lmcat5) # Adjusted R-squared: 0.8388

lmcat6 <- lm(Sum_Approval.Branch~naicscode, data = df)
summary(lmcat6) # Adjusted R-squared: 8.115e-05 

lmcat7 <- lm(Sum_Approval.Branch~ruralurbanindicator, data = df)
summary(lmcat7) # Adjusted R-squared: 0.005283 

lmcat8 <- lm(Sum_Approval.Branch~lmiindicator, data = df)
summary(lmcat8)# Adjusted R-squared: 0.003843       

lmcat9 <- lm(Sum_Approval.Branch~ minority, data = df)
summary(lmcat9)# Adjusted R-squared: 0.003843 

lmcat10 <- lm(Sum_Approval.Branch~ FintechPartnership, data = df)
summary(lmcat10)# Adjusted R-squared: 0.6864 


#Dummy code all categorical variables with adjusted R squared above .6
df$cb_1 <- ifelse(df$CB == 1, 1, 0)
df$cb_0 <- ifelse(df$CB == 0, 1, 0)

df$FintechPartnership_1 <- ifelse(df$FintechPartnership == 1, 1, 0)
df$FintechPartnership_0 <- ifelse(df$FintechPartnership == 0, 1, 0)

'''
#install.packages("fastDummies")
library(fastDummies)
# Create dummy variable
df <- dummy_cols(df, select_columns = "originatinglenderstate")
'''

'''
# Test independence of the categorical vars
chisq.test(df$CB,df$FintechPartnership) #independent
chisq.test(df$CB,df$originatinglenderstate)#independent
chisq.test(df$FintechPartnership,df$originatinglenderstate)#independent
'''

# lmmodel1 without originatingstate
lmmodel1 = lm(Sum_Approval.Branch~Asset + LoanToAsset + Office_Count + CoreRatio + State_Count + 
                   currentapprovalamount + Unbanked + Number_Households + jobsreported + forgivenessamount + 
                   cb_1 + cb_0 + FintechPartnership_1 + FintechPartnership_0, data=df)
summary(lmmodel1)
# currentapprovalamount, cb_0, FintechPartnership_0: Insignificant

# lmmodel2: Update from lmmodel1
lmmodel2 = lm(Sum_Approval.Branch~Asset + LoanToAsset + Office_Count + CoreRatio + State_Count + 
                Unbanked + Number_Households + jobsreported + forgivenessamount + 
                cb_1 + FintechPartnership_1, data=df)
summary(lmmodel2)
# Adjusted R-squared:  0.7693




                   
                   
