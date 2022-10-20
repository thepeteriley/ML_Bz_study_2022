
###################################################################################

# regression model for Bz
# This can be used to recreate all the figures and tables from the 
# 2022 manuscript by Riley, Reiss, and Moestl. 
# Written by Pete Riley (pete@predsci.com)

# This is a pre-release version suitable for the editors and reviewers of the 
# manuscript. The final version will be cleaned and curated and made available
# via a GitHub repository. 

###################################################################################

# this function takes an input vector and renormalises it so that it runs 
# from 0 to 1. 

normalize <- function(x) {
  return ((x - min(x)) / (max(x) - min(x)))
}

###################################################################################

# MAIN PROGRAMME

# load libraries, some of them may not be needed for the main plots, but they 
# are all available from the "Tools -> Install packages..." option 

library(tidyverse)
library(caret)
library(randomForest)
library(varImp)
library(car)
library(RColorBrewer) # for the color palette
library(xgboost)
library(relaimpo)
library(earth)
library(DALEX)
library(MASS)
library(leaps)
library(stargazer)

# Set these options to print out summaries of the analysis to the screen 
# and as a LaTeX-formatted output for the manuscript. 

printSummary = T
printLatex = T

#####################################################################################

# get the parameter data
# During the initial analysis, I used data from the Reiss et al. 2021 paper, which had 348 observations 
# (i.e., ICMEs). 
# Chris had expanded the ICME database since that initial study, and provided an additional several
# hundred events. 
# Additionally, as well as using Bz min as the output variable, Chris also provided a dataset 
# with B-max, which had led to even higher associations in Martin's original paper, thus, 
# the analysis can be run on one of three datasets: 

dataset_A = "reiss21b-features-icme-sheath.csv" # original data with B-min target
dataset_B = "features_521_events_bz_target.csv" # revised data with B-min target
dataset_C = "features_521_events_btot_target.csv" # revised data with Btot target

# For the main results of the paper, the following line should be set to dataset_A
# However, to play with different options, which are summarized briefly in the 
# paper, set this dataset to dataset_B.
# I didn't pursue the analysis for Btot, however, since we have the data
# it's included as dataset_C for completeness. 

# Finally, as an option for the reader, a new dataset (say, D) could be 
# created that is some optimum version of _B, perhaps focusing on those 
# events that have the fasted ejecta. I would suspect that the 
# relationships would be even stronger for these events. However, this is 
# beyond the scope of this paper...but potentially useful if this 
# analysis is going to be reformulated into a forecast tool. 

dataset = read.csv(paste("/Users/pete/Dropbox/research/Bz_regress/",dataset_A,sep=""),header=T)

# although it probably doesn't make a difference for the ML techniques we're applying here, 
# it's probably good practice to strip the ID variable from the dataframe. 
# In principle, this could be a way that a particular algorithm could latch onto 
# recogising a particular example. 

dataset <- dataset[-1]

# print out table 1 to summarise the properties of the dataset 

if (printSummary) {
  summary(dataset)
}

if (printLatex) {
  stargazer(dataset)
}

#####################################################################################

# PLOTTING OPTIONS

# 1 = scatterplot

whPlot = 1

#####################################################################################

# Since the features' min/max values can vary significantly from one to the next 
# cf. speed (100's), field strength (1's), and temperature (1e5's), it might be 
# important to renormalize the data either by doing a min/max renormalisation
# or a z-score standardization. It turns out that none of them make a qualitative difference 
# in the results, and, in many cases, not a quantitative difference, 
# at least to within two sig. fig. However, for the purposes 
# of comparison, using either the min-max or z-score renormalization might 
# provide more information to the reader in the tables, so we should choose one of them.
# Since we're not splitting the data into a training and evaluation dataset, 
# there would be no issues with introducing a new observation that exceeds the 
# min/max of the training set, so having a bounded range for all of the 
# features would seem to be the most informative way to renormalise the 
# data. 
# However, if there are any outliers, then the min/max renormalization could 
# artificially compress datasets with outliers. From table 1, there is evidence 
# for some outliers, such as the max.vt value. Without going through each datapoint 
# separately, it would be better to avoid this issue, so we'll settle on 
# using the z-score renormalisation. 

# First capture the 'raw', un-normalised data. Need this for making 
# table 1, which is a summary of the explanatory parameters and their 
# variations. 
dataset_raw = dataset

# here is the renormalization based on min/max approach
dataset_n <- as.data.frame(lapply(dataset[1:(dim(dataset)[2])], normalize))

# here's the renormalization based on the z-score approach
dataset_z <- as.data.frame(scale(dataset[1:(dim(dataset)[2])]))

dataset <- dataset_z

#####################################################################################

# Build a linear model for Bz-min (or Bmax) using various techniques
# There are many techniques that can be applied from a simple regression 
# to several complicated ML approaches. 
# Specify here which approach to do. However, for all of them, an initial formula 
# must be created.
# wh_tech can be set as follows: 
# 1: simple multiple regression with some set of predefined explanatory (input) vars. 
# 2: stepwise regression model using MASS package
# 3: Random Forest Method
# 4: xgboost method
# 5: MARS method
# 6: Step-wise Regression Method
# 7: DALEX method

wh_tech = 1

#####################################################################################

# STATISTICAL ANALYSIS

#####################################################################################

# define several formulae for using in the analysis:

# here's the formula for the ALL predictors: 
formula1 = Target ~ .

# Here is the formula for the most significant variables from formula 1
formula2 = Target ~ mean.bt. +  std.bt. + cv.bt. + mean.vt. + cv.vt. +  mean.np. + max.np. + max.tp. + minmax.tp.

# This is a formula based on a subjective subset of variables after the initial round of analysis was complete.
# It can be adjusted to play with different combinations in an ad hoc way. 
#formula3 = Target ~ std.bz. + std.bt. + min.bz. + + mean.bz. + std.by. + mean.np. + minmax.vt. + max.by. + mean.bt. + min.by.
# formula3 = Target ~ mean.bt. + std.bt. + cv.bt. +  mean.np. + max.np. + max.tp.
# get rid of max.tp. for this analysis
formula3 = Target ~ mean.bt. + std.bt. + cv.bt. +  mean.np. + max.np.

# set the main formula to be analysed to be one of these three options. 
formula_anal = formula1 # this is the complete set of input variables. 

# technically, the simplest model is the following, but it's also needed 
# by the more sophisticated ML packages, so it's defined here.
# This needs to be tidied up for final release, but the two main 
# regressors are 1 (which is the _anal regressor) and 3, which is the subset 
# of the input variables that are most significant. 
regressor1 = lm(formula = formula1, data = dataset) 
regressor2 = lm(formula = formula2, data = dataset) 
regressor3 = lm(formula = formula3, data = dataset) 
regressor = lm(formula = formula_anal, data = dataset) 

#####################################################################################

if (wh_tech == 1) {  # Simplest technique - multiple regression

  print("Full set of Features:")
  
  if (printSummary) {
    summary(regressor)
  }
  
  if (printLatex) {
    stargazer(regressor)
  }
  
  print("Most significant five features from full ")
  if (printSummary) {
    summary(regressor3)
  }
  
  if (printLatex) {
    stargazer(regressor3)
  }
  
}

#####################################################################################

if (wh_tech == 2) {  # stepwise regression model using MASS package
  
  step.model <- stepAIC(regressor, direction = "both",trace = FALSE)
  
  if (printSummary) {
    summary(step.model)
  }
  if (printLatex) {
    stargazer(step.model)
  }

}

#####################################################################################

if (wh_tech == 3) {  # Random Forest Method
# Note that the "Random" part of this technique means that the same results 
# as in Table 5 will never be reproduced. However, the same trends should 
# always be there, particularly for the most important variables. 
  regressor_RF <- randomForest(formula = formula_anal,data = dataset,importance=TRUE) 
  output = caret::varImp(regressor_RF, conditional = TRUE) # conditional=True, adjusts for correlations between predictors
  output
  
}

#####################################################################################

if (wh_tech == 4) {  # xgboost method
  # as with the Random Forest, you get slightly different results
  # each time you run this. Got to love statistics!
  regressor_xg_1=train(formula1,data = dataset,method = "xgbTree", trControl = trainControl("cv", number = 10),scale=T)
  #regressor_xg_2=train(formula2,data = dataset,method = "xgbTree", trControl = trainControl("cv", number = 10),scale=T)
  output1 = caret::varImp(regressor_xg_1)
  output1
  #output2 = caret::varImp(regressor_xg_2)
  #output2
}

#####################################################################################

if (wh_tech == 5) {  # MARS
  
  regressor_MARS <- earth(formula_anal, data=dataset) # build model
  ev <- evimp (regressor_MARS) # estimate variable importance
  plot (ev)

}

#####################################################################################

if (wh_tech == 6) {  # Step-wise Regression Method
  
  base.mod <- lm(Target ~ 1 , data = dataset) # base intercept only model
  all.mod <- lm(formula_anal , data = dataset) # full model with all predictors
  stepMod <- step(base.mod, scope = list(lower = base.mod, upper = all.mod), direction = "both", trace = 1, steps = 1000) # perform step-wise algorithm
  #shortlistedVars <- names(unlist(stepMod[[1]])) # get the shortlisted variable.
  #shortlistedVars <- shortlistedVars[!shortlistedVars %in% "(Intercept)"] # remove intercept
  #shortlistedVars

}

#####################################################################################

if (wh_tech == 7) {  # DALEX Method

  regressor_DALEX <- randomForest(formula_anal,data = dataset, importance=TRUE) # fit the random forest with default parameter
  explained_rf <- explain(regressor_DALEX, data = dataset, y=dataset$Target) # Variable importance with DALEX
  varimps = variable_importance(explained_rf, type='raw') # Get the variable importances
  print(varimps)
  plot(varimps)

}

#####################################################################################
#####################################################################################

# PLOTS

#####################################################################################
#####################################################################################

if (whPlot == 1) {

  # dev.new()
  #formula = ~Target + min.np. + mean.np. + max.np. + mean.bt. + max.bt. + min.by.+ max.by.
  formula3 = ~Target + mean.bt. + std.bt. + cv.bt. +  mean.np. + max.np.
  #formula = formula1 
  #formula = formula3
  # note that the syntax for formula3 here is different than that 
  # required for the regressor analysis. This is, apparently a 
  # bug/feature, perhaps related to some legacy syntax. 
  scatterplotMatrix(formula3, data=dataset , 
                    #reg.line="" , smoother="", #col=my_colors , 
                    #smoother.args=list(col="grey") , cex=1.5 , 
                    #pch=c(15,16,17) , 
                    main="Scatterplot Matrix for significant explanatory variables")
  
}

#####################################################################################

