library(tidyverse)

lgb_pred <- read_csv("lgb_pred.csv")

lgb_pred <- lgb_pred %>% mutate(prediction=x) %>% select(prediction)

pacman::p_load(tidyverse, skimr, GGally, plotly, viridis, caret, randomForest, e1071, rpart, xgboost, h2o, corrplot, rpart.plot, lightgbm, data.table)
wine <- read_csv("winequality-red.csv")
colnames(wine) <- wine %>% colnames() %>% str_replace_all(" ","_")

wine$quality <- as.numeric(as.factor(wine$quality)) - 1

set.seed(1)

inTrain <- createDataPartition(wine$quality, p=.9, list = F)

train <- wine[inTrain,]
valid <- wine[-inTrain,]
rm(inTrain)
n=1
valid$quality %>% str()

lgb2 <- (lgb_pred>.6) %>% which()

lgb2 %% 6 - 1 
