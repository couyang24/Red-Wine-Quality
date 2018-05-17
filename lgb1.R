if (!require("pacman")) install.packages("pacman")
pacman::p_load(tidyverse, skimr, GGally, plotly, viridis, caret, randomForest, e1071, rpart, xgboost, h2o, corrplot, rpart.plot)

wine <- read_csv("../input/winequality-red.csv")

colnames(wine) <- wine %>% colnames() %>% str_replace_all(" ","_")
wine$quality <- as.factor(wine$quality)

set.seed(1)
inTrain <- createDataPartition(wine$quality, p=.9, list = F)

train <- wine[inTrain,]
test <- wine[-inTrain,]
rm(inTrain)

library(microbenchmark, quietly=TRUE)
library(lightgbm, quietly=TRUE)
lgb.train = lgb.Dataset(as.matrix(train[, colnames(train) != "quality"]), label = train$quality)
lgb.test = lgb.Dataset(as.matrix(test[, colnames(test) != "quality"]), label = test$quality)

params.lgb = list(
  objective = "binary"
  , metric = "auc"
  , min_data_in_leaf = 1
  , min_sum_hessian_in_leaf = 100
  , feature_fraction = 1
  , bagging_fraction = 1
  , bagging_freq = 0
)

# Get the time to train the lightGBM model
lgb.bench = microbenchmark(
  lgb.model <- lgb.train(
    params = params.lgb
    , data = lgb.train
    #, valids = list(test = lgb.test)
    , learning_rate = 0.1
    , num_leaves = 7
    , num_threads = 2
    , nrounds = 500
    #, early_stopping_rounds = 40
    , eval_freq = 20
  )
  , times = 5L
)
print(lgb.bench)
print(max(unlist(lgb.model$record_evals[["test"]][["auc"]][["eval"]])))

# get feature importance
lgb.feature.imp = lgb.importance(lgb.model, percentage = TRUE)

# make test predictions
lgb.test = predict(lgb.model, newdata =  as.matrix(test[, colnames(test) != "quality"]), n = lgb.model$best_iter)
confusionMatrix(lgb.test, test$quality)

# auc.lgb = roc(test$Class, lgb.test, plot = TRUE, col = "green")
# print(auc.lgb)
library(data.table)
fw
