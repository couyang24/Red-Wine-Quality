library(tidyverse)
library(skimr)
library(GGally)
library(corrplot)
library(plotly)
library(viridis)
library(caret)
library(randomForest)
library(e1071)
library(rpart)
library(rattle)
library(rpart.plot)
library(xgboost)
library(h2o)

wine <- read_csv("winequality-red.csv")

wine %>% skim() %>% kable()

colnames(wine) <- wine %>% colnames() %>% str_replace_all(" ","_")



wine %>% 
  mutate(quality = as.factor(quality)) %>% 
  select(-c(residual_sugar, free_sulfur_dioxide, total_sulfur_dioxide, chlorides)) %>% 
  ggpairs(aes(color = quality,alpha=0.4),
          columns=1:7,
          lower=list(continuous="points"),
          upper=list(continuous="blank"),
          axisLabels="none", switch="both")

wine %>% cor() %>% corrplot.mixed(upper = "ellipse", tl.cex=.8, tl.pos = 'lt', number.cex = .8)

wine %>% corrgram(lower.panel=panel.shade, upper.panel=panel.ellipse)

wine$quality <- as.factor(wine$quality)

wine %>% 
  plot_ly(x=~alcohol,y=~volatile_acidity,z= ~sulphates, color=~quality, hoverinfo = 'text', colors = viridis(3),
          text = ~paste('Quality:', quality,
                        '<br>Alcohol:', alcohol,
                        '<br>Volatile Acidity:', volatile_acidity,
                        '<br>sulphates:', sulphates)) %>% 
  add_markers(opacity = 0.8) %>%
  layout(title = "3D Wine",
         annotations=list(yref='paper',xref="paper",y=1.05,x=1.1, text="quality",showarrow=F),
         scene = list(xaxis = list(title = 'Alcohol'),
                      yaxis = list(title = 'Volatile Acidity'),
                      zaxis = list(title = 'sulphates')))


set.seed(1)
inTrain <- createDataPartition(wine$quality, p=.9, list = F)

train <- wine[inTrain,]
valid <- wine[-inTrain,]
rm(inTrain)

# rpart
set.seed(1)
rpart_model <- rpart(quality~alcohol+volatile_acidity+citric_acid+
                   density+pH+sulphates, train)

rpart.plot(rpart_model)

# fancyRpartPlot(rpart_model)

rpart_result <- predict(rpart_model, newdata = valid[,!colnames(valid) %in% c("quality")],type='class')

confusionMatrix(valid$quality,rpart_result)

varImp(rpart_model) %>% kable()

rm(rpart_model, rpart_result)

# randomforest
set.seed(1)
rf_model <- randomForest(quality~alcohol+volatile_acidity+citric_acid+
                           density+pH+sulphates,train)
rf_result <- predict(rf_model, newdata = valid[,!colnames(valid) %in% c("quality")])

confusionMatrix(valid$quality,rf_result)

varImp(rf_model) %>% kable()

varImpPlot(rf_model)

rm(rf_model, rf_result)

# svm
set.seed(1)
svm_model <- svm(quality~alcohol+volatile_acidity+citric_acid+
                           density+pH+sulphates,train)
svm_result <- predict(svm_model, newdata = valid[,!colnames(valid) %in% c("quality")])

confusionMatrix(valid$quality,svm_result)
rm(svm_model, svm_result)

# xgboost
data.train <- xgb.DMatrix(data = data.matrix(train[, !colnames(valid) %in% c("quality")]), label = train$quality)
data.valid <- xgb.DMatrix(data = data.matrix(valid[, !colnames(valid) %in% c("quality")]))


finalresult <- data.frame()

for(j in 1:10){
  result_list <- vector()
  for(i in 1:100){
    parameters <- list(
      # General Parameters
      booster            = "gbtree",          # default = "gbtree"
      silent             = 0,                 # default = 0
      # Booster Parameters
      eta                = 0.1,               # default = 0.2, range: [0,1]
      gamma              = j/10,                 # default = 0,   range: [0,???]
      max_depth          = 8,                 # default = 5,   range: [1,???]
      min_child_weight   = 2,                 # default = 2,   range: [0,???]
      subsample          = .8,                 # default = 1,   range: (0,1]
      colsample_bytree   = .8,                 # default = 1,   range: (0,1]
      colsample_bylevel  = 1,                 # default = 1,   range: (0,1]
      lambda             = 1,                 # default = 1
      alpha              = 0,                 # default = 0
      # Task Parameters
      objective          = "multi:softmax",   # default = "reg:linear"
      eval_metric        = "merror",
      num_class          = 7,
      seed               = 1               # reproducability seed
    )
  
  xgb_model <- xgb.train(parameters, data.train, nrounds = 100)
  
  xgb_pred <- predict(xgb_model, data.valid)
  
  (result <- confusionMatrix(as.factor(xgb_pred+2), valid$quality))
  
  result_list[i] <- result$overall[1]
  
  rm(xgb_model, xgb_pred, parameters)
  }
  finalresult[j,1] <- result_list %>% mean()
}

finalresult









rm(xgb_model, xgb_pred, data.train, data.valid, parameters)





# h2o
h2o.init()
h2o.train <- as.h2o(train)
h2o.valid <- as.h2o(valid)


h2o.model <- h2o.deeplearning(x = setdiff(names(train), c("quality")),
                              y = "quality",
                              training_frame = h2o.train,
                              # activation = "RectifierWithDropout", # algorithm
                              # input_dropout_ratio = 0.2, # % of inputs dropout
                              # balance_classes = T,
                              # momentum_stable = 0.99,
                              # nesterov_accelerated_gradient = T, # use it for speed
                              epochs = 1000,
                              standardize = TRUE,         # standardize data
                              hidden = c(100, 100),       # 2 layers of 00 nodes each
                              rate = 0.05,                # learning rate
                              seed = 1                # reproducability seed
)

h2o.predictions <- h2o.predict(h2o.model, h2o.valid) %>% as.data.frame()

confusionMatrix(h2o.predictions$predict, valid$quality)
rm(h2o.model, h2o.train, h2o.valid, h2o.predictions)
