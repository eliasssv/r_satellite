#install.packages("mlbench")
#install.packages("caret")
library(mlbench)
library(caret)

#Carrega dataframe
data(Satellite)

df <- Satellite[c(17:20, 37)]

#conta nro de linhas
numRows <- nrow(df)

#seta o seed para que seja possível reproduzir
set.seed(1) 

#gera lista random de números 
indexes <- createDataPartition(df$classes, p = 0.80, list = FALSE)

#DFs de treino e teste
df.train <- df[indexes,]
df.test  <- df[-indexes,]

#### RANDOM FOREST ####
set.seed(1) 
model.rf <- train(classes~., 
                  data=df.train, 
                  method="rf",
                  trControl = trainControl("cv", number = 10),
                  preProcess = c("center","scale")
)
predicted.rf <-predict(model.rf, df.test)
cm.rf <- confusionMatrix(data=predicted.rf, reference = df.test$classes)
cm.rf #Accuracy : 0.8442 

#### SVM ####
set.seed(1) 
model.svm <- train(classes~., 
                   data=df.train, 
                   method="svmRadial",
                   trControl = trainControl("cv", number = 10),
                   preProcess = c("center","scale")
)
predicted.svm <-predict(model.svm, df.test)
cm.svm <- confusionMatrix(data=predicted.svm, reference = df.test$classes)
cm.svm #Accuracy : 0.8567  

#### RNA ####
set.seed(1) 
model.rna <- train(classes~., 
                   data=df.train, 
                   method="nnet",
                   trControl = trainControl("cv", number = 10),
                   preProcess = c("center","scale"), 
                   trace=FALSE
)
predicted.rna <-predict(model.rna, df.test)
cm.rna <- confusionMatrix(data=predicted.rna, reference = df.test$classes)
cm.rna #Accuracy : 0.8544

## MELHOR -> SVM - Executar com todos os dados
library('kernlab')
print(model.svm)
# sigma = 0.9749486
model.final <- ksvm(type="C-svc", classes~., data=Satellite, kernel="rbfdot",
                    C=1.0, kpar=list(sigma=0.9749486))
predicted.final <- predict(model.final, Satellite)

cm.final <- confusionMatrix(predicted.final, Satellite$classes)
cm.final

#Salva modelo final
saveRDS(model.final, "satellite_svm.rds")