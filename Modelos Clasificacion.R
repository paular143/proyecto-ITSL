#PROYECTO STATISTICAL LEARNING

#MODELOS CLASIFICACION
#=========================================================================================================
library(caret)
library(missForest)
library(caTools)
library(ISLR)
library(knitr)
library(MASS)
library(gbm)
library(pROC)
library(plotly)
#=========================================================================================================

datosClas = read.csv("trainClas_limpia.csv") #Leer datos entrenamiento ya limpios
testClas=read.csv("testClas_limpia.csv") #Leer datos del test ya limpios

#=========================================================================================================
#Division de los datos

names(datosClas)[3]='State'
names(testClas)[2]='State'

#Para separar la base de Train en train y un test auxliar correr:
set.seed(123)
spr = 3/4
split = sample.split(datosClas[,1], SplitRatio=spr )
train=subset(datosClas, split==T)
test=subset(datosClas, split==F)

#Para entrenar el modelo con toda la base de entrenamiento correr esta linea:
#train = datosClas

#Imputacion de NA en base testReg
set.seed(123)
imp = missForest(testClas[,])
testClas = data.frame(imp$ximp)

#Quitar lo de los estados
#testClas$State[110]="KY"
train[,3]=NULL
test[,3]=NULL
testClas[,2]=NULL
#=========================================================================================================
##################################################################################
# Regresion Logistica
#AUC=0.939
#Accuracy=0.900
model_log=glm(Clas~.,family=binomial(link='logit'),data=train)
pred_log=predict(model_log,test,type="response")
roc_log=roc(test$Clas,pred_log)
plot(roc_log)
auc_log=auc(roc_log)

pred_log <- ifelse(pred_log > 0.5,1,0)
confusionMatrix(test$Clas,pred_log)

#Prediccion en base testReg (Resultados)
Res_log=predict(model_log,testClas,type="response") 
Res_log <- ifelse(Res_log > 0.5,1,0)
k=data.frame(seq(from=1, to=dim(testClas)[1], by=1),Res_log)
write.table(k,file="Regresion_Logistica.csv",sep=",",qmethod="double",row.names=FALSE,col.names = c('ID','Clas'))

##################################################################################
#LDA
#Este modelo es mejor cuando se cumplen los supuestos de normalidad y varianzas iguales
#AUC=0.936
#Accuracy=0.8864
model_lda <- lda(Clas~., data=train)
pred_lda <- predict(model_lda, newdata=test)
pred_lda1=as.numeric(pred_lda$class)-1
pred_lda=pred_lda$posterior
pred_lda=pred_lda[,1]
roc_lda=roc(test$Clas,pred_lda)
plot(roc_lda)
auc_lda=auc(roc_lda)

confusionMatrix(test$Clas,pred_lda1)
#Prediccion en base testReg (Resultados)
Res_lda=predict(model_lda,testClas,type="response") 
Res_lda=as.numeric(Res_lda$class)-1
k=data.frame(seq(from=1, to=dim(testClas)[1], by=1),Res_lda)
write.table(k,file="LDA.csv",sep=",",qmethod="double",row.names=FALSE,col.names = c('ID','Clas'))

##################################################################################
#QDA
#AUC=0.908
#Accuracy=0.8809
model_qda <- qda(Clas~., method='moment',data=train)
pred_qda <- predict(model_qda, test)
pred_qda1=as.numeric(pred_qda$class)-1
pred_qda=pred_qda$posterior
pred_qda=pred_qda[,1]
roc_qda=roc(test$Clas,pred_qda)
plot(roc_qda)
auc_qda=auc(roc_qda)

confusionMatrix(test$Clas,pred_qda1)
#Prediccion en base testReg (Resultados)
Res_qda=predict(model_qda,testClas,type="response") 
Res_qda=as.numeric(Res_qda$class)-1
k=data.frame(seq(from=1, to=dim(testClas)[1], by=1),Res_qda)
write.table(k,file="QDA.csv",sep=",",qmethod="double",row.names=FALSE,col.names = c('ID','Clas'))

############################################################################
#Boosting
#AUC=0.943
#Accuracy=0.8892
model_boo=gbm(Clas~.,data=train,distribution="bernoulli",n.trees=1000,interaction.depth=3,shrinkage=0.1)
pred_boo=predict(model_boo,test,type="response",n.trees=500)
roc_boo=roc(test$Clas,pred_boo)
plot(roc_boo)
auc_boo=auc(roc_boo)

pred_boo <- ifelse(pred_boo > 0.5,1,0)
confusionMatrix(test$Clas,pred_boo)
#Prediccion en base testReg (Resultados)
Res_boo=predict(model_boo,testClas,type="response",n.trees=500) 
Res_boo <- ifelse(Res_boo > 0.5,1,0)
k=data.frame(seq(from=1, to=dim(testClas)[1], by=1),Res_boo)
write.table(k,file="Boosting.csv",sep=",",qmethod="double",row.names=FALSE,col.names = c('ID','Clas'))

############################################################################
#Componentes Principales
pc <- prcomp(train[,2:dim(train)[2]])
pctest<- predict(pc, newdata = test)
pctestClas<-predict(pc,newdata=testClas)

mydata <- data.frame(Clas = train[, 1], pc$x)
mydatatest<-data.frame(Clas=test[,1],pctest)
mydatatestClas<-data.frame(Clas=rep(1,dim(pctestClas)[1]),pctestClas)

desv=pc$sdev^2
desv=desv/sum(desv)
desv=desv[1:10]
graficar=data.frame(seq(from=1, to=length(desv), by=1),desv)
names(graficar)[1]='seq'

p = plot_ly(graficar, x = ~seq, y = ~desv, type = 'bar', name = 'Componentes', marker = list(color = "red")) %>%
  layout(xaxis= list(title = 'Componentes Principales'), yaxis = list(title = 'Porcentaje de varianza'), barmode = 'group')
p

#######################################################################################################################################
# LDA con Componentes Principales
#AUC=0.9361
#Accuracy=0.8864
model_lda_pc <- lda(Clas~., data=mydata)
pred_lda_pc <- predict(model_lda_pc, mydatatest)
pred_lda_pc1=as.numeric(pred_lda_pc$class)-1
pred_lda_pc=pred_lda_pc$posterior
pred_lda_pc=pred_lda_pc[,1]
roc_lda_pc=roc(test$Clas,pred_lda_pc)
plot(roc_lda_pc)
auc_lda_pc=auc(roc_lda_pc)
auc_lda_pc

confusionMatrix(test$Clas,pred_lda_pc1)
#Prediccion en base testReg (Resultados)
Res_lda_pc=predict(model_lda_pc,mydatatestClas,type="response")
Res_lda_pc=as.numeric(Res_lda_pc$class)-1
k=data.frame(seq(from=1, to=dim(testClas)[1], by=1),Res_lda_pc)
write.table(k,file="LDA_PC.csv",sep=",",qmethod="double",row.names=FALSE,col.names = c('ID','Clas'))

##################################################################################
# Regresion Logistica con Componentes Principales
#AUC=0.9387
#Accuracy=0.9003
mydata=mydata[,1:3]
mydatatest=mydatatest[,1:3]
model_log_pc=glm(Clas~.,family=binomial(link='logit'),data=mydata)
pred_log_pc=predict(model_log_pc,mydatatest,type="response")
roc_log_pc=roc(test$Clas,pred_log_pc)
plot(roc_log_pc)
auc_log_pc=auc(roc_log_pc)
auc_log_pc
pred_log_pc <- ifelse(pred_log_pc > 0.5,1,0)
confusionMatrix(test$Clas,pred_log_pc)

#Prediccion en base testReg (Resultados)
Res_log_pc=predict(model_log_pc,mydatatestClas,type="response") 
Res_log_pc <- ifelse(Res_log_pc > 0.5,1,0)
k=data.frame(seq(from=1, to=dim(testClas)[1], by=1),Res_log_pc)
write.table(k,file="Regresion_Logistica_PC.csv",sep=",",qmethod="double",row.names=FALSE,col.names = c('ID','Clas'))


##################################################################################
#Random Forest a muerte
library(randomForest)
library(pROC)
rf.datos=randomForest(Clas~.,data=train,mtry=8)
summary(rf.datos)
imp=importance(rf.datos)
print(imp)
varImpPlot(rf.datos)
pred_rf=predict(rf.datos,test,type="response")
roc_rf=roc(test$Clas,pred_rf)
plot(roc_rf)
auc_rf=auc(roc_rf)

pred_rf <- ifelse(pred_rf > 0.5,1,0)
confusionMatrix(test$Clas,pred_rf)

#Prediccion en base testReg (Resultados)
Res_RF=predict(rf.datos,testClas,type="response") 
Res_RF <- ifelse(Res_RF > 0.5,1,0)
k=data.frame(seq(from=1, to=dim(testClas)[1], by=1),Res_RF)
write.table(k,file="Random_Forest_a_muerte.csv",sep=",",qmethod="double",row.names=FALSE,col.names = c('ID','Clas'))

##################################################################################
#Support Vector Machines
#AUC=0.8659
#Accuracy=0.935
library(e1071)
svm_fit=svm(Clas~.,data=train)
summary(svm_fit)  #Resultado del modelo (significancia individual)
tune_svm=tune(svm,Clas~.,data=test)
svm_fit=svm(Clas~.,data=train, cost=0.1,gamma=0.1,probability=T,kernel="radial")
pred_svm=predict(svm_fit,test,probability=T,decision.values=F)
roc_svm=roc(test$Clas,predisvm)
plot(roc_svm)
auc_svm=auc(roc_svm)
pred_svm <- ifelse(pred_svm > 0.5,1,0)

#Prediccion en base testReg (Resultados)
Res_SVM=predict(svm_fit,testClas,type="response") 
Res_SVM <- ifelse(Res_SVM > 0.5,1,0)
k=data.frame(seq(from=1, to=dim(testClas)[1], by=1),Res_SVM)
write.table(k,file="Support_Vector_Machine.csv",sep=",",qmethod="double",row.names=FALSE,col.names = c('ID','Clas'))

