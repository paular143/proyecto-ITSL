#PROYECTO STATISTICAL LEARNING

#MODELOS REGRESION
#=========================================================================================================

datosReg = read.csv("trainReg_limpia.csv") #Leer datos entrenamiento ya limpios
testReg=read.csv("testReg_limpia.csv") #Leer datos del test ya limpios

#=========================================================================================================
#Division de los datos

names(datosReg)[3]='State'
names(testReg)[2]='State'

#Para entrenar el modelo con toda la base de entrenamiento correr esta linea:
train = datosReg

#Para separar la base de Train en train y un test auxliar correr:
library(caTools)
set.seed(123)
spr = 3/4
split = sample.split(datosReg[,1], SplitRatio=spr )
train=subset(datosReg, split==T)
test=subset(datosReg, split==F)

#Imputacion de NA en base testReg
library(missForest)
set.seed(123)
imp = missForest(testReg[,])
testReg = data.frame(imp$ximp)

testReg$State[110]='KY'


#=========================================================================================================
#MODELOS 
#=========================================================================================================
#REGRESION LINEAL
#Score en Kaggle: 1288195.94990


testReg=data.frame("Indice"=seq(1),testReg)

X=data.frame(model.matrix(Indice~.,train[,])[,-1])
Xtest=data.frame(model.matrix(Indice~.,test[,])[,-1])
XtestReg=data.frame(model.matrix(Indice~.,testReg[,])[,-1])

X[,c(8,9,47)]=NULL # correr si train = datosReg
XtestReg$StateKS = NULL
reg=lm(train$Indice~.,data=X) #Modelo de regresion lineal. Para no meter State poner -c(1,3)

#Correr solo si se hizo la division de test auxiliar
pred_lineal=predict(reg,Xtest)
mseregnormal=mean((pred_lineal-test$Indice)^2)
mseregnormal 
#Sin modificar variables y sin meter estado MSE = 951444.5
#Sin modificar variables y metiendo estado MSE = 1221554
#El MSE haciendo log10 y log(1+x) y quitando outliers con score 1.7 es 1028617   
#Modificando variables e Indice y metiendo estado MSE = 642872.8 (sumando 2415)

#Prediccion en base testReg (Resultados)
RespuestaRegLineal=predict(reg,XtestReg) #-2 para no meter State
k=data.frame(1:709,RespuestaRegLineal)
write.table(k,file="reg_lineal.csv",sep=",",qmethod="double",row.names=FALSE,col.names = c('ID','Indice'))


#=========================================================================================================
#LASSO 
#Score en Kaggle: 914081.32580 (Metiendo estados)

library(glmnet)
testReg=data.frame("Indice"=seq(1),testReg)

X=model.matrix(Indice~.,train[,])[,-1]
Xtest=model.matrix(Indice~.,test[,])[,-1]
XtestReg=model.matrix(Indice~.,testReg[,])[,-1]


#Correr estas lineas si train = datosReg
X=data.frame(X)
X[,c(14)]=NULL 
X=as.matrix(X)


#Calibracion
set.seed(123)
cvmod=cv.glmnet(X,train$Indice,alpha=0.5)
cvmod$lambda.min
plot(cvmod)

#Grafico de betas
mod_pen2_plot=glmnet(X,train$Indice,alpha=0.5)
plot(mod_pen2_plot,xvar=c("lambda"))
#plot(mod_pen2_plot,xvar=c("dev"))

mod_pen2=glmnet(X,train$Indice,alpha=0.5,lambda=cvmod$lambda.min)  #MODELO LASSO


#Correr solo si se hizo la division de test auxiliar
pred_lasso=predict(mod_pen2,Xtest)
msep2=mean((test$Indice-pred_lasso)^2)
msep2
#Lasso (alpha=1) sin modificar variables MSE = 1032452 (seed 111)
#Lasso (alpha=1) haciendo log10 y log(x+1) MSE = 1133614 (seed 111)
#Lasso (alpha=1) haciendo todas las modificaciones MSE = 1654616 (seed 111)

#Haciendo Lasso (alpha = 1) y modificando Indice el MSE es 748298.3
#Modificando Indice y modificaciones a variables (63,62) el MSE es 706162.1

#Prediccion en base testReg (Resultados)
RespuestaLasso = predict(mod_pen2,XtestReg)
k=data.frame(1:709,RespuestaLasso)
write.table(k,file="reg_lasso.csv",sep=",",qmethod="double",row.names=FALSE,col.names = c('ID','Indice'))

#=========================================================================================================
#PCR
#Score en Kaggle: 974560.55922

library(pls)
testReg=data.frame("Indice"=seq(1),testReg)

X=model.matrix(Indice~.,train[,])[,-1]
Xtest=model.matrix(Indice~.,test[,])[,-1]
XtestReg=model.matrix(Indice~.,testReg[,])[,-1]

#Correr estas lineas si train = datosReg
XtestReg=XtestReg[,-14]
X=X[,-c(8,9,47)]

X=data.frame(X)
Xtest=data.frame(Xtest)
XtestReg = data.frame(XtestReg)

lm2=pcr(train$Indice~.,data=X,scale=F,validation="CV")

mses = rep(0,109)
#Calibracion para ncomp
for(i in 1:109){
  pred_pcr=predict(lm2,Xtest,ncomp=i)
  msePCR=mean((test$Indice-pred_pcr)^2)
  mses[i]=msePCR
}
which.min(mses)
#se minimiza con ncomp = 12 (sin modificar variables)
#se minimiza con ncomp = 11 (para variables modificadas)

#Correr solo si se hizo la division de test auxiliar
pred_pcr=predict(lm2,Xtest,ncomp=11)
msePCR=mean((test$Indice-pred_pcr)^2)
msePCR
#Sin modificar variables el MSE = 647470.5
#haciendo log10 y log(x+1) MSE = 701948
#haciendo log10 y log(x+1) y modif a var: 63,62. MSE = 823305.2

#Prediccion en base testReg (Resultados)
RespuestaPCR=predict(lm2,XtestReg,ncomp=12)
m=length(testReg[,2])
validationplot(lm2,val.type="MSEP",main="MSE FRENTE A COMPONENTES",col=14)
PCR1=data.frame(1:m,RespuestaPCR)
write.table(PCR1,file="reg_PCR.csv",sep=",",qmethod="double",row.names=FALSE,col.names = c('ID','Indice'))


#=========================================================================================================
#PLS

library(pls)

lm3=plsr(train$Indice~.,data=train[,-1],scale=T,validation="CV") #Modelo

#Calibrar ncomp para la prediccion
msePLS_calib = rep(0,109)
for(i in 1:109){
  predpp1=predict(lm3,test[,-1],ncomp=i)
  msePLS=mean((test$Indice-predpp1)^2)
  msePLS_calib[i]=msePLS
}
which.min(msePLS_calib) 
#se minimiza con ncomp = 14 (sin modificar variables)
#se minimiza con ncomp = 55 (para variables modificadas)

#Correr solo si se hizo la division de test auxiliar
predpp1=predict(lm3,test[,-1],ncomp=14)
msePLS=mean((test$Indice-predpp1)^2)
msePLS
#sin modificar variables el MSE = 1164477
#haciendo log10 y log(x+1) MSE = 1211128


#Prediccion en base testReg (Resultados)
RespuestaPLSR=predict(lm3,ncomp=14,newdata=testReg)
m=length(testReg[,2])
datosRta=data.frame(seq(1:m,m))
PLS1=data.frame(seq(from=1, to=m,length.out = m),RespuestaPLSR)
write.table(PLS1,file="regresion_PLSR.csv",sep=",",qmethod="double",row.names=FALSE,col.names = c('ID','Indice'))

#=========================================================================================================
#FORWARD

library(ISLR)
library(leaps)

testReg=data.frame("Indice"=seq(1),testReg)

regf=regsubsets(train$Indice~.,data=train[,-1],nvmax=108,method="forward") #Modelo

sumregf=summary(regf)
coef1=coef(regf,which.min(sumregf$cp))
which.min(sumregf$cp)
plot(sumregf$cp,type="b",col="red",main="CP vs numero de variables-Forward",xlab="componentes",ylab="CP")


#Correr solo si se hizo la division de test auxiliar
a1=model.matrix(test$Indice~.,data=test)
predTestf=a1[,names(coef1)]%*%coef1
mseForward=mean((test$Indice-predTestf)^2)
mseForward
#sin modificar variables el MSE = 2686576
#El MSE es 952926


#Prediccion en base testReg (Resultados)
z=length(testReg[,2])
a4=model.matrix(testReg$Indice~.,data=testReg)
RespuestaForward=a4[,names(coef1)]%*%coef1
k=data.frame(seq(from=1, to=m),RespuestaForward)
write.table(k,file="reg_forward.csv",sep=",",qmethod="double",row.names=FALSE,col.names = c('ID','Indice'))


#=========================================================================================================
#EXHAUSTIVE

library(ISLR)
library(leaps)

testReg=data.frame("Indice"=seq(1),testReg)

regexa=regsubsets(train$Indice~.,data=train[,-1],nvmax=1,method="exhaustive") #Modelo

sumregexa=summary(regexa)
coef1=coef(regexa,which.min(sumregexa$adjr2))
which.max(sumregexa$adjr2)


#Correr solo si se hizo la division de test auxiliar
a1=model.matrix(test$Indice~.,data=test)
predTestf=a1[,names(coef1)]%*%coef1
mseExhaustive=mean((test$Indice-predTestf)^2)
mseExhaustive
#sin modificar variables MSE = 



#Prediccion en base testReg (Resultados)
z=length(testReg[,2])
a4=model.matrix(testReg$Indice~.,data=testReg)
RespuestaExhaustive=a4[,names(coef1)]%*%coef1
k=data.frame(seq(from=1, to=m),RespuestaExhaustive)
write.table(k,file="reg_exhaustive.csv",sep=",",qmethod="double",row.names=FALSE,col.names = c('ID','Indice'))


#=========================================================================================================
#Generalized Additive Models GAMS

testReg=data.frame("Indice"=seq(1),testReg)

X=data.frame(model.matrix(Indice~.,train[,])[,-1])
Xtest=data.frame(model.matrix(Indice~.,test[,])[,-1])
XtestReg=data.frame(model.matrix(Indice~.,testReg[,])[,-1])

#Modelo
library(gam)
gams.fit=gam(train$Indice~.,data=X)
summary(gams.fit)


trainpred=predict(gams.fit, Xtest)
msegams=mean((trainpred-test$Indice)^2)
msegams
#Sin modificar variables y sin meter estado MSE = 951444.5
#Sin modificar variables metiendo estado MSE = 1221554n
#El MSE es 1028617 (Igual que el de reg lineal)


b=predict(gams.fit,datosTest)
help(predict)
b=data.frame(seq(from=1,to=m,length.out=m),b)
write.table(b,file="SUBIRGAMS.csv",sep=",",qmethod="double",row.names=FALSE)

#=========================================================================================================
#Combinacion lineal de lasso y PCR que son los que mejor han dado (Sin modificar variables)

alpha = seq(from=0, to=1, by=0.1)
lasso=data.frame(pred_lasso)
pcr = data.frame(pred_pcr)

mses = 1:11
for(i in alpha){
  pred_combinacion = i*(lasso)+(1-i)*pcr
  mse_combinacion = mean((test$Indice-pred_combinacion)^2)
  mses[i*10+1]=mse_combinacion
}
which.min(mses)

#=========================================================================================================
#CART

library(rpart)
library(rpart.plot)


mtree = rpart(Indice ~ ., data = train, method = "anova" )
rpart.plot(mtree, type = 3 ,digits = 3, fallen.leaves = TRUE)

pred_tree = predict(mtree,test[,-1])
mse = mean((pred_tree - test$Indice)^2)
mse

#Sin modificar variables MSE = 2736429

#=========================================================================================================
#Random Forest Regression

mses = 1:7
for(i in 1:7){
  mforest = randomForest(Indice ~ ., data = train, ntree=1000, nodesize = i)
  
  pred_forest = predict(mforest,test[,-1])
  mse = mean((pred_forest - test$Indice)^2)
  mses[i] = mse
}
#Con log10 y log(x+1), ntree = 1000, nodesize = 5, MSE = 1246566

mforest = randomForest(Indice ~ ., data = train, ntree=1000, nodesize = 5)
pred_forest = predict(mforest,test[,-1])
mse = mean((pred_forest - test$Indice)^2)
mse
varImpPlot(mforest,sort=TRUE,n.var=12)

levels(testReg$State) = levels(train$State)
RespuestaForest = predict(mforest,testReg)
k=data.frame(1:709,RespuestaForest)
write.table(k,file="reg_forest.csv",sep=",",qmethod="double",row.names=FALSE,col.names = c('ID','Indice'))
