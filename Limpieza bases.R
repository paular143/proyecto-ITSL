#PROYECTO STATISTICAL LEARNING

#SELECCION DE VARIABLES Y LIMPIEZA DE DATOS
#=========================================================================================================

#(1) Para limpiar base REGRESION correr estas dos lineas:
train = read.csv("trainReg.csv")
test=read.csv("testReg.csv") 

#(2) Para limpiar base CLASIFICACION correr estas dos lineas:
train = read.csv("trainClas.csv") 
test = read.csv("testClas.csv") 

#=========================================================================================================

#Eliminar variables con faltantes
train[,1]=NULL #Eliminar primera columna que no es predictiva 
test[,1]=NULL

#Analisis de datos faltantes
library(VIM)
datosfaltantes= train[,sapply(colSums(is.na(train)),function(x) any(x > 0 ))]
aggr(datosfaltantes,combined=FALSE)

train = train[,!sapply(colSums(is.na(train)),function(x) any(x > 0 ))] #Eliminar variables con faltantes

#Eliminar variables correlacionadas
tmp =cor(train[,-1])
tmp[upper.tri(tmp)] = 0
diag(tmp) = 0
train = train[,!apply(tmp,2,function(x) any(abs(x) > 0.9 ))]

#Dejar en la base Test solo las variables que se seleccionaron en Train
test=test[,which(names(test)%in%names(train))]
test = test[,order(names(test))]
train=train[,order(names(train))] #la variable a predecir queda en la primera columna


#Outliers
library(DMwR)
outlier.scores = lofactor(train[,-c(1,3)], k=5)
outliers=as.numeric((outlier.scores>=1.7)) #Si consideramos un score mayor o igual a 1.7 hay 47 outliers
train = train[-which(outliers ==1),]


#Modificacion para tener histogramas simetricos    
par(mfrow=c(3,3))
for(i in c(1:2,4:dim(train)[2]))
{hist(train[,i], xlab = paste('Variable',i), main = paste('Histograma de variable',i))
}

  #Log10 para variables con hist no simetricos y rangos muy altos
  var_mod = c(60,66)
  train[,var_mod]=sapply(train[,var_mod],log10)
  test[,var_mod-1]=sapply(test[,var_mod-1],log10)
  
  #Log(1+x) para variables con hist no simetricos y rangos pequenos
  var_mod = c(25,31,32,38,44)
  train[,var_mod]=sapply(train[,var_mod],log1p)
  test[,var_mod-1]=sapply(test[,var_mod-1],log1p)
  
  #Log(x)
  var_mod = c(63)
  train[,var_mod]=sapply(train[,var_mod],log)
  test[,var_mod-1]=sapply(test[,var_mod-1],log)
  
  #Log(x) para variable 62
  var_mod = c(62)
  train[,var_mod]=sapply(train[,var_mod],function(x) log(x-10000))
  test[,var_mod-1]=sapply(test[,var_mod-1],function(x) log(x-10000))
  

#=========================================================================================================
#Guardar csv de bases limpias

#(1) Para guardar base limpia REGRESION correr estas dos lineas:
write.csv(train, file = "trainReg_limpia.csv", row.names = FALSE)
write.csv(test, file = "testReg_limpia.csv", row.names = FALSE)

#(2) Para guardar base limpia CLASIFICACION correr estas dos lineas:
write.csv(train, file = "trainClas_limpia.csv", row.names = FALSE)
write.csv(test, file = "testClas_limpia.csv", row.names = FALSE)

