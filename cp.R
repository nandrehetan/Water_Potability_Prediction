#COURSE PROJ----
cp<- read.csv("water_potability.csv")

#NA values
str(cp)
summary(cp)

#impute missing values using knn
library(DMwR2)
cp<-knnImputation(cp, k = 3, scale = TRUE, meth = "weighAvg",distData = NULL)

#remove outliers using boxplot
for (i in 2:10) {
  cp<-cp[!cp$ph %in% boxplot.stats(cp$ph)$out,]
  cp<-cp[!cp$Hardness %in% boxplot.stats(cp$Hardness)$out,]
  cp<-cp[!cp$Solids %in% boxplot.stats(cp$Solids)$out,]
  cp<-cp[!cp$Chloramines %in% boxplot.stats(cp$Chloramines)$out,]
  cp<-cp[!cp$Sulfate %in% boxplot.stats(cp$Sulfate)$out,]
  cp<-cp[!cp$Conductivity %in% boxplot.stats(cp$Conductivity)$out,]
  cp<-cp[!cp$Organic_carbon %in% boxplot.stats(cp$Organic_carbon)$out,]
  cp<-cp[!cp$Trihalomethanes %in% boxplot.stats(cp$Trihalomethanes)$out,]
  cp<-cp[!cp$Turbidity %in% boxplot.stats(cp$Turbidity)$out,]
  
}

#class imbalance
dim(cp)
head(cp)
table(cp$Potability)
prop.table(table(cp$Potability))

#smote
library(smotefamily)
smote_out=SMOTE(X=cp,target=cp$Potability,K=3,dup_size =1)
cp=smote_out$data
cp<-cp[,-11]

table(cp$Potability)
prop.table(table(cp$Potability))

cp$Potability=as.factor(cp$Potability)

#create training and testing data partitions
library(caret)
set.seed(9999)
cp<-cp[sample(1:nrow(cp)), ]
train <- createDataPartition(cp[,"Potability"],p=0.8,list=FALSE)
trn <- cp[train,]
tst <- cp[-train,]

#Algorithms applying
ctrl<-trainControl(method = "cv",number = 10)

#Decision Trees
set.seed(9999)
dec1<-train(Potability~.,data = trn,method="rpart",trControl=ctrl,tuneGrid = expand.grid(cp = 0.001))#cp - hyperparameter
pred_1<-predict(dec1,tst)
confusionMatrix(table(tst[,"Potability"],pred_1))

#Random forest
set.seed(9999)
rand1<-train(Potability~.,data = trn,method="rf",trControl=ctrl,tuneGrid = expand.grid(mtry = 3.16))#hyperparameter - mtry
pred_2<-predict(rand1,tst)
confusionMatrix(table(tst[,"Potability"],pred_2))

#Xgb linear
set.seed(9999)
xgb_lin<-train(Potability~.,data=trn,method="xgbLinear",trControl=ctrl,tuneGrid=expand.grid(eta = 0.3,nrounds=150,lambda=0.1,alpha=0.1))
pred_3<-predict(xgb_lin,tst)
confusionMatrix(table(tst[,"Potability"],pred_3))

#XGboost tree
set.seed(9999)
xgb_grid_1 = expand.grid(nrounds=150,eta=0.3, gamma=0.2, max_depth=6, min_child_weight=1, subsample=1, colsample_bytree=1)
xgbTree1<-train(Potability~.,data=trn,method="xgbTree",trControl=ctrl,tuneGrid = xgb_grid_1)
pred_4<-predict(xgbTree1,tst)
confusionMatrix(table(tst[,"Potability"],pred_4))

#SVM Radial
set.seed(9999)
svm_r_grid =expand.grid(sigma = c(0.01, 0.015, 0.2),C = c(0.75, 0.9, 1, 1.1, 1.25))
svm_r<-train(Potability~.,data=trn,method="svmRadial",trControl=ctrl,tuneGrid = svm_r_grid)
pred_6<-predict(svm_r,tst)
confusionMatrix(table(tst[,"Potability"],pred_6))

#SVM Polynomial
set.seed(9999)
svm_p_grid =expand.grid(degree=2, scale=5, C=5)
svm_p<-train(Potability~.,data=trn,method="svmPoly",trControl=ctrl,tuneGrid = svm_p_grid)
pred_7<-predict(svm_p,tst)
confusionMatrix(table(tst[,"Potability"],pred_7))

#Adaboost 
set.seed(9999)
adagrid = expand.grid( mfinal = 100,coeflearn = c("Breiman", "Freund", "Zhu"),maxdepth = 30)
ada<-train(Potability~.,data=trn,method="AdaBoost.M1",trControl=ctrl, tuneGrid = adagrid)
pred_9<-predict(ada,tst)
confusionMatrix(table(tst[,"Potability"],pred_9))

#PRINCIPAL COMPONENT ANALYSIS----
# Load the water dataset
water<- read.csv("water_potability.csv")
# Prepare the data
#water <- na.omit(water) # Remove rows with missing values
library(VIM)
water<-kNN(water,k=5)

#remove outliers using boxplot
for (i in 1:9) {
  water<-water[!water$ph %in% boxplot.stats(water$ph)$out,]
  water<-water[!water$Hardness %in% boxplot.stats(water$Hardness)$out,]
  water<-water[!water$Solids %in% boxplot.stats(water$Solids)$out,]
  water<-water[!water$Chloramines %in% boxplot.stats(water$Chloramines)$out,]
  water<-water[!water$Sulfate %in% boxplot.stats(water$Sulfate)$out,]
  water<-water[!water$Conductivity %in% boxplot.stats(water$Conductivity)$out,]
  water<-water[!water$Organic_carbon %in% boxplot.stats(water$Organic_carbon)$out,]
  water<-water[!water$Trihalomethanes %in% boxplot.stats(water$Trihalomethanes)$out,]
  water<-water[!water$Turbidity %in% boxplot.stats(water$Turbidity)$out,]
  
}

#pca
mydata <- water[, 1:10] # Select the numerical variables to apply PCA to
pca_result <- prcomp(mydata, center = TRUE, scale. = TRUE) # Apply PCA and save the results to res.pca
mydata_pca <- predict(pca_result, newdata = mydata)
data_pca<-data.frame (PC1=mydata_pca[,1],PC2=mydata_pca[,2],PC3=mydata_pca[,3],PC4=mydata_pca[,4],PC5=mydata_pca[,5],Potability=water$Potability)
data_pca$Potability<-as.factor(data_pca$Potability)

library(caret)
#create training and testing data partitions
set.seed(9999)
train <- createDataPartition(data_pca[,"Potability"],p=0.8,list=FALSE)
trn <- data_pca[train,]
tst <- data_pca[-train,]

#Algorithms applying
ctrl<-trainControl(method = "cv",number = 10)

#Decision Trees
set.seed(9999)
dec1<-train(Potability~.,data = trn,method="rpart",trControl=ctrl,tuneGrid = expand.gr id(cp = 0.001))#cp - hyperparameter
pred_1<-predict(dec1,tst)
confusionMatrix(table(tst[,"Potability"],pred_1))

#Random forest
set.seed(9999)
rand1<-train(Potability~.,data = trn,method="rf",trControl=ctrl,tuneGrid = expand.grid(mtry = 2))#hyperparameter - mtry
pred_2<-predict(rand1,tst)
confusionMatrix(table(tst[,"Potability"],pred_2))

#Xgb linear
set.seed(9999)
xgb_lin<-train(Potability~.,data=trn,method="xgbLinear",trControl=ctrl,tuneGrid=expand.grid(eta = 0.1,nrounds=150,lambda=0.5,alpha=0.5))
pred_3<-predict(xgb_lin,tst)
confusionMatrix(table(tst[,"Potability"],pred_3))

#XGboost tree
set.seed(9999)
xgb_grid_1 = expand.grid(nrounds = 150,max_depth = 10,eta = 0.3,gamma = 5,colsample_bytree = 0.9,min_child_weight = 10,subsample = 0.8)
xgbTree1<-train(Potability~.,data=trn,method="xgbTree",trControl=ctrl,tuneGrid = xgb_grid_1)
pred_4<-predict(xgbTree1,tst)
confusionMatrix(table(tst[,"Potability"],pred_4))

#SVM Radial
set.seed(9999)
svm_r_grid =expand.grid(sigma = c(0.01, 0.015, 0.2),C = c(0.75, 0.9, 1, 1.1, 1.25))
svm_r<-train(Potability~.,data=trn,method="svmRadial",trControl=ctrl,tuneGrid = svm_r_grid)
pred_6<-predict(svm_r,tst)
confusionMatrix(table(tst[,"Potability"],pred_6))

#SVM Polynomial
set.seed(9999)
svm_p_grid =expand.grid(degree=2, scale=5, C=5)
svm_p<-train(Potability~.,data=trn,method="svmPoly",trControl=ctrl,tuneGrid = svm_p_grid)
pred_7<-predict(svm_p,tst)
confusionMatrix(table(tst[,"Potability"],pred_7))

#Adaboost 
set.seed(9999)
adagrid = expand.grid( mfinal = 100,coeflearn = c("Breiman", "Freund", "Zhu"),maxdepth = 30)
ada<-train(Potability~.,data=trn,method="AdaBoost.M1",trControl=ctrl, tuneGrid = adagrid)
pred_9<-predict(ada,tst)
confusionMatrix(table(tst[,"Potability"],pred_9))

