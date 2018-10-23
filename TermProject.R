library(ROSE)
library(ROCR)
library(glmnet)
library(e1071)
library(caret)
library(rpart)
library(readr)


termProject <- read_csv("~/termProject.csv", 
                        col_types = cols(EDUCATION = col_factor(levels = c("1","2", "3", "4", "5", "6")), 
                                         MARRIAGE = col_factor(levels = c("1", "2", "3")),
                                         PAY_0 = col_factor(levels = c("-2", "-1", "0", "1", "2", "3", "4", "5", "6", "7", "8", "9")),
                                         PAY_2 = col_factor(levels = c("-2",  "-1", "0", "1", "2", "3", "4", "5", "6", "7", "8", "9")),
                                         PAY_3 = col_factor(levels = c("-2", "-1", "0", "1", "2", "3", "4", "5",  "6", "7", "8", "9")),
                                         PAY_4 = col_factor(levels = c("-2", "-1", "0", "1", "2", "3", "4", "5", "6", "7", "8", "9")),
                                         PAY_5 = col_factor(levels = c("-2", " -1", "0", "1", "2", "3", "4", "5", "6", "7", "8", "9")),
                                         PAY_6 = col_factor(levels = c("-2", "-1", "0", "1", "2", "3", "4", "5",  "6", "7", "8", "9")),
                                         SEX = col_factor(levels = c("1", "2")),
                                         default = col_factor(levels = c("0", "1"))
                        )
)
dim(termProject)
sum(is.na(termProject))
termProject=na.omit(termProject) # remove NA data
dim(termProject)
sum(is.na(termProject))



#  50% training sets and 50% test sets
set.seed(1)
train=sample(1:nrow(termProject),0.5*nrow(termProject))
test=(-train)
termProject.train=termProject[train,]
summary(termProject.train$default)
summary(termProject[test,]$default)

library(ROSE)
tra_data_over <- ovun.sample(default ~ ., data = termProject.train, method = "over",N = 18902)$data
summary(tra_data_over$default)
tra_data_both <- ovun.sample(default ~ ., data = termProject.train, method = "both",N = 10000)$data
summary(tra_data_both$default)
tra_data_rose <- ROSE(default ~ ., data = termProject.train, seed = 1)$data
summary(tra_data_rose$default)


library(randomForest)
#build random forest models
 set.seed(10)
 tree.rose = randomForest(default~., mtry=5,  importance =TRUE, data = tra_data_rose)
 tree.over = randomForest(default~., mtry=5,  importance =TRUE, data = tra_data_over)
 tree.both = randomForest(default~., mtry=5,  importance =TRUE, data = tra_data_both)

#make predictions on test set
 rose.pr = predict(tree.rose , type="prob", termProject[test,])[,2]
 rose.pred = prediction(rose.pr, y.test)
 over.pr = predict(tree.over , type="prob", termProject[test,])[,2]
 over.pred = prediction(over.pr, y.test)
 both.pr = predict(tree.both , type="prob", termProject[test,])[,2]
 both.pred = prediction(both.pr, y.test)
 

#AUC ROSE
 rose.perf = performance(rose.pred,"tpr","fpr")
 plot(rose.perf, col=1,lwd=2)
 performance(rose.pred,"auc")@y.values[[1]]
#AUC Oversampling
 over.perf = performance(over.pred,"tpr","fpr")
 plot(over.perf, add=TRUE, col=2,lwd=2)
 performance(over.pred,"auc")@y.values[[1]]
#AUC Both
 both.perf = performance(both.pred,"tpr","fpr")
 plot(both.perf, add=TRUE, col=3,lwd=2)
 performance(both.pred,"auc")@y.values[[1]]
 
legend("bottomright", legend=c("SMOTE: AUC = 0.7674", "Over: AUC = 0.7852","Both: AUC = 0.7813"),
        col=c(1,2,3), lwd=2)

# According to the AUC value, we choose tra_data_both as the training set in our project
train_set= tra_data_over
summary(train_set)



####----------------- Random Forest ---------------#########
set.seed(10)
rf.fit = tree.over
yhat.rf = predict(rf.fit ,termProject[test,])
confusionMatrix(reference=y.test,data=yhat.rf, positive="1")



#### ---------   Feature Selection 
grid=10^seq(10,-2,length=100)
#produce a matrix corresponding to the predictors
x=model.matrix(default~.,train_set)[,-1]
y= train_set$default

lasso.mod=glmnet(x,y,family = "binomial",alpha=1,lambda=grid)

#10-flod cross validation
lasso.cv.out=cv.glmnet(x,y,family = "binomial",alpha=1)
#plots
par(mfrow=c(2,1))
plot(lasso.mod, xvar="dev", main="Lasso")
plot(lasso.cv.out, main="Lasso")

#---- coefficients
lasso.bestlam=lasso.cv.out$lambda.1se
lasso.bestlam
best.lasso.coef=predict(lasso.cv.out,type="coefficients",s=lasso.bestlam)[1:89,]
#best.lasso.coef
best.lasso.coef[best.lasso.coef!=0]



#####----------- Logistic Regression -------------####

#----find error rate for logstic Regression
x.test=model.matrix(default~.,termProject[test,])[,-1]
y.test= termProject[test,]$default

lasso.probs=predict(lasso.mod,type="response",s=lasso.bestlam,newx=x.test)
lasso.pred=rep(0,length(y.test))
lasso.pred[lasso.probs>.5]=1
mean(lasso.pred==y.test)
error.rate01= mean(lasso.pred!=y.test)
error.rate01


# confusion Matrix table
confusionMatrix(reference=as.factor(y.test),data=as.factor(lasso.pred), positive="1")



######----------------- SVM ---------------------##########


##########-------------Best-first search -------------####
library(FSelector)
library(rpart)

evaluator <- function(subset) {
  #k-fold cross validation
  k <- 5
  splits <- runif(nrow(train_set))
  results = sapply(1:k, function(i) {
    test.idx <- (splits >= (i - 1) / k) & (splits < i / k)
    train.idx <- !test.idx
    test <- train_set[test.idx,]
    train <- train_set[train.idx,]
    tree <- rpart(as.simple.formula(subset, "default"), train)
    error.rate = sum(test$default != predict(tree, test, type="c")) / nrow(test)
    return(1 - error.rate)
  })
  print(subset)
  print(mean(results))
  
  return(mean(results))
}

set.seed(100)
subset <- best.first.search(names(train_set)[-24], evaluator)
#f <- as.simple.formula(subset, "default")
print(subset)

#selectedVariable= c( "LIMIT_BAL" "SEX"       "PAY_0"     "PAY_4"    )
selectedVariable=subset

dat = data.frame(x=train_set[,selectedVariable], y= train_set$default)
testdat= data.frame(x=termProject[test,selectedVariable], y=y.test)

#------  RBF Kernel --------------------
svmfit.best = svm(y~., data=dat, kernel = "radial", gamma=1, cost=10)
ypred = predict(svmfit.best, testdat)
confusionMatrix(reference=y.test,data=ypred, positive="1")



set.seed(101)
tune.out= tune(svm, y~., data=dat, kernel = "radial", 
               ranges = list(cost=c(0.1, 1, 10, 100), gamma= c (0.5, 1, 2, 3))
               )

summary (tune.out)

svmfit.best = svm(y~., data=dat, kernel = "radial", gamma=3, cost=100)
SVM.pred = predict(svmfit.best, testdat)
confusionMatrix(reference=y.test,data=SVM.pred, positive="1")


#------  Linear Kernel --------------------
svmfit.best2 = svm(y~., data=dat[train,], kernel = "linear", cost=1)
ypred = predict(svmfit.best2, testdat)
confusionMatrix(reference=y.test,data=ypred, positive="1")

set.seed(200)
l.tune.out=tune(svm , y~., data=dat, kernel ="linear",
              ranges = list(cost=c(0.1, 1, 10, 100, 1000)))
summary(l.tune.out)

svmfit.best2 = svm(y~., data=dat, kernel = "linear", cost=1000, probability = TRUE)
ypred = predict(svmfit.best2, testdat)
confusionMatrix(reference=y.test,data=ypred, positive="1")






#########-------------   ROC Curve    --------------##########
library(ROCR)

# ROC for Logistic Regression
lr.pr=lasso.probs
lr.pred <- prediction(lr.pr,y.test)
lr.perf <- performance(lr.pred, measure = "tpr", x.measure = "fpr")   
# plot the curve
par(mfrow=c(1,1))
plot(lr.perf,main="ROC Curve",col=4,lwd=2) 
abline(a=0,b=1,lwd=2,lty=2,col="gray")
# AUC value
lr.auc0= performance(lr.pred,"auc")
lr.auc= lr.auc0@y.values[[1]]

# ROC for SVM
svm.pr = predict(svmfit.best2, testdat, probability = TRUE)

svm.pr2=attr(svm.pr, "probabilities")[,2]

#svm.pr = predict(svmfit.best2, type="prob", testdat)[,2]
svm.pred = prediction(svm.pr2, y.test)
svm.perf = performance(svm.pred,"tpr","fpr")
plot(svm.perf,add=TRUE,col=2,lwd=2)
svm.auc <- performance(svm.pred,"auc")@y.values[[1]]

# ROC for random forest

rf.pr = predict(rf.fit , type="prob", termProject[-train ,])[,2]
rf.pred = prediction(rf.pr, y.test)
rf.perf = performance(rf.pred,"tpr","fpr")

plot(rf.perf,add=TRUE, col=3,lwd=2)
rf.auc <- performance(rf.pred,"auc")@y.values[[1]]

rf.auc
lr.auc
svm.auc

legend("bottomright", legend=c("Logistic: AUC = 0.7827", "SVM: AUC = 0.7677","RF: AUC = 0.7852"),
       col=c(4,2,3), lwd=2)

