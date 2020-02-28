#################################################################
## Comparison with other methods - santner

#load package
library(penalized)
library(randomForest)
library(tgp)
library(MoEClust)
library(condmixt)

plist=c(1,seq(5,100,5),125,150,200)

# Lasso
MAE.l1=rep(0,length(plist))
MSE.l1=rep(0,length(plist))
EC.l1=rep(0,length(plist))
meancilength.l1=rep(0,length(plist))
mincilength.l1=rep(0,length(plist))
maxcilength.l1=rep(0,length(plist))
l1error.l1=rep(0,length(plist))

# RF
MAE.rf=rep(0,length(plist))
MSE.rf=rep(0,length(plist))

# TGP
MAE.tgp=rep(0,length(plist))
MSE.tgp=rep(0,length(plist))
EC.tgp=rep(0,length(plist))
meancilength.tgp=rep(0,length(plist))
mincilength.tgp=rep(0,length(plist))
maxcilength.tgp=rep(0,length(plist))
l1error.tgp=rep(0,length(plist))

# Single GP
MAE.bgp=rep(0,length(plist))
MSE.bgp=rep(0,length(plist))
EC.bgp=rep(0,length(plist))
meancilength.bgp=rep(0,length(plist))
mincilength.bgp=rep(0,length(plist))
maxcilength.bgp=rep(0,length(plist))
l1error.bgp=rep(0,length(plist))

# # MoE
# MAE.moe=rep(0,length(plist))
# MSE.moe=rep(0,length(plist))
# EC.moe=rep(0,length(plist))
# meancilength.moe=rep(0,length(plist))
# mincilength.moe=rep(0,length(plist))
# maxcilength.moe=rep(0,length(plist))
# l1error.moe=rep(0,length(plist))
# VI_dist.moe=matrix(0,length(plist),1)
# k.moe=matrix(0,length(plist),1)

## Mixture of NN
# MAE.cnn=rep(0,length(plist))
# MSE.cnn=rep(0,length(plist))
# EC.cnn=rep(0,length(plist))
# meancilength.cnn=rep(0,length(plist))
# mincilength.cnn=rep(0,length(plist))
# maxcilength.cnn=rep(0,length(plist))
# l1error.cnn=rep(0,length(plist))

for (p in plist){
  
  # Load training data
  dataname=paste('./iso_santner',p,'_mixture_train.csv',sep="")
  data=read.csv(dataname,header=F)
  y=data[,p+1]
  x=data[,1:p]
  y=as.matrix(y)
  x=as.matrix(x)
  N=dim(y)[1]
  X=cbind(rep(1,N), x)
  config_true=data[,p+2]+1
  
  par(mfrow=c(1,1),mar=c(2.5,2.5,1.5,.5)+.1, mgp=c(1.5, .5, 0))
  plot(x[,1],y)
  points(x[config_true==2,1],y[config_true==2],col=2)
  
  
  # Load test data
  testdataname=paste('./iso_santner',p,'_mixture_test.csv',sep="")
  testdata=read.csv(testdataname,header=F)
  ytest=testdata[,p+1]
  xtest=testdata[,1:p]
  ytest=as.matrix(ytest)
  xtest=as.matrix(xtest)
  Ntest=dim(xtest)[1]
  Xtest=cbind(rep(1,Ntest),xtest)
  
  #compute true density on a grid
  gridsize=0.01
  ygrid=seq(-3,3,gridsize)
  ngrid=length(ygrid)
  truedensity=matrix(0,Ntest, ngrid)
  truemean=matrix(0,Ntest,1)
  tau=1.2
  mu=c(2,6)
  xtestavg=apply(xtest,1,mean)
  pxtest=exp(-tau^2/2*(xtestavg-mu[1])^2)/(exp(-tau^2/2*(xtestavg-mu[1])^2)+exp(-tau^2/2*(xtestavg-mu[2])^2))
  beta1=c(0.1,0.6)
  beta2=c(-0.1,0.4)
  meantest=matrix(0,Ntest,2)
  meantest[,1]=exp(beta1[1]*xtestavg)*cos(beta1[2]*pi*xtestavg)
  meantest[,2]=exp(beta2[1]*xtestavg)*cos(beta2[2]*pi*xtestavg)
  sigma1=0.15
  sigma2=0.05
  for (ind in 1:ngrid){
    truedensity[,ind]=(1-pxtest)*dnorm(ygrid[ind],meantest[,1],sigma1)+(pxtest)*dnorm(ygrid[ind],meantest[,2],sigma2)
  }
  plot(xtestavg,ytest,pch="*")
  points(xtestavg[testdata[,p+2]==1],meantest[testdata[,p+2]==1,1],pch="*",col=2)
  points(xtestavg[testdata[,p+2]==0],meantest[testdata[,p+2]==0,2],pch="*",col=3)
  y_hat_true=(1-pxtest)*(meantest[,1])+(pxtest)*(meantest[,2])
  points(xtestavg,y_hat_true,pch='x',col='grey')
  
  plot(xtestavg, pxtest, col=1)
  points(xtestavg,1-pxtest,col=2)
  
  #xtestsort=sort(xtestavg,index.return=T)
  #image(xtestsort$x,ygrid,truedensity[xtestsort$ix,])
  #points(xtestavg,ytest,pch="*")
  #apply(truedensity,1,sum)*gridsize
  
  ##################################################
  ### Penalized regression
  ##################################################

  # fit model: Lasso
  
  auxind=p==plist
  opt1=optL1(response=y, penalized = x[,1:p])
  pen.modl1=penalized(response=y, penalized = x[,1:p],lambda1 = opt1$lambda)
  show(pen.modl1)
  coefficients(pen.modl1)
  coefficients(pen.modl1, "penalized")
    
  # predict test data
  pred.l1=predict(pen.modl1,penalized=matrix(xtest[,1:p],Ntest,p))
  plot(xtestavg,ytest,pch="*")
  points(xtestavg,y_hat_true,pch="*",col="grey")
  points(xtestavg,pred.l1[,1],pch="*",col=2)
    
  MAE.l1[auxind]=sum(abs(ytest-pred.l1[,1]))/Ntest
  MSE.l1[auxind]=sqrt(sum((ytest-pred.l1[,1])^2)/Ntest)
    
  # compute coverage
  pred.l1=cbind(pred.l1,pred.l1[,1]-qnorm(.975)*sqrt(pred.l1[,2]), pred.l1[,1]+qnorm(.975)*sqrt(pred.l1[,2]))
  EC.l1[auxind]=sum(ytest>pred.l1[,3]&ytest<pred.l1[,4])/Ntest
    
  # size of intervals
  cilength.l1=pred.l1[,4]-pred.l1[,3]
  meancilength.l1[auxind]=mean(cilength.l1)
  mincilength.l1[auxind]=min(cilength.l1)
  maxcilength.l1[auxind]=max(cilength.l1)
    
  # compute l1 distance
  # y-grid
  density.l1=matrix(0,Ntest, ngrid)
  for (ind in 1:ngrid){
    density.l1[,ind]=dnorm(ygrid[ind],pred.l1[,1],sqrt(pred.l1[,2]))
  }
  l1error.l1[auxind]=mean(apply(abs(density.l1-truedensity),1,sum)*gridsize)
  #image(xtestsort$x,ygrid,density.l1[xtestsort$ix,])
  # points(xtestavg,ytest,pch="*")
  #apply(density.l1,1,sum)*gridsize

  ##################################################
  ### Random Forests
  ##################################################
  
  rf.mod=randomForest(x=as.matrix(x[,1:p],N,p), y=as.vector(y))
  
  # predict test data
  pred.rf=predict(rf.mod,newdata=as.matrix(xtest[,1:p],Ntest,p), type="response")
  plot(xtestavg,ytest,pch="*")
  points(xtestavg,y_hat_true,pch="*",col="grey")
  points(xtestavg,pred.rf,pch="*",col=2)
  
  MAE.rf[auxind]=sum(abs(ytest-pred.rf))/Ntest
  MSE.rf[auxind]=sqrt(sum((ytest-pred.rf)^2)/Ntest)
  
  ##################################################
  ### Treed GP
  ##################################################
  
  btgp.mod <- btgp(X=as.matrix(x[,1:p],N,p), XX=as.matrix(xtest[,1:p],Ntest,p), Z=y,BTE = c(2000, 10000, 10),corr="exp", meanfn="constant")
  
  plot(xtestavg,ytest,pch="*")
  points(xtestavg,y_hat_true,pch="*",col="grey")
  points(xtestavg,btgp.mod$ZZ.mean,pch="*",col=2)
  
  MAE.tgp[auxind]=sum(abs(ytest-btgp.mod$ZZ.mean))/Ntest
  MSE.tgp[auxind]=sqrt(sum((ytest-btgp.mod$ZZ.mean)^2)/Ntest)
  
  # compute coverage
  EC.tgp[auxind]=sum(ytest>btgp.mod$ZZ.q1&ytest<btgp.mod$ZZ.q2)/Ntest
  
  # size of intervals
  meancilength.tgp[auxind]=mean(btgp.mod$ZZ.q)
  mincilength.tgp[auxind]=min(btgp.mod$ZZ.q)
  maxcilength.tgp[auxind]=max(btgp.mod$ZZ.q)
  
  # compute l1 distance
  # y-grid
  density.tgp=matrix(0,Ntest, ngrid)
  for (ind in 1:ngrid){
    density.tgp[,ind]=dnorm(ygrid[ind],btgp.mod$ZZ.mean,sqrt(btgp.mod$ZZ.s2))
  }
  l1error.tgp[auxind]=mean(apply(abs(density.tgp-truedensity),1,sum)*gridsize)
  #plot(ygrid, density.tgp[1,],type='l')
  #lines(ygrid, truedensity[1,],col=2)
  #image(xtestsort$x,ygrid,density.tgp[xtestsort$ix,])
  # points(xtestavg,ytest,pch="*")
  #apply(density.tgp,1,sum)*gridsize
  
  ##################################################
  ### Single GP
  ##################################################
  
  bgp.mod <- bgp(X=as.matrix(x[,1:p],N,p), XX=as.matrix(xtest[,1:p],Ntest,p), Z=y,BTE = c(2000, 10000, 10),corr="exp", meanfn="constant")
  
  plot(xtestavg,ytest,pch="*")
  points(xtestavg,y_hat_true,pch="*",col="grey")
  points(xtestavg,bgp.mod$ZZ.mean,pch="*",col=2)
  
  MAE.bgp[auxind]=sum(abs(ytest-bgp.mod$ZZ.mean))/Ntest
  MSE.bgp[auxind]=sqrt(sum((ytest-bgp.mod$ZZ.mean)^2)/Ntest)
  
  # compute coverage
  EC.bgp[auxind]=sum(ytest>bgp.mod$ZZ.q1&ytest<bgp.mod$ZZ.q2)/Ntest
  
  # size of intervals
  meancilength.bgp[auxind]=mean(bgp.mod$ZZ.q)
  mincilength.bgp[auxind]=min(bgp.mod$ZZ.q)
  maxcilength.bgp[auxind]=max(bgp.mod$ZZ.q)
  
  # compute l1 distance
  # y-grid
  density.gp=matrix(0,Ntest, ngrid)
  for (ind in 1:ngrid){
    density.gp[,ind]=dnorm(ygrid[ind],bgp.mod$ZZ.mean,sqrt(bgp.mod$ZZ.s2))
  }
  l1error.bgp[auxind]=mean(apply(abs(density.gp-truedensity),1,sum)*gridsize)
  #plot(ygrid, density.gp[1,],type='l')
  #lines(ygrid, truedensity[1,],col=2)
  #image(xtestsort$x,ygrid,density.gp[xtestsort$ix,])
  # points(xtestavg,ytest,pch="*")
  #apply(density.bgp,1,sum)*gridsize
  
  ##################################################
  ### MoE
  ##################################################
  # 
  # yxdata=data.frame(y,x)
  # xnam=paste0('x', 1:p)
  # names(yxdata)=c('y',xnam)
  # (fmla <- as.formula(paste("y ~ ", paste(xnam, collapse= "+"))))
  # moe.mod=MoE_clust(yxdata, G=10:20, gating=  fmla, expert= fmla, verbose=FALSE,nstarts=50,modelnames='VII',init.z="random",exp.init=list(mahalanobis=F),tau=0.01,criterion="bic")
  # NOT WORKING!
  
  # ##################################################
  # ### Conditional Mixture of neural networks
  # ##################################################
  # nloglike=Inf
  # thetaop=0
  # mop=0
  # hop=0
  # for (h in c(2:5)){
  #   for (m in c(2:10)){
  #     thetafit <-condgaussmixt.train(h,m,t(x),y, nstart = 10)
  #     newnloglike=condgaussmixt.nll(thetafit,h,m,t(x),y)[1]
  #     if(newnloglike<nloglike){
  #       nloglike=newnloglike
  #       thetaop=thetafit
  #       mop=m
  #       hop=h
  #     }
  #   }
  # }
  # #thetainit <- condgaussmixt.init(p,h,m,ytrain)
  # #thetafit <- condgaussmixt.fit(thetainit,h,m,t(xtrain),ytrain)
  # thetaop <- condgaussmixt.fit(thetaop,hop,mop,t(x),y)
  # 
  #Need larger grid!!
  # gridsize=0.01
  # ygrid=seq(-40,40,gridsize)
  # ngrid=length(ygrid)
  # truedensity=matrix(0,Ntest, ngrid)
  # truemean=matrix(0,Ntest,1)
  # tau=1.2
  # mu=c(2,6)
  # xtestavg=apply(xtest,1,mean)
  # pxtest=exp(-tau^2/2*(xtestavg-mu[1])^2)/(exp(-tau^2/2*(xtestavg-mu[1])^2)+exp(-tau^2/2*(xtestavg-mu[2])^2))
  # beta1=c(0.1,0.6)
  # beta2=c(-0.1,0.4)
  # meantest=matrix(0,Ntest,2)
  # meantest[,1]=exp(beta1[1]*xtestavg)*cos(beta1[2]*pi*xtestavg)
  # meantest[,2]=exp(beta2[1]*xtestavg)*cos(beta2[2]*pi*xtestavg)
  # sigma1=0.15
  # sigma2=0.05
  # for (ind in 1:ngrid){
  #   truedensity[,ind]=(1-pxtest)*dnorm(ygrid[ind],meantest[,1],sigma1)+(pxtest)*dnorm(ygrid[ind],meantest[,2],sigma2)
  # }
  # plot(xtestavg,ytest,pch="*")
  # points(xtestavg[testdata[,p+2]==1],meantest[testdata[,p+2]==1,1],pch="*",col=2)
  # points(xtestavg[testdata[,p+2]==0],meantest[testdata[,p+2]==0,2],pch="*",col=3)
  # y_hat_true=(1-pxtest)*(meantest[,1])+(pxtest)*(meantest[,2])
  # points(xtestavg,y_hat_true,pch='x',col='grey')
  # 
  # plot(xtestavg, pxtest, col=1)
  # points(xtestavg,1-pxtest,col=2)
  # 
  # params.mixt <- condgaussmixt.fwd(thetaop,hop,mop,t(xtest))
  # cnn.dens=matrix(0,Ntest,ngrid)
  # for (j in 1:ngrid){
  #    cnn.dens[,j]=dcondgaussmixt(params.mixt,mop,rep(ygrid[j],Ntest), trunc=TRUE)
  #  }
  # #cnn.dens[is.nan(cnn.dens)]=0
  # l1error.cnn[auxind]=mean(apply(abs(cnn.dens-truedensity),1,sum)*gridsize)
  # cnn.median=condgaussmixt.quant(thetaop,hop,mop,t(xtest),0.5,-3,3,trunc=TRUE)
  # cnn.mean=apply(cnn.dens*t(matrix(ygrid,ngrid,Ntest)),1,sum)*gridsize
  # plot(xtestavg,ytest,pch="*",col="grey",ylim=c(-4,4))
  # points(xtestavg,cnn.mean,pch='x',col=1)
  # points(xtestavg,cnn.median,pch='x',col=2)
  # plot(ygrid,truedensity[6,],type="l",col=1)
  # lines(ygrid,cnn.dens[6,],col=2)
  # #image(xtestsort$x,ygrid,cnn.dens[xtestsort$ix,])
  # #apply(cnn.dens,1,sum)*gridsize
}

##########################################

 meancilength.edp=c(0.5518051730023771, 0.7839028647486943, 0.7179570473556918, 0.8450194108368718,  0.8377833115932694, 0.9468311073184711, 0.9129492781775133, 0.9414651517910688, 0.9112264258633832, 1.2784446654712196, 1.0161213113986995, 1.0527393764179798, 0.9224425690390451, 0.9557534542271835, 1.0085996600481832, 1.0716298980518004, 0.9045036619208816, 1.0131381676095832, 1.0738822622285036, 1.1896266579781072, 1.1765994263427442, 1.0859016484017436, 1.366133295876257)
 meancilength.dp=c(0.6892373531340621, 1.155177407938208, 1.004697592962842, 0.988490691154232, 0.9874815878119069, 1.1474480381486283, 1.1303321079623587, 1.1414071024982924,  1.0742774909397619, 1.325496937719263, 1.1659392373453539, 1.2848257383314248, 1.2777271173231102, 1.1124233543992545, 1.173718496377338, 1.196122171678709, 1.0654942926311632, 1.193001882887554, 1.1134866330806017, 1.2406684352300836, 1.2042257532479024, 1.0879082896852152, 1.3137421551215918)
 EC.edp=c(0.9716, 0.9408, 0.8804, 0.9368, 0.936, 0.9636, 0.954, 0.9644, 0.9504, 0.9856, 0.9636, 0.9676, 0.9692, 0.9684, 0.9592, 0.9708, 0.9488, 0.9596, 0.97, 0.9736, 0.9704, 0.9656, 0.9708)
 EC.dp=c(0.9736, 0.9732, 0.9512, 0.9556, 0.9452, 0.9692, 0.9752, 0.9576, 0.9592, 0.976, 0.9668, 0.9816, 0.9756, 0.9712, 0.9712, 0.9784, 0.9596, 0.9768, 0.9616, 0.9672, 0.9752, 0.9584, 0.978)
 l1error.dp=c(0.2778644883189885229, 0.8114141385895649039, 0.7965898013019411250, 0.9144361306397439382, .9339101379334611153, .9427315527156543418, .9243647597765439761, 1.013843071030508858, 0.9732653692689633429, 1.018577869385822288, 0.9770547624763760153, 0.9942224066167837382, 1.008345345117334757, 0.9633689865076283665, 0.9944639949952567282, 0.9706742041237809149, 0.9573096059339781805, 0.9912540585555621453, 0.9733228880873832090, 0.9949255096126810027, 1.011340344541501679, 0.9561460658195591877, 1.048380676346099438, 1.019189200615140845)
l1error.edp=c(.2116799111910779296, .6430292618563266949, .7838374757919847058, .7326971047252314184, .8294885548352257665, .8426679654023020438, .8182640296170814453, .8722419268364260958, .8484166535949115850, .9575901450953518967, .8976834725528326508, .8984519350882814726, .8069547029705254060, .8625826069628165227, .8983512382042285749, .9005671659603953216, .8370275305306497104, .8749987406485879582, .9014307322173212844, .9565124350939224751, .9714544970209253449, 0.9238911363301419710, 1.075153261533327020, 1.008280877186718882)
# epoch.edp=c(16.3551,15.7773,22.9451, 30.8157, 38.3631, 46.2794)
# epoch.dp=c(14.4199,15.3368,39.1138, 47.2233, 51.7646,58.562)

#Plot L1
par(mfrow=c(1,1),mar=c(2.2,2.2,.5,.5)+.1, mgp=c(1.2, .1, 0))
plot(plist,l1error.l1,type='b',col=3, ylim=c(-0.2,2),xlab='D',ylab="L1 error",cex=1.5, font=2,font.lab=2,cex.lab=1.4,lwd=2)
lines(plist,l1error.edp,type='b',col=2,cex=1.5,lwd=2)
lines(plist,l1error.dp,type='b',col=1,cex=1.5,lwd=2)
lines(plist,l1error.bgp,type='b',col=4,cex=1.5,lwd=2)
lines(plist,l1error.tgp,type='b',col=5,cex=1.5,lwd=2)
lines(plist,l1error.lin,type='b',col=6,cex=1.5,lwd=2)
legend(0.2,0.2,c("DP","EDP","EDPlin","Lasso","GP","TGP"), col=c(1,2,6,3,4,5), lty=c(1,1),lwd=c(2,2),ncol=3,cex=1.4,text.font=2,x.intersp=0.25,y.intersp=0.8,seg.len = 1)

#Plot L1 - no lasso
par(mfrow=c(1,1),mar=c(2.2,2.2,.5,.5)+.1, mgp=c(1.2, .1, 0))
plot(plist,l1error.l1,type='b',col="white", ylim=c(0.1,1.6),xlab='D',ylab="L1 error",cex=1.5, font=2,font.lab=2,cex.lab=1.4,lwd=2)
lines(plist,l1error.edp,type='b',col=2,cex=1.5,lwd=2)
lines(plist,l1error.dp,type='b',col=1,cex=1.5,lwd=2)
lines(plist,l1error.bgp,type='b',col=4,cex=1.5,lwd=2)
lines(plist,l1error.tgp,type='b',col=5,cex=1.5,lwd=2)
lines(plist,l1error.lin,type='b',col=6,cex=1.5,lwd=2)
legend(10,0.4,c("DP","EDP","EDPlin","GP","TGP"), col=c(1,2,6,4,5), lty=c(1,1),lwd=c(2,2),ncol=3,cex=1.4,text.font=2,x.intersp=0.25,y.intersp=0.8,seg.len = 1)



#Plot credible interval length
par(mfrow=c(1,1),mar=c(2.2,2.2,.5,.5)+.1, mgp=c(1.2, .1, 0))
plot(plist,meancilength.l1,type='b',col=3, ylim=c(-0.5,4),xlab='D',ylab="Mean CI length",cex=1.5, font=2,font.lab=2,lwd=2,cex.lab=1.4)
lines(plist[plist!=200],meancilength.edp[plist!=200],type='b',col=2,cex=1.5,lwd=2)
lines(plist[plist!=200],meancilength.dp[plist!=200],type='b',col=1,cex=1.5,lwd=2)
lines(plist,meancilength.bgp,type='b',col=4,cex=1.5,lwd=2)
lines(plist,meancilength.tgp,type='b',col=5,cex=1.5,lwd=2)
lines(plist,meancilength.lin,type='b',col=6,cex=1.5,lwd=2)
legend(0.2,0.3,c("DP","EDP","EDPlin","Lasso","GP","TGP"), col=c(1,2,6,3,4,5), lty=c(1,1),lwd=c(2,2),ncol=3,cex=1.4,text.font=2,x.intersp=0.25,y.intersp=0.8,seg.len = 1)

#Plot credible interval length - no lasso
par(mfrow=c(1,1),mar=c(2.2,2.2,.5,.5)+.1, mgp=c(1.2, .1, 0))
plot(plist[plist!=200],meancilength.l1[plist!=200],type='b',col="white", ylim=c(0.1,2.7),xlab='D',ylab="Mean CI length",cex=1.5, font=2,font.lab=2,lwd=2,cex.lab=1.4)
lines(plist[plist!=200],meancilength.edp[plist!=200],type='b',col=2,cex=1.5,lwd=2)
lines(plist[plist!=200],meancilength.dp[plist!=200],type='b',col=1,cex=1.5,lwd=2)
lines(plist[plist!=200],meancilength.bgp[plist!=200],type='b',col=4,cex=1.5,lwd=2)
lines(plist[plist!=200],meancilength.tgp[plist!=200],type='b',col=5,cex=1.5,lwd=2)
lines(plist[plist!=200],meancilength.lin[plist!=200],type='b',col=6,cex=1.5,lwd=2)
legend(10,0.6,c("DP","EDP","EDPlin","GP","TGP"), col=c(1,2,6,4,5), lty=c(1,1),lwd=c(2,2),ncol=3,cex=1.4,text.font=2,x.intersp=0.25,y.intersp=0.8,seg.len = 1)

#Plot credible interval length - only edp/dp
par(mfrow=c(1,1),mar=c(2.2,2.2,.5,.5)+.1, mgp=c(1.2, .1, 0))
plot(plist[plist!=200],meancilength.l1[plist!=200],type='b',col="white", ylim=c(0.45,1.5),xlab='D',ylab="Mean CI length",cex=1.5, font=2,font.lab=2,lwd=2,cex.lab=1.4)
lines(plist[plist!=200],meancilength.edp[plist!=200],type='b',col=2,cex=1.5,lwd=2)
lines(plist[plist!=200],meancilength.dp[plist!=200],type='b',col=1,cex=1.5,lwd=2)
#lines(plist,meancilength.bgp,type='b',col=4,cex=1.5,lwd=2)
#lines(plist,meancilength.tgp,type='b',col=5,cex=1.5,lwd=2)
#lines(plist,meancilength.lin,type='b',col=6,cex=1.5,lwd=2)
legend(10,0.6,c("DP","EDP"), col=c(1,2,6,4,5), lty=c(1,1),lwd=c(2,2),ncol=3,cex=1.4,text.font=2,x.intersp=0.25,y.intersp=0.8,seg.len = 1)


#Plot credible interval length
par(mfrow=c(1,1),mar=c(2,2,.5,.5)+.1, mgp=c(1, .25, 0))
plot(plist,mincilength.l1,type='b',col=3, ylim=c(-0.5,4),xlab='D',ylab="Min CI length",cex=1.5, font=2,font.lab=2,lwd=2,cex.lab=1.2)
lines(plist,EDPCI,type='b',col=2,cex=1.5,lwd=2)
#lines(plist,DPCI,type='b',col=1,cex=1.5,lwd=2)
lines(plist,mincilength.bgp,type='b',col=4,cex=1.5,lwd=2)
lines(plist,mincilength.tgp,type='b',col=5,cex=1.5,lwd=2)
lines(plist,mincilength.lin,type='b',col=6,cex=1.5,lwd=2)
legend(0,0,c("DP","EDP","EDPlin", "Lasso","GP","TGP"), col=c(1,2,6,3,4,5), lty=c(1,1),lwd=c(2,2),horiz=T,cex=1,text.font=2)

#Plot credible interval length
par(mfrow=c(1,1),mar=c(2,2,.5,.5)+.1, mgp=c(1, .25, 0))
plot(plist,maxcilength.l1,type='b',col=3, ylim=c(0,10),xlab='D',ylab="Max CI length",cex=1.5, font=2,font.lab=2,lwd=2)
#lines(plist,EDPCI,type='b',col=2,cex=1.5,lwd=2)
#lines(plist,DPCI,type='b',col=1,cex=1.5,lwd=2)
lines(plist,maxcilength.bgp,type='b',col=4,cex=1.5,lwd=2)
lines(plist,maxcilength.tgp,type='b',col=5,cex=1.5,lwd=2)
lines(plist,maxcilength.lin,type='b',col=6,cex=1.5,lwd=2)
legend(1,0.4,c("DP","EDP","EDPlin","Lasso","GP","TGP"), col=c(1,2,6,3,4,5), lty=c(1,1),lwd=c(2,2),horiz=T,cex=0.65,text.font=2)

#Plot coverage
par(mfrow=c(1,1),mar=c(2.2,2.2,.5,.5)+.1, mgp=c(1.2, .1, 0))
plot(plist,EC.l1,type='b',col=3, ylim=c(0.6,1),xlab='D',ylab="Coverage",cex=1.5, font=2,font.lab=2,lwd=2,cex.lab=1.4)
lines(plist[p!=plist],EC.edp[p!=plist],type='b',col=2,cex=1.5,lwd=2)
lines(plist[p!=plist],EC.dp[p!=plist],type='b',col=1,cex=1.5,lwd=2)
lines(plist,EC.bgp,type='b',col=4,cex=1.5,lwd=2)
lines(plist,EC.tgp,type='b',col=5,cex=1.5,lwd=2)
lines(plist,EC.lin,type='b',col=6,cex=1.5,lwd=2)
lines(plist,rep(0.95,length(plist)),col="grey",lwd=2)
legend(0.2,0.68,c("DP","EDP","EDPlin","Lasso","GP","TGP"), col=c(1,2,6,3,4,5), lty=c(1,1),lwd=c(2,2),ncol=3,cex=1.4,text.font=2,x.intersp=0.25,y.intersp=0.8,seg.len = 1)

#Plot coverage - no lasso
par(mfrow=c(1,1),mar=c(2.2,2.2,.5,.5)+.1, mgp=c(1.2, .1, 0))
plot(plist[plist!=200],EC.l1[plist!=200],type='b',col="white", ylim=c(0.6,1),xlab='D',ylab="Coverage",cex=1.5, font=2,font.lab=2,lwd=2,cex.lab=1.4)
lines(plist[plist!=200],EC.edp[p!=plist],type='b',col=2,cex=1.5,lwd=2)
lines(plist[plist!=200],EC.dp[p!=plist],type='b',col=1,cex=1.5,lwd=2)
lines(plist[plist!=200],EC.bgp[plist!=200],type='b',col=4,cex=1.5,lwd=2)
lines(plist[plist!=200],EC.tgp[plist!=200],type='b',col=5,cex=1.5,lwd=2)
lines(plist[plist!=200],EC.lin[plist!=200],type='b',col=6,cex=1.5,lwd=2)
lines(plist,rep(0.95,length(plist)),col="grey",lwd=2)
legend(0.2,0.68,c("DP","EDP","EDPlin","GP","TGP"), col=c(1,2,6,4,5), lty=c(1,1),lwd=c(2,2),ncol=3,cex=1.4,text.font=2,x.intersp=0.25,y.intersp=0.8,seg.len = 1)

#Plot coverage - only dp and edp
par(mfrow=c(1,1),mar=c(2.2,2.2,.5,.5)+.1, mgp=c(1.2, .1, 0))
plot(plist[p!=plist],EC.l1[p!=plist],type='b',col="white", ylim=c(0.85,1),xlab='D',ylab="Coverage",cex=1.5, font=2,font.lab=2,lwd=2,cex.lab=1.4)
lines(plist[p!=plist],EC.edp[p!=plist],type='b',col=2,cex=1.5,lwd=2)
lines(plist[p!=plist],EC.dp[p!=plist],type='b',col=1,cex=1.5,lwd=2)
# lines(plist,EC.bgp,type='b',col=4,cex=1.5,lwd=2)
# lines(plist,EC.tgp,type='b',col=5,cex=1.5,lwd=2)
# lines(plist,EC.lin,type='b',col=6,cex=1.5,lwd=2)
lines(plist,rep(0.95,length(plist)),col="grey",lwd=2)
legend(0.2,0.87,c("DP","EDP"), col=c(1,2), lty=c(1,1),lwd=c(2,2),ncol=3,cex=1.4,text.font=2,x.intersp=0.25,y.intersp=0.8,seg.len = 1)

#Plot MAE
par(mfrow=c(1,1),mar=c(2,2,.5,.5)+.1, mgp=c(1, .25, 0))
plot(plist,MAE.l1,type='b',col=3, ylim=c(-0.14,0.8),xlab='D',ylab="MAE",cex=1.5, font=2,font.lab=2,lwd=2)
#lines(plist,EDPCI,type='b',col=2,cex=1.5,lwd=2)
#lines(plist,DPCI,type='b',col=1,cex=1.5,lwd=2)
lines(plist,MAE.bgp,type='b',col=4,cex=1.5,lwd=2)
lines(plist,MAE.tgp,type='b',col=5,cex=1.5,lwd=2)
lines(plist,MAE.lin,type='b',col=6,cex=1.5,lwd=2)
lines(plist,MAE.rf,type='b',col=7,cex=1.5,lwd=2)
legend(1,0.05,c("DP","EDP","EDPlin","Lasso","GP","TGP","RF"), col=c(1,2,6,3,4,5,7), lty=c(1,1),lwd=c(2,2),ncol=4,cex=1.4,text.font=2,x.intersp=0.25,y.intersp=0.8,seg.len = 1)

#Plot MSE
par(mfrow=c(1,1),mar=c(2,2,.5,.5)+.1, mgp=c(1, .25, 0))
plot(plist,MSE.l1,type='b',col=3, ylim=c(-0.14,1.1),xlab='D',ylab="MSE",cex=1.5, font=2,font.lab=2,lwd=2)
#lines(plist,EDPCI,type='b',col=2,cex=1.5,lwd=2)
#lines(plist,DPCI,type='b',col=1,cex=1.5,lwd=2)
lines(plist,MSE.bgp,type='b',col=4,cex=1.5,lwd=2)
lines(plist,MSE.tgp,type='b',col=5,cex=1.5,lwd=2)
lines(plist,MSE.lin,type='b',col=6,cex=1.5,lwd=2)
lines(plist,MSE.rf,type='b',col=7,cex=1.5,lwd=2)
legend(0,0.1,c("DP","EDP","EDPlin","Lasso","GP","TGP","RF"), col=c(1,2,6,3,4,5,7), lty=c(1,1),lwd=c(2,2),ncol=4,cex=1.4,text.font=2,x.intersp=0.25,y.intersp=0.8,seg.len = 1)


write.csv(data.frame(p=plist,l1error.l1,l1error.bgp,l1error.tgp,l1error.lin),file="l1error.csv")
write.csv(data.frame(p=plist,meancilength.l1,meancilength.bgp,meancilength.tgp,meancilength.lin),file="meancilength.csv")
write.csv(data.frame(p=plist,EC.l1,EC.bgp,EC.tgp,EC.lin),file="EC.csv")
write.csv(data.frame(p=plist,MSE.l1,MSE.bgp,MSE.tgp,MSE.lin),file="MSE.csv")
write.csv(data.frame(p=plist,DP=kVI[,1],EDP=kVI[,2], EDPx= kxtot,EDPlin=kVI_lin),file="clusters.csv")
write.csv(data.frame(p=plist,DPVI=VI_dist[,1], DPcb=cb_size[,1],EDPVI=VI_dist[,2],EDPcb=cb_size[,2],EDPlinVI=VI_dist_lin, EDPlincb=cb_size_lin),file="VIdist.csv")

