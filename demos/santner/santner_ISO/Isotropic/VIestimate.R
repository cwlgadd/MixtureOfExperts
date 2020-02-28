##############
## Find VI estimate of partition 
#############

library(mcclust.ext)



plist=c(1,seq(5,100,5),125,150,200)
for (P in plist){
  # Load matrix
  csvname=paste('./EmE/EmE',P,'_initS_1_linexam/feature.csv',sep="")
  config=read.csv(csvname,header=F)
  print(paste('For P=',P,', Iterations completed=',dim(config)[1],sep=''))
}

plist=c(1,seq(5,100,5),125,150,200)
VI_dist=matrix(0,length(plist),2)
VI_postexp=matrix(0,length(plist),2)
cb_size=matrix(0,length(plist),2)
kVI=matrix(0,length(plist),2)
kxVI=list()

for (P in plist){
  # Load matrix
  csvname=paste('./EmE/EmE',P,'_initS_1_linexam/feature.csv',sep="")
   config=read.csv(csvname,header=F)
   print(paste('For P=',P,', Iterations completed=',dim(config)[1],sep=''))
   dataname=paste('./iso_santner',P,'_mixture_train.csv',sep="")
   data=read.csv(dataname,header=F)
   model_ind=2 # 1 for JmE and 2 for EmE
   config=as.matrix(config)
   Y=data[,P+1]
   X=data[,1:P]
   Y=as.matrix(Y)
   X=as.matrix(X)
   S=dim(config)[1]
   N=dim(config)[2]
   #Remove burnin
   #burnin=min(1000,floor(S/2))
   burnin=max(1000,floor(S/5))
   if (S<2000){burnin=floor(S/2)}
   #maxs=min(5000,S)
   maxs=S
  config=config[(burnin+1):maxs,]
   S=dim(config)[1]
   
   #relabel clusters
   for (s in 1:S){
     config_s=config[s,]
     labels_s=unique(config_s)
     for (j in 1:length(labels_s)){
       config[s,config_s==labels_s[j]]=j
     }
   }
   
   ## Heat map of posterior similarity matrix
   psm=comp.psm(config)
   #plotname=paste('./EmE/EmE',P,'_initS_1_linexam/psm.pdf',sep="")
   plotname=paste('./EmE/EmE',P,'_initS_1_linexam/psm_v2.pdf',sep="")
   pdf(plotname)
   par(mfrow=c(1,1),mar=c(1.5,1.6,.5,.5)+.1, mgp=c(.5, .5, 0))
   plotpsm(psm, xlab="Permuted observation index",ylab="Permuted observation index",xaxt="n",yaxt="n",cex.lab=1.6,font.lab=2)
   dev.off()
   
   ## VI
   output_vi=minVI(psm,config,method=("all"),include.greedy=TRUE,suppress.comment = F)
   config_vi=output_vi$cl[1,]
   print(paste('k_VI=',max(config_vi),sep=''))
   config_evi=output_vi$value[1]
   
   # Poster expected VI
   VI_postexp[P==plist,model_ind]=VI(data[,P+2]+1,config)
   
   # VI 
   VI_dist[P==plist,model_ind]=VI(data[,P+2]+1,matrix(config_vi,1,N))
   
   # number of clusters in VI estimate
   k_hat=max(config_vi)
   kVI[P==plist,model_ind]=k_hat
   
   ## Plot clustering estimate
   #plotname=paste('./EmE/EmE',P,'_initS_1_linexam/VIestimate.pdf',sep="")
   plotname=paste('./EmE/EmE',P,'_initS_1_linexam/VIestimate_v2.pdf',sep="")
   pdf(plotname)
   Xmean=apply(X,1,mean)
   cl=c("black", "red", "green", "blue", "yellow", "orange", "magenta", "cyan", "gray","darkgray","blueviolet","darkmagenta","darkred", "darkorange", "darkorange4", "cornflowerblue", "darkolivegreen2", "darkolivegreen4", "brown","chartreuse","burlywood1", "violet","violetred","yellow4","tomato3","springgreen", "slateblue","tan","seashell","thistle" ) 
   par(mfrow=c(1,1),mar=c(2.5,2.5,1.5,.5)+.1, mgp=c(1.5, .5, 0))
   plot(Xmean,Y,xlab="x",ylab="y",ylim=c(min(Y)-.1,max(Y)+.1),cex.lab=1.6,font=2,font.lab=2,lwd=2)
   for(j in 1:k_hat){
     points(Xmean[config_vi==j], Y[config_vi==j],col=cl[j],pch=16,cex=1.5)
   }
   dev.off()
   
   ## Credible ball
   cb=credibleball(config_vi, config)
   summary(cb)
   cb_size[P==plist,model_ind]=cb$dist.horiz
   print(paste('VI_dist=',VI_dist[P==plist,model_ind],' CB size=',cb_size[P==plist,model_ind]))
   
   ## For Enriched model ONLY
   config_vi_x=rep(0,N)
   csvname=paste('./EmE/EmE',P,'_initS_1_linexam/covariate.csv',sep="")
   configx=read.csv(csvname,header=F)
   configx=as.matrix(configx)
   #Remove burnin
   configx=configx[(burnin+1):maxs,]
   pind=which(P==plist)
   kxVI[[pind]]=rep(0,k_hat)
   for (j in 1:k_hat){
     if (sum(config_vi==j)>1){
     configj=matrix(config[,config_vi==j],S,sum(config_vi==j))
     configxj=matrix(configx[,config_vi==j],S,sum(config_vi==j))
     #relabel clusters
     for (s in 1:S){
       config_s=configj[s,]
       labels_s=unique(config_s)
       aux=0
       for (h in 1:length(labels_s)){
         configx_s=configxj[s,config_s==labels_s[h]]
         labelsx_s=unique(configx_s)
         for (l in 1:length(labelsx_s)){
           configxj[s,config_s==labels_s[h]][configx_s==labelsx_s[l]]=aux+l
         }
         aux=aux+length(labelsx_s)
       }
     }
     ## Heat map of posterior similarity matrix
     psmj=comp.psm(configxj)
     #par(mfrow=c(1,1),mar=c(1.5,1.5,.5,.5)+.1, mgp=c(.5, .5, 0))
     #plotpsm(psmj, xlab="Permuted observation index",ylab="Permuted observation index",xaxt="n",yaxt="n",cex.lab=1.2,font.lab=2)
     #plotpsm(psmj, xlab="",ylab="")
     ## VI
     outputj_vi=minVI(psmj,configxj,method=("all"),include.greedy=TRUE,suppress.comment = F)
     configj_vi=outputj_vi$cl[1,]
     kxVI[[pind]][j]=max(configj_vi)
     
     #config_vi_x[config_vi==j]=configj_vi
     # for (h in 1:max(configj_vi)) {
     #   print(apply(matrix(Xall[config_vi==j&config_vi_x==h,1:P],sum(config_vi==j&config_vi_x==h),P),2,mean))
     # }
     # Xj=data.frame(x=Xall[config_vi==j,1:P])
     # plot(outputj_vi,data=Xj)
     }
     else{
       kxVI[[pind]][j]=1
     }
   }
   print(paste('Number of x-clusters in each y-cluster=',kxVI[[pind]]))
   
   #relabel clusters
   for (s in 1:S){
     config_s=config[s,]
     labels_s=unique(config_s)
     aux=0
     for (h in 1:length(labels_s)){
       configx_s=configx[s,config_s==labels_s[h]]
       labelsx_s=unique(configx_s)
       for (l in 1:length(labelsx_s)){
         configx[s,config_s==labels_s[h]][configx_s==labelsx_s[l]]=aux+l
       }
       aux=aux+length(labelsx_s)
     }
   }
   #plotname=paste('./EmE/EmE',P,'_initS_1_linexam/psm_x.pdf',sep="")
   plotname=paste('./EmE/EmE',P,'_initS_1_linexam/psm_x_v2.pdf',sep="")
   pdf(plotname)
   psmx=comp.psm(configx)
   plotpsm(psmx, xlab="",ylab="")
   dev.off()
}
##################################

## for DP model

for (P in plist){
  # Load matrix
  csvname=paste('./JmE/JmE',P,'/feature.csv',sep="")
  config=read.csv(csvname,header=F)
  print(paste('For P=',P,', Iterations completed=',dim(config)[1],sep=''))
}

for (P in plist){
  # Load matrix
  csvname=paste('./JmE/JmE',P,'/feature.csv',sep="")
  config=read.csv(csvname,header=F)
  print(paste('For P=',P,', Iterations completed=',dim(config)[1],sep=''))
  dataname=paste('./iso_santner',P,'_mixture_train.csv',sep="")
  data=read.csv(dataname,header=F)
  model_ind=1 # 1 for JmE and 2 for EmE
  config=as.matrix(config)
  Y=data[,P+1]
  X=data[,1:P]
  Y=as.matrix(Y)
  X=as.matrix(X)
  S=dim(config)[1]
  N=dim(config)[2]
  #Remove burnin
  #burnin=min(1000,floor(S/2))
  burnin=max(1000,floor(S/5))
  if (S<2000){burnin=floor(S/2)}
  #maxs=min(5000,S)
  maxs=S
  config=config[(burnin+1):maxs,]
  S=dim(config)[1]
  
  #relabel clusters
  for (s in 1:S){
    config_s=config[s,]
    labels_s=unique(config_s)
    for (j in 1:length(labels_s)){
      config[s,config_s==labels_s[j]]=j
    }
  }
  
  ## Heat map of posterior similarity matrix
  psm=comp.psm(config)
  #plotname=paste('./EmE/EmE',P,'_initS_1_linexam/psm.pdf',sep="")
  plotname=paste('./JmE/JmE',P,'/psm_v2.pdf',sep="")
  pdf(plotname)
  par(mfrow=c(1,1),mar=c(1.5,1.5,.5,.5)+.1, mgp=c(.5, .5, 0))
  plotpsm(psm, xlab="Permuted observation index",ylab="Permuted observation index",xaxt="n",yaxt="n",cex.lab=1.4,font.lab=2)
  dev.off()
  
  ## VI
  output_vi=minVI(psm,config,method=("all"),include.greedy=TRUE,suppress.comment = F)
  config_vi=output_vi$cl[1,]
  print(paste('k_VI=',max(config_vi),sep=''))
  config_evi=output_vi$value[1]
  
  # Poster expected VI
  VI_postexp[P==plist,model_ind]=VI(data[,P+2]+1,config)
  
  # VI 
  VI_dist[P==plist,model_ind]=VI(data[,P+2]+1,matrix(config_vi,1,N))
  
  # number of clusters in VI estimate
  k_hat=max(config_vi)
  kVI[P==plist,model_ind]=k_hat
  
  ## Plot clustering estimate
  #plotname=paste('./EmE/EmE',P,'_initS_1_linexam/VIestimate.pdf',sep="")
  plotname=paste('./JmE/JmE',P,'/VIestimate_v2.pdf',sep="")
  pdf(plotname)
  Xmean=apply(X,1,mean)
  cl=c("black", "red", "green", "blue", "yellow", "orange", "magenta", "cyan", "gray","darkgray","blueviolet","darkmagenta","darkred", "darkorange", "darkorange4", "cornflowerblue", "darkolivegreen2", "darkolivegreen4", "brown","chartreuse","burlywood1", "violet","violetred","yellow4","tomato3","springgreen", "slateblue","tan","seashell","thistle" ) 
  par(mfrow=c(1,1),mar=c(2.5,2.5,1.5,.5)+.1, mgp=c(1.5, .5, 0))
  plot(Xmean,Y,xlab="x_avg",ylab="y",ylim=c(min(Y)-.1,max(Y)+.1),cex.lab=1.4,font=2,font.lab=2,lwd=2)
  for(j in 1:k_hat){
    points(Xmean[config_vi==j], Y[config_vi==j],col=cl[j],pch=16,cex=1.5)
  }
  dev.off()
  
  ## Credible ball
  cb=credibleball(config_vi, config)
  summary(cb)
  cb_size[P==plist,model_ind]=cb$dist.horiz
  print(paste('VI_dist=',VI_dist[P==plist,model_ind],' CB size=',cb_size[P==plist,model_ind]))
}

################################

#Plot k
kxtot=rep(0,length(plist))
for (ind in 1:length(plist)){
  kxtot[ind]=sum(kxVI[[ind]])
}
par(mfrow=c(1,1),mar=c(2.2,2.2,.5,.5)+.1, mgp=c(1.2, .1, 0))
plot(plist,kVI[,2],type='b',col=2, ylim=c(-0.9,13),xlab='D',ylab="clusters",cex=1.5, font=2,font.lab=2,lwd=2,cex.lab=1.4)
lines(plist,kxtot,type='b',lty=2, col=2,cex=1.5,lwd=2)
lines(plist,kVI[,1],type='b',col=1,cex=1.5,lwd=2)
lines(plist,kVI_lin,type='b',col=6,cex=1.5,lwd=2)
legend(0.1,1.5,c("EDP:y-clusters","EDP:x-clusters","DP","EDPlin"), col=c(2,2,1,6), lty=c(1,2,1,1),lwd=c(2,2,2,2),ncol=2, cex=1.4,text.font=2,x.intersp=0.25,y.intersp=0.8,seg.len = 1)

#Plot VI distance with true
par(mfrow=c(1,1),mar=c(2.2,2.2,.5,.5)+.1, mgp=c(1.2, .1, 0))
plot(plist,VI_dist[,2],type='b',col=2, ylim=c(-0.5,3),xlab='D',ylab="VI distance",cex=1.5, font=2,font.lab=2,lwd=2,cex.lab=1.4)
lines(plist,cb_size[,2],type='b',lty=2,col=2,cex=1.5,lwd=2)
lines(plist,VI_dist[,1],type='b',col=1,cex=1.5,lwd=2)
lines(plist,cb_size[,1],type='b',lty=2,col=1,cex=1.5,lwd=2)
#lines(plist,VI_dist_lin,type='b',col=6,cex=1.5,lwd=2)
#lines(plist,cb_size_lin,type='b',lty=2,col=6,cex=1.5,lwd=2)
#legend(0.1,-0.2,c("EDP","DP","EDPlin"), col=c(2,1,6), lty=c(1,1,1),lwd=c(2,2,2),horiz=T,cex=1.4,text.font=2,x.intersp=0.25,y.intersp=0.25,seg.len = 1)
legend(0.1,-0.2,c("EDP","DP"), col=c(2,1), lty=c(1,1),lwd=c(2,2),horiz=T,cex=1.4,text.font=2,x.intersp=0.25,y.intersp=0.25,seg.len = 1)

