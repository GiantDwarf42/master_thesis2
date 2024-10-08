## internal functions: do not use any of them directly!
simu_px_brownresnick <- function(no.simu=1, idx,  N, trend, chol.mat) {
  stopifnot(length(idx)==1 || length(idx)==no.simu)
  
  # random component
  res <- t(chol.mat)%*%matrix(rnorm(N*no.simu), ncol=no.simu)
  #res <- t(chol.mat)%*%matrix(1, nrow = N, ncol = no.simu)
  
  if (!is.matrix(trend)) {
    res <- exp(t(res - trend))
  } else {
    res <- exp(t(res - trend[,idx]))   
  }
  return(res/res[cbind(1:no.simu,idx)])
}

## main functions

simu_specfcts <- function(loc=0, scale=1, shape=1, no.simu=1, 
                          coord, vario) {

 
  N <- nrow(coord) 
  
  
  if (length(loc)  ==1) loc   <- rep(loc  , times=N)
  if (length(scale)==1) scale <- rep(scale, times=N)
  if (length(shape)==1) shape <- rep(shape, times=N)
  stopifnot(all(scale>1e-12))
  

  stopifnot(is.function(vario))
  
  cov.mat <- sapply(1:N, function(i) sapply(1:N, function(j) 
                     vario(coord[i,]) + vario(coord[j,]) - vario(coord[i,]-coord[j,])))
  cov.mat <- cov.mat + 1e-6 
  #add constant random effect to avoid numerical problems            
  
  # cholevski decomposition
  chol.mat <- chol(cov.mat)
  
  
  trend <- sapply(1:N, function(k) sapply(1:N, function(j) vario(coord[j,]-coord[k,])))
   
  res <- matrix(0, nrow=no.simu, ncol=N)
  counter <- rep(0, times=no.simu)
  
  # random component
  poisson <- rexp(no.simu)
  #poisson <- rep(1, no.simu)

  ind <- rep(TRUE, times=no.simu)
  
  while (any(ind)) {
    n.ind <- sum(ind)
    counter[ind] <- counter[ind] + 1
    shift <- sample(1:N, n.ind, replace=TRUE)

    proc <- simu_px_brownresnick(no.simu=n.ind, idx=shift, N=N, trend=trend, chol.mat=chol.mat)
 
    stopifnot(dim(proc)==c(n.ind, N))
    proc <- N*proc/rowSums(proc)     
    res[ind,] <- pmax(res[ind,], proc/poisson[ind])
    
    # random component
    poisson[ind] <- poisson[ind] + rexp(n.ind)
    #poisson[ind] <- poisson[ind] + rep(1, n.ind)
    
    
    ind <- (N/poisson > apply(res, 1, min))
  }
  res <- sapply(1:N, function(i) {
           if (abs(shape[i]<1e-12)) {   
             return(log(res[,i])*scale[i] + loc[i])
           } else {
             return(1/shape[i]*(res[,i]^shape[i]-1)*scale[i] + loc[i])
           }
         })   
   
  return(list(res=res, counter=counter)) 
}

simu_extrfcts <- function(coord, vario, 
                loc=1, scale=1, shape=1, no.simu=1) {
                            
        browser()
  
        stopifnot(!missing(coord))
        if (!is.matrix(coord)) coord <- matrix(coord, ncol=1)   
        N <- nrow(coord)
  
  
        stopifnot((N==round(N)) & (N>=1))
        stopifnot((no.simu==round(no.simu)) & (no.simu>=1))
  
        if (length(loc)  ==1) loc   <- rep(loc  , times=N)
        if (length(scale)==1) scale <- rep(scale, times=N)
        if (length(shape)==1) shape <- rep(shape, times=N)
        stopifnot(all(scale>1e-12))


        stopifnot(is.function(vario))
        cov.mat <- sapply(1:N, function(i) sapply(1:N, function(j) 
                     vario(coord[i,]) + vario(coord[j,]) - vario(coord[i,]-coord[j,])))
        cov.mat <- cov.mat + 1e-6

        
        #add constant random effect to avoid numerical problems            
        chol.mat <- chol(cov.mat)
  
   
        res <- matrix(0, nrow=no.simu, ncol=N)
        counter <- rep(0, times=no.simu)
   
        for (k in 1:N) {
                poisson <- rexp(no.simu)
                #poisson <- rep(1, no.simu)

                trend <- sapply(1:N, function(j) vario(coord[j,]-coord[k,]))
                

                while (any(1/poisson > res[,k])) {
                        ind <- (1/poisson > res[,k])
                        n.ind <- sum(ind)
                        idx <- (1:no.simu)[ind]
                        counter[ind] <- counter[ind] + 1

                        proc <- simu_px_brownresnick(no.simu=n.ind, idx=k, N=N, trend=trend, chol.mat=chol.mat)
                
                        stopifnot(dim(proc)==c(n.ind, N))
                
                
                        if (k==1) {
                                ind.upd <- rep(TRUE, times=n.ind)
                        } else {
                                ind.upd <- sapply(1:n.ind, function(i) 
                                                        all(1/poisson[idx[i]]*proc[i,1:(k-1)] <= res[idx[i],1:(k-1)]))
                        }
                        if (any(ind.upd)) {
                                idx.upd <- idx[ind.upd]
                                res[idx.upd,] <- pmax(res[idx.upd,], 1/poisson[idx.upd]*proc[ind.upd,])
                        }
                        poisson[ind] <- poisson[ind] + rexp(n.ind)
                        #poisson[ind] <- poisson[ind] + rep(1,n.ind)
                        } 
                }
        res <- sapply(1:N, function(i) {
                if (abs(shape[i]<1e-12)) {   
                return(log(res[,i])*scale[i] + loc[i])
                } else {
                return(1/shape[i]*(res[,i]^shape[i]-1)*scale[i] + loc[i])
                }
         })   
   
        return(list(res=res, counter=counter))  
}

