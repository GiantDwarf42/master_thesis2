
source("Hüsler_Reis_reduced.R")

#source("simu_Dombry_et_al.R")

set.seed(42)
## simulate Brown-Resnick processes via the sprectral measure (corresponds to 
## the algorithm devised by Dieker and Mikosch, 2105) and extremal functions
coord <- cbind(1:5, -2:2)
vario <- function(x) sqrt(sum(x^2))
coord
                     

res1  <- simu_specfcts(no.simu=10, coord=coord, 
                       vario=vario)




res1



res2 <- simu_extrfcts(no.simu = 10, coord = coord, vario = vario)
res2





simu_extrfcts <- function(coord, vario, 
                loc=1, scale=1, shape=1, no.simu=1) {
                            
        
  
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


N <- nrow(coord)

for (k in 1:N) {

                trend <- sapply(1:N, function(j) vario(coord[j,]-coord[k,]))
                print(trend)
                
}






trend















simu_extrfcts <- function(model, N, loc=1, scale=1, shape=1, no.simu=1, 
                          coord, vario, corr, dof, theta, weights, alpha) {
                            
  stopifnot(model %in% c("brownresnick", "extremalt", "logistic", "neglogistic", "dirichlet"))
  
  if (model %in% c("brownresnick", "extremalt")) {
    stopifnot(!missing(coord))
    if (!is.matrix(coord)) coord <- matrix(coord, ncol=1)   
    if (!missing(N)) stopifnot(N==nrow(coord))   
    N <- nrow(coord)   
  }
  
  stopifnot((N==round(N)) & (N>=1))
  stopifnot((no.simu==round(no.simu)) & (no.simu>=1))
  
  if (length(loc)  ==1) loc   <- rep(loc  , times=N)
  if (length(scale)==1) scale <- rep(scale, times=N)
  if (length(shape)==1) shape <- rep(shape, times=N)
  stopifnot(all(scale>1e-12))
  
  if (model=="brownresnick") {
    stopifnot(is.function(vario))
    cov.mat <- sapply(1:N, function(i) sapply(1:N, function(j) 
                     vario(coord[i,]) + vario(coord[j,]) - vario(coord[i,]-coord[j,])))
    cov.mat <- cov.mat + 1e-6 
    #add constant random effect to avoid numerical problems            
    chol.mat <- chol(cov.mat)
  } else if (model=="extremalt") {
    stopifnot(is.function(corr))
    stopifnot(dof>1e-12)
    diff.vector <- cbind(as.vector(outer(coord[,1],coord[,1],'-')),
                         as.vector(outer(coord[,2],coord[,2],'-')))  
    cov.mat.tmp <- matrix(apply(diff.vector, 1, function(x) corr(x)), ncol=N)       
  } else if (model=="logistic") {
    stopifnot(1e-12 < theta & theta < 1 - 1e-12)    
  } else if (model=="neglogistic") {
    stopifnot(theta > 1e-12) 
  } else if (model=="dirichlet") {
    m <- length(weights)
    stopifnot(all(weights>=0))
    stopifnot(abs(sum(weights)-1)<1e-12)
    stopifnot(length(alpha)==N*m)
    stopifnot(all(alpha>1e-12))
    if (N > 1 & m > 1) {
      stopifnot(is.matrix(alpha))
    } else {
      if (!is.matrix(alpha)) dim(alpha) <- c(N,m)
    }
    stopifnot(all(dim(alpha)==c(N,m)))
    norm.alpha <- apply(alpha, 2, function(alpha) alpha/sum(alpha))
    stopifnot(all(abs(norm.alpha %*% weights - rep(1/N, times=N))<1e-12))  
  }
   
  res <- matrix(0, nrow=no.simu, ncol=N)
  counter <- rep(0, times=no.simu)
   
  for (k in 1:N) {
    poisson <- rexp(no.simu)
    if (model == "brownresnick") {
      trend <- sapply(1:N, function(j) vario(coord[j,]-coord[k,]))
    } else if (model == "extremalt") {
      cov.vec  <- apply(coord, 1, function(x) corr(x-coord[k,]))
      cov.mat  <- (cov.mat.tmp - outer(cov.vec, cov.vec, '*'))/(dof+1) + 1e-6
      chol.mat <- chol(cov.mat)
      mu <- apply(coord, 1, function(x) corr(coord[k,]-x))    
    }
    while (any(1/poisson > res[,k])) {
      ind <- (1/poisson > res[,k])
      n.ind <- sum(ind)
      idx <- (1:no.simu)[ind]
      counter[ind] <- counter[ind] + 1
      proc <- switch(model,
                "brownresnick" = simu_px_brownresnick(no.simu=n.ind, idx=k, N=N, trend=trend, chol.mat=chol.mat),
                "extremalt"    = simu_px_extremalt(no.simu=n.ind, idx=k, N=N, dof=dof, mu=mu, chol.mat=chol.mat),
                "logistic"     = simu_px_logistic(no.simu=n.ind, idx=k, N=N, theta=theta),
                "neglogistic"  = simu_px_neglogistic(no.simu=n.ind, idx=k, N=N, theta=theta),
                "dirichlet"    = simu_px_dirichlet(no.simu=n.ind, idx=k, N=N, weights=weights, alpha=alpha, norm.alpha=norm.alpha)
              )
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
