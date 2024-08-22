
#source("G:/My Drive/Studium/UNIGE_Master/Thesis/Master_Thesis/Code2/Hüsler_Reis_reduced.R")
#source("G:/My Drive/Studium/UNIGE_Master/Thesis/Master_Thesis/Code2/simu_Dombry_et_al.R")

#source("simu_Dombry_et_al.R")
#source("Hüsler_Reis_reduced.R")




# Load the 'evd' package
library(evd)


set.seed(42)
## simulate Brown-Resnick processes via the sprectral measure (corresponds to 
## the algorithm devised by Dieker and Mikosch, 2105) and extremal functions
# coord <- cbind(1:10, -5:4)
# 
# coord <- cbind(c(-1,-1,1,1), c(-1,1,-1,1))
# coord <- cbind(1:10, -5:4)

coord <- cbind(c(1,0,0,0.5), c(0,0,1,0))

coord
vario <- function(x) 1 * sqrt(sum(x^2))^1
coord




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
  norm_factor <- res[cbind(1:no.simu,idx)]
  
  result <- res/norm_factor
  return(result)
}

## main functions

simu_extrfcts <- function(coord, vario, 
                          loc=1, scale=1, shape=1, no.simu=1) {
  
  #browser()
  
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
        #test <- sapply(1:n.ind, function(i) proc[i,1:(k-1)]) 
      }
      if (any(ind.upd)) {
        idx.upd <- idx[ind.upd]
        res[idx.upd,] <- pmax(res[idx.upd,], 1/poisson[idx.upd]*proc[ind.upd,])
      }
      poisson[ind] <- poisson[ind] + rexp(n.ind)
      #poisson[ind] <- poisson[ind] + rep(1,n.ind)
    } 
  }
  
  test <- sapply (1:N, function(i) {
    if (abs(shape[i]<1e-12)) {   
      log(res[,i])*scale[i] + loc[i]}})
  
  for (i in 1:N){
    
  
    i_test <- i
    shape_i <- shape[i]
    log_res_i <- log(res[,i])
    
    sahep_i <- scale[i]
    loc_i <- loc[i]
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

# for standard gumbell case => loc=0, scale=1 and shape=0

res1  <- simu_extrfcts(no.simu=1000000, coord=coord, 
                       vario=vario, loc = 0, scale = 1, shape=0)
res1 <- as.data.frame(res1$res)


res1




# Define a sequence of x values
x <- seq(-10, 10, length = 10000)

# Compute the density of the standard Gumbel distribution
y <- dgumbel(x, loc = 0, scale = 1)

# Plot the density
plot(x, y, type = "l", lwd = 2, col = "blue",
     main = "Density of the Standard Gumbel Distribution",
     xlab = "x", ylab = "Density")

for (i in 1:ncol(res1)){
  
  plot(density(as.matrix((res1[i]))),xlim=c(-5,10), col="red")
  lines(x, y, type = "l", lwd = 2, col = "blue",
             main = "Density of the Standard Gumbel Distribution",
             xlab = "x", ylab = "Density")
}



# get sum of truth value for first variable
u1 <- colSums(res1[1]<=1)
u1

p_u1 <- u1/nrow(res1)
p_u1

table(res1[1]<=1)
# logical for both variables to be smaller 1
u2_log <- rowSums(res1[c(1,4)] <=1) == 2
u2_log 
table(u2_log)

p_u2 <- sum(u2_log)/(nrow(res1))
p_u2

theta_sim <- log(p_u2)/log(p_u1)
theta_sim

vario(coord)

theta_theory <- 2*pnorm(sqrt(vario(coord)/2))

t <- sapply(1:nrow(coord), function(x){ vario(coord[x,])})
t
theta_theory <- 2*pnorm(sqrt(sum(0.5)/2))

theta_theory


theta_sim
coord








simu_extrfcts <- function(model, N, loc=1, scale=1, shape=1, no.simu=1, 
                           coord, vario, corr, dof, theta, weights, alpha) {

  
















                            
#   stopifnot(model %in% c("brownresnick", "extremalt", "logistic", "neglogistic", "dirichlet"))
  
#   if (model %in% c("brownresnick", "extremalt")) {
#     stopifnot(!missing(coord))
#     if (!is.matrix(coord)) coord <- matrix(coord, ncol=1)   
#     if (!missing(N)) stopifnot(N==nrow(coord))   
#     N <- nrow(coord)   
#   }
  
#   stopifnot((N==round(N)) & (N>=1))
#   stopifnot((no.simu==round(no.simu)) & (no.simu>=1))
  
#   if (length(loc)  ==1) loc   <- rep(loc  , times=N)
#   if (length(scale)==1) scale <- rep(scale, times=N)
#   if (length(shape)==1) shape <- rep(shape, times=N)
#   stopifnot(all(scale>1e-12))
  
#   if (model=="brownresnick") {
#     stopifnot(is.function(vario))
#     cov.mat <- sapply(1:N, function(i) sapply(1:N, function(j) 
#                      vario(coord[i,]) + vario(coord[j,]) - vario(coord[i,]-coord[j,])))
#     cov.mat <- cov.mat + 1e-6 
#     #add constant random effect to avoid numerical problems            
#     chol.mat <- chol(cov.mat)
#   } else if (model=="extremalt") {
#     stopifnot(is.function(corr))
#     stopifnot(dof>1e-12)
#     diff.vector <- cbind(as.vector(outer(coord[,1],coord[,1],'-')),
#                          as.vector(outer(coord[,2],coord[,2],'-')))  
#     cov.mat.tmp <- matrix(apply(diff.vector, 1, function(x) corr(x)), ncol=N)       
#   } else if (model=="logistic") {
#     stopifnot(1e-12 < theta & theta < 1 - 1e-12)    
#   } else if (model=="neglogistic") {
#     stopifnot(theta > 1e-12) 
#   } else if (model=="dirichlet") {
#     m <- length(weights)
#     stopifnot(all(weights>=0))
#     stopifnot(abs(sum(weights)-1)<1e-12)
#     stopifnot(length(alpha)==N*m)
#     stopifnot(all(alpha>1e-12))
#     if (N > 1 & m > 1) {
#       stopifnot(is.matrix(alpha))
#     } else {
#       if (!is.matrix(alpha)) dim(alpha) <- c(N,m)
#     }
#     stopifnot(all(dim(alpha)==c(N,m)))
#     norm.alpha <- apply(alpha, 2, function(alpha) alpha/sum(alpha))
#     stopifnot(all(abs(norm.alpha %*% weights - rep(1/N, times=N))<1e-12))  
#   }
   
#   res <- matrix(0, nrow=no.simu, ncol=N)
#   counter <- rep(0, times=no.simu)
   
#   for (k in 1:N) {
#     poisson <- rexp(no.simu)
#     if (model == "brownresnick") {
#       trend <- sapply(1:N, function(j) vario(coord[j,]-coord[k,]))
#     } else if (model == "extremalt") {
#       cov.vec  <- apply(coord, 1, function(x) corr(x-coord[k,]))
#       cov.mat  <- (cov.mat.tmp - outer(cov.vec, cov.vec, '*'))/(dof+1) + 1e-6
#       chol.mat <- chol(cov.mat)
#       mu <- apply(coord, 1, function(x) corr(coord[k,]-x))    
#     }
#     while (any(1/poisson > res[,k])) {
#       ind <- (1/poisson > res[,k])
#       n.ind <- sum(ind)
#       idx <- (1:no.simu)[ind]
#       counter[ind] <- counter[ind] + 1
#       proc <- switch(model,
#                 "brownresnick" = simu_px_brownresnick(no.simu=n.ind, idx=k, N=N, trend=trend, chol.mat=chol.mat),
#                 "extremalt"    = simu_px_extremalt(no.simu=n.ind, idx=k, N=N, dof=dof, mu=mu, chol.mat=chol.mat),
#                 "logistic"     = simu_px_logistic(no.simu=n.ind, idx=k, N=N, theta=theta),
#                 "neglogistic"  = simu_px_neglogistic(no.simu=n.ind, idx=k, N=N, theta=theta),
#                 "dirichlet"    = simu_px_dirichlet(no.simu=n.ind, idx=k, N=N, weights=weights, alpha=alpha, norm.alpha=norm.alpha)
#               )
#       stopifnot(dim(proc)==c(n.ind, N))
#       if (k==1) {
#         ind.upd <- rep(TRUE, times=n.ind)
#       } else {
#         ind.upd <- sapply(1:n.ind, function(i) 
#                                    all(1/poisson[idx[i]]*proc[i,1:(k-1)] <= res[idx[i],1:(k-1)]))
#       }
#       if (any(ind.upd)) {
#         idx.upd <- idx[ind.upd]
#         res[idx.upd,] <- pmax(res[idx.upd,], 1/poisson[idx.upd]*proc[ind.upd,])
#       }
#       poisson[ind] <- poisson[ind] + rexp(n.ind)
#     } 
#   }
#   res <- sapply(1:N, function(i) {
#            if (abs(shape[i]<1e-12)) {   
#              return(log(res[,i])*scale[i] + loc[i])
#            } else {
#              return(1/shape[i]*(res[,i]^shape[i]-1)*scale[i] + loc[i])
#            }
#          })   
   
#   return(list(res=res, counter=counter))  
# }



# simu_px_brownresnick <- function(no.simu=1, idx,  N, trend, chol.mat) {
#   stopifnot(length(idx)==1 || length(idx)==no.simu)
#   res <- t(chol.mat)%*%matrix(rnorm(N*no.simu), ncol=no.simu)
#   if (!is.matrix(trend)) {
#     res <- exp(t(res - trend))
#   } else {
#     res <- exp(t(res - trend[,idx]))   
#   }
#   return(res/res[cbind(1:no.simu,idx)])
# }


#res2 <- simu_extrfcts(model="brownresnick", no.simu = 10, coord = coord, vario = vario)
#res2



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

#res2 <- simu_extrfcts(model="brownresnick", no.simu = 10, coord = coord, vario = vario)
#res2

res1  <- simu_extrfcts(no.simu=10, coord=coord, 
                       vario=vario)
