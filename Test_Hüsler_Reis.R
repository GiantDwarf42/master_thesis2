
source("Hüsler_Reis_reduced.R")
#source("simu_Dombry_et_al.R")

set.seed(42)

## simulate Brown-Resnick processes via the sprectral measure (corresponds to 
## the algorithm devised by Dieker and Mikosch, 2105) and extremal functions
coord <- cbind(1:5, -2:2)
vario <- function(x) sqrt(sum(x^2))
res1  <- simu_specfcts(model="brownresnick", no.simu=100, coord=coord, 
                       vario=vario)
                     

res1  <- simu_specfcts(no.simu=100, coord=coord, 
                       vario=vario)

coord

res1

N <- nrow(coord)



cov.mat <- sapply(1:N, function(i) sapply(1:N, function(j) 
                     vario(coord[i,]) + vario(coord[j,]) - vario(coord[i,]-coord[j,])))

cov.mat


chol(cov.mat)

trend <- sapply(1:N, function(k) sapply(1:N, function(j) vario(coord[j,]-coord[k,])))
trend

poisson <- rexp(10)
poisson


simu_px_brownresnick <- function(no.simu=1, idx,  N, trend, chol.mat) {
  stopifnot(length(idx)==1 || length(idx)==no.simu)
  res <- t(chol.mat)%*%matrix(rnorm(N*no.simu), ncol=no.simu)

  res <- t(chol.mat)%*%matrix(matrix(1, nrow = N, ncol = no.simu), ncol=no.simu)

  if (!is.matrix(trend)) {
    res <- exp(t(res - trend))
  } else {
    res <- exp(t(res - trend[,idx]))   
  }

  print(res)
  return(res/res[cbind(1:no.simu,idx)])
}


N <- 5
no.simu <- 10



set.seed(42)

n.ind <- 10

shift <- sample(1:N, n.ind, replace=TRUE)

chol.mat <- chol(cov.mat)

no.simu
shift
N
trend
chol.mat

simu_px_brownresnick(no.simu, shift, N, trend, chol.mat)


res = -5:4
(N/poisson > apply(matrix(res), 1, min))

N/poisson

res













length(shift)
chol.mat


res1  <- simu_specfcts(no.simu=100, coord=coord, 
                       vario=vario)
res1


matrix(1:N*no.simu, nrow = N)


matrix(rnorm(N*no.simu), ncol=no.simu)

matrix(1, nrow = N, ncol = no.simu)
