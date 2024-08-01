
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

N <- nrow(coord)



cov.mat <- sapply(1:N, function(i) sapply(1:N, function(j) 
                     vario(coord[i,]) + vario(coord[j,]) - vario(coord[i,]-coord[j,])))

cov.mat


chol(cov.mat)

trend <- sapply(1:N, function(k) sapply(1:N, function(j) vario(coord[j,]-coord[k,])))
trend

poisson <- rexp(10)
poisson



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



matrix(1, nrow = N, ncol = no.simu)



rep(1, no.simu)






length(shift)
chol.mat


res1  <- simu_specfcts(no.simu=100, coord=coord, 
                       vario=vario)
res1


matrix(1:N*no.simu, nrow = N)


matrix(rnorm(N*no.simu), ncol=no.simu)

matrix(1, nrow = N, ncol = no.simu)
