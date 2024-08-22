
source("Hüsler_Reis_reduced.R")


set.seed(42)
## simulate Brown-Resnick processes via the sprectral measure (corresponds to 
## the algorithm devised by Dieker and Mikosch, 2105) and extremal functions
coord <- cbind(1:5, -2:2)
vario <- function(x) sqrt(sum(x^2))
coord
                     

res1  <- simu_extrfcts(no.simu=10, coord=coord, 
                       vario=vario)
res1


