
source("simu_Dombry_et_al.R")

## simulate Brown-Resnick processes via the sprectral measure (corresponds to 
## the algorithm devised by Dieker and Mikosch, 2105) and extremal functions
coord <- cbind(1:5, -2:2)
vario <- function(x) sqrt(sum(x^2))
res1  <- simu_specfcts(model="brownresnick", no.simu=100, coord=coord, 
                       vario=vario)

coord

res1
