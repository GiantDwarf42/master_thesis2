The R functions simu_extrfcts and simu_specfcts generate max-stable random 
vectors using Algorithm 1 (simulation via extremal functions) and Algorithm 2
(simulation via the spectral measure), respectively, in

Dombry, Clément, Engelke, Sebastian and Oesting, Marco (2016+). 
Exact simulation of max-stable processes. Biometrika.

The available max-stable models comprise the models presented in Section 4 
("Examples") of the above-mentioned paper: the Brown-Resnick process, the
extremal-t process, the logistic model, the negative logistic model and the
Dirichlet model. 

Usage:

simu_extrfcts(model, N, loc=1, scale=1, shape=1, no.simu=1, 
              coord, vario, corr, dof, theta, weights, alpha)
simu_specfcts(model, N, loc=1, scale=1, shape=1, no.simu=1, 
              coord, vario, corr, dof, theta, weights, alpha)

Arguments:

model:	one of the following character strings representing one of the above 
        mentioned models: 'brownresnick', 'extremalt', 'logistic', 
        'neglogistic' of 'dirichlet'
N:	dimension of the max-stable vector; not necessary in case of a spatial
	model ('brown-resnick' or 'extremalt') where the dimension is 
        implicitly given by the coordinates
loc, scale, shape: the GEV parameters; can be given either as single numbers
        (which means that the parameter value is the same for any component of 
        the max-stable vector) or as a vector with each component being the 
        parameter value for the corresponding component of the max-stable 
        vector
no.simu: number of independent realizations to be simulated
coord:  a matrix of size N x d which contains the coordinates corresponding to 
        the components of the random vector; here, each row corresponds to one
	component; only used for spatial models ('brownresnick' or 'extremalt')
vario:	a function returning the value of a variogram as a function of the 
	spatial difference vector; only used for the 'brownresnick' model
corr:	a function returning the correlation as a function of the 
	spatial difference vector; only used for the 'extremalt' model
dof:	the degrees of freedom in the 'extremalt' model (also called
        'alpha' in the above mentioned paper)
theta:	a parameter of the 'logistic' and 'neglogistic' model
weights: the mixture weights for the 'dirichlet' model (also called
        'lambda' in the above mentioned paper)
alpha:	a matrix with N rows which contains parameters of the 'dirichlet'
        model; each column corresponds to one model included in the mixture

Value:

The functions simu_estrfcts and simu_specfcts return a list with the elements
res:	a matrix of size no.simu x N which contains the simulated realizations
	of the max-stable vector
counter: a vector of the number of spectral functions that were simulated to
        obtain the realizations of the max-stable vector

Examples:

source("simu_Dombry_et_al.R")

## simulate Brown-Resnick processes via the sprectral measure (corresponds to 
## the algorithm devised by Dieker and Mikosch, 2105) and extremal functions
coord <- cbind(1:5, -2:2)
vario <- function(x) sqrt(sum(x^2))
res1  <- simu_specfcts(model="brownresnick", no.simu=100, coord=coord, 
                       vario=vario)
res2  <- simu_extrfcts(model="brownresnick", no.simu=100, coord=coord, 
                       vario=vario)

## simulate extremal-t processes via extremal functions
coord <- cbind(1:5, -2:2)
corr  <- function(x) exp(-sum(x^2))
res   <- simu_extrfcts(model="extremalt", no.simu=50, dof=1.5, coord=coord, 
                       corr=corr)

## simulate multivariate logistic and negative logistic models with standard 
## Gumbel marginals via extremal functions
res1 <- simu_extrfcts(model="logistic", N=10, no.simu=20, theta=0.7, loc=0, 
                      scale=1, shape=0)
res2 <- simu_extrfcts(model="neglogistic", N=10, no.simu=20, theta=0.7, loc=0, 
                      scale=1, shape=0)

## simulate Dirichlet mixture models via extremal functions
weights <- c(0.3,0.5,0.2)
alpha <- cbind(c(1,1,1,1,1),c(1.5,1.5,1.5,1.5,1.5), c(0.8,0.8,0.8,0.8,0.8))
res <- simu_extrfcts(model="dirichlet", N=5, no.simu=250, weights=weights, 
                     alpha=alpha)


