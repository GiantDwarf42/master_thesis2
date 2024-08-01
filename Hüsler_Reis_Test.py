#%%
 #self written modules
import MMD

#%%

import torch
import numpy as np
from datetime import datetime
import pandas as pd
import matplotlib.pyplot as plt
from scipy.stats import norm


#%%
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
torch.set_default_dtype(torch.float64)



#%%
# this is necessary to not just get the same values 150 times
seed = int(42)
torch.manual_seed(seed)
np.random.seed(seed)
#%%

coord = torch.tensor([np.arange(1.,6.), np.arange(-2.,3.,)]).T

coord

#%%
def vario_function(x):
	return torch.sqrt(torch.sum(x**2, dim=-1))

vario = vario_function

N = coord.shape[0]
#%%


#%%
def sim_huesler_reis(coord, vario, loc=1., scale=1., shape=1., no_simu=1):

	N = coord.shape[0]

	if isinstance(loc, float):

		loc = torch.tensor(np.repeat(loc, N), requires_grad=True).to(device)

	if isinstance(scale, float):

		scale = torch.tensor(np.repeat(scale, N), requires_grad=True).to(device)

	if isinstance(shape, float):

		shape = torch.tensor(np.repeat(shape, N), requires_grad=True).to(device)

	assert torch.all(scale > 1e-12), f"Not all elements in 'scale' {scale} are greater than 1e-12"

	assert callable(vario), f" vario must be a function"

	# calculate the covariance matrix
	# Compute pairwise differences using broadcasting
	coord_i = coord.unsqueeze(1).expand(N, N, 2)  # Shape (N, N, 2)
	coord_j = coord.unsqueeze(0).expand(N, N, 2)  # Shape (N, N, 2)
	diff = coord_i - coord_j  # Shape (N, N, 2)

	# Apply the vario function
	vario_diff = vario(diff)  # Shape (N, N)

	# Apply the vario function to the original coordinates
	vario_coord = vario(coord)  # Shape (N,)

	# Compute the covariance matrix
	cov_mat = vario_coord.unsqueeze(1) + vario_coord.unsqueeze(0) - vario_diff
	# I guess we add this for numerical reasons?
	cov_mat = cov_mat + 1e-6

	# cholevski decomposition for upper triangular matrix

	chol_mat = torch.cholesky(cov_mat, upper=True)

	# get the trend which is the same as the difference
	trend = vario_diff

	# Initialize a zero matrix res with shape (no_simu, N)
	res = torch.zeros((no_simu, N))

	# Initialize a zero vector counter with length no_simu
	counter = torch.zeros(no_simu, dtype=torch.int)

	# draw exponential rv
	poisson = torch.zeros([no_simu]).exponential_(lambd=1).to(device)
	#poisson = torch.ones([no_simu]).to(device)

	# get the loop termination condition
	ind = torch.tensor(np.repeat(True, no_simu))


	# actual algorithm => hopefully auto.diff can handle this loop

	while(ind.any()):
		
		
		n_ind = torch.sum(ind).item()
		counter[ind] = counter[ind] + 1


		shift = torch.randint(0, N, (n_ind,), dtype=torch.int)


		# draw from the Hüsler Reis distribution
		proc = simu_px_brownresnick(n_ind, shift, N, trend, chol_mat)


		assert proc.shape == (n_ind, N), f"Shape of proc {proc.shape} does not match the expected dimensions {(n_ind, N)}"

		proc = N * proc / proc.sum(dim=1, keepdim=True)

		# maybe unsqueeze is an issue keep in mind
		res[ind, :] = torch.maximum(res[ind, :], proc / poisson[ind].unsqueeze(1))

		# create additional exponential term
		exp_rv = torch.zeros(n_ind).exponential_(lambd=1).to(device)

		poisson[ind] = poisson[ind] + exp_rv

		ind = (N / poisson > res.min(dim=1).values)

		print(f"{ind}")



	print("loop done")
	res_transformed = torch.where(
		torch.abs(shape) < 1e-12,
		torch.log(res) * scale.unsqueeze(0) + loc.unsqueeze(0),
		(1 / shape.unsqueeze(0)) * (res ** shape.unsqueeze(0) - 1) * scale.unsqueeze(0) + loc.unsqueeze(0)
		)


	return {"res": res_transformed,
	 	"counter": counter}

	


#%%
def simu_px_brownresnick(no_simu, idx, N, trend, chol_mat):

	# Check the condition and raise an error if not met
	assert idx.numel() == 1 or idx.numel() == no_simu, "Length of idx must be 1 or no_simu"


	# Generate random normal matrix with N rows and no_simu columns
	random_matrix = torch.randn(N, no_simu)

	#random_matrix = torch.ones(N, no_simu)

	# Perform matrix multiplication
	res = torch.mm(chol_mat.t(), random_matrix)

	# Apply trend and calculate exponentiated results
	if not isinstance(trend, torch.Tensor):
		trend = torch.tensor(trend)

	# Apply trend and calculate exponentiated results
	if trend.dim() == 1:
        	res = torch.exp((res - trend).t())
	
	else:
		res = torch.exp((res - trend[:, idx]).t())

	
	# Normalize the results
	norm_factor = res[torch.arange(no_simu), idx]
	res = res / norm_factor.unsqueeze(1)



	return res


#%%


no_simu = 10
N = 5
shift = torch.tensor([1, 5, 1, 1, 2, 4, 2, 2, 1, 4]) -1

shift

#%%
trend = torch.tensor([[0., 1.414214, 2.828427, 4.242641, 5.656854],
		     [1.414214, 0.000000, 1.414214, 2.828427, 4.242641],
		     [2.828427, 1.414214, 0.000000, 1.414214, 2.828427],
		     [4.242641, 2.828427, 1.414214, 0.000000, 1.414214],
		     [5.656854, 4.242641, 2.828427, 1.414214, 0.000000]])

trend
#%%
chol_mat = torch.tensor([[2.114743, 1.446002, 1.138503, 1.000847, 0.9288973],
			 [0.000000, 1.543118, 1.409857, 1.350201, 1.3190206],
			 [0.000000, 0.000000, 1.648063, 1.617555, 1.6016097],
			 [0.000000, 0.000000, 0.000000, 1.674810, 1.6674951],
			 [0.000000, 0.000000, 0.000000, 0.000000, 1.6798721]])

chol_mat
#%%
t = simu_px_brownresnick(no_simu, shift, N, trend, chol_mat)

t.shape
#%%
idx = 10
res = sim_huesler_reis(coord, vario, no_simu=100)


#%%
coord
#%%
res
#%%
#%%
#%%
#%%
#%%
#%%
#%%
#%%
#%%
#%%
#%%
#%%