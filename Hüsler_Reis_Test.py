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

coord = torch.tensor([np.arange(1.,6.), np.arange(-2.,3.,)]).t()

coord

#%%
def vario_function(x):
	return torch.sqrt(torch.sum(x**2, dim=-1))

vario = vario_function

N = coord.shape[0]

res1 = MMD.sim_huesler_reis(coord, vario, device, no_simu=10)
res1


#%%
def sim_huesler_reis_ext(coord, vario, loc=0., scale=1., shape=1., no_simu=1):

	assert isinstance(coord, torch.tensor), f"coord must be a torch.tensor but is a {type(coord)}"

	N = coord.shape[0]

	if isinstance(loc, float):

		loc = torch.tensor(np.repeat(loc, N), requires_grad=True)
		loc = loc.to(device)
     

	if isinstance(scale, float):

		scale = torch.tensor(np.repeat(scale, N), requires_grad=True)
		scale = scale.to(device)

	if isinstance(shape, float):

		shape = torch.tensor(np.repeat(shape, N), requires_grad=True)
		shape = shape.to(device) 

	assert torch.all(scale > 1e-12), f"all scale values must be bigger than 1e-12"

	assert callable(vario), f"vario must be a function" 

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
	chol_mat = torch.linalg.cholesky(cov_mat, upper=True)

	# Initialize a zero matrix res with shape (no_simu, N)
	res = torch.zeros((no_simu, N))

	# Initialize a zero vector counter with length no_simu
	counter = torch.zeros(no_simu, dtype=torch.int)


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

#%%

#%%




