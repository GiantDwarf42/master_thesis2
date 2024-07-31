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
seed = int(task_id)
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

		loc = torch.tensor(np.repeat(loc, N))

	if isinstance(scale, float):

		scale = torch.tensor(np.repeat(scale, N))

	if isinstance(shape, float):

		shape = torch.tensor(np.repeat(shape, N))

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


	



#%%


sim_huesler_reis(coord, vario)
#%%
#%%
torch.tensor([np.repeat(1., 10), np.repeat(0.,10)]).all() < -0.5


#%%
1e-12 +1
#%%

callable(vario)
#%%


#%%
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

cov_mat
#%%
#%%

#%%

#%%