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



#%%
def sim_huesler_reis_ext(coord, vario, device, loc=0., scale=1., shape=1., no_simu=1):

	assert isinstance(coord, torch.Tensor), f"coord must be a torch.tensor but is a {type(coord)}"

	N = coord.shape[0]

	if isinstance(loc, float):

		loc = torch.tensor(np.repeat(loc, N), device=device)
		
     

	if isinstance(scale, float):

		scale = torch.tensor(np.repeat(scale, N), device=device)
		

	if isinstance(shape, float):

		shape = torch.tensor(np.repeat(shape, N), device=device)
		

	assert torch.all(scale > 1e-12), f"all scale values must be bigger than 1e-12"

	assert callable(vario), f"vario must be a function" 

	# Compute covariance matrix using broadcasting
	coord_i = coord.unsqueeze(1)  # Shape (N, 1, d)
	coord_j = coord.unsqueeze(0)  # Shape (1, N, d)
	cov_matrix = vario(coord_i) + vario(coord_j) - vario(coord_i - coord_j)
	cov_matrix += 1e-6  # Add small constant for numerical stability

	# cholevski decomposition for upper triangular matrix
	chol_mat = torch.linalg.cholesky(cov_matrix, upper=True)


	# Initialize a zero matrix res with shape (no_simu, N)
	res = torch.zeros((no_simu, N))


	# Initialize a zero vector counter with length no_simu
	counter = torch.zeros(no_simu, dtype=torch.int)



	for k in range(N):

		# create additional exponential term
		# random component
		poisson = torch.zeros(no_simu).exponential_(lambd=1).to(device)
		#poisson = torch.ones(no_simu).to(device)

		trend = vario(coord -coord[k])

		while torch.any(1 / poisson > res[:, k]):
			ind = 1 / poisson > res[:, k]
			n_ind = ind.sum().item()
			idx = torch.arange(no_simu)[ind]
			counter[ind] += 1

			proc = MMD.simu_px_brownresnick(no_simu=n_ind, idx=torch.tensor([k]), N=N, trend=trend, chol_mat=chol_mat)
			
			assert proc.shape == (n_ind, N), f"Shape of proc {proc.shape} does not match the expected dimensions {(n_ind, N)}"


			if k == 1:

				ind_upd = torch.tensor(np.repeat(True, n_ind))
			else:
				ind_upd = torch.tensor([torch.all(1 / poisson[idx[i]] * proc[i, :k] <= res[idx[i], :k]) for i in range(n_ind)])

			if ind_upd.any():
				idx_upd = idx[ind_upd]
				res[idx_upd, :] = torch.maximum(res[idx_upd, :], 1 / poisson[idx_upd].unsqueeze(1) * proc[ind_upd, :])

			poisson[ind] = poisson[ind] + torch.zeros(n_ind).exponential_(lambd=1)

	
	# Apply final transformation
	res_transformed = torch.where(
		torch.abs(shape) < 1e-12,
		torch.log(res) * scale.unsqueeze(0) + loc.unsqueeze(0),
		(1 / shape.unsqueeze(0)) * (res ** shape.unsqueeze(0) - 1) * scale.unsqueeze(0) + loc.unsqueeze(0)
    )


	return {"res": res_transformed, "counter": counter}
#%%

#%%

#%%
res2 = sim_huesler_reis_ext(coord, vario, device, no_simu=10)
res2

#%%
type(coord)
#%%
res2["res"].shape

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




