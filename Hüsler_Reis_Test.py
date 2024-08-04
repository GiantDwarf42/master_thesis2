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
res1 = MMD.sim_huesler_reis_ext(coord, vario, device, no_simu=10)
res1


#%%

def vario_alpha_pnorm(x, alpha=1., p=2.):

	norm = alpha * (torch.sum(x**p, dim=-1))**(1/p)

	return norm 
	

#%%
vario = vario_alpha_pnorm
#%%
res2 = MMD.sim_huesler_reis_ext(coord, vario, device, no_simu=10)
res2

#%%
type(coord)
#%%
res2["res"].shape

#%%
torch.exp(10,30)


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




