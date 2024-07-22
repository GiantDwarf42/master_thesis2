# %% [markdown]
# # Finding the source of NaN






# %% [markdown]
# # Setup

# %%
import torch
import numpy as np
from datetime import datetime
import pandas as pd
import matplotlib.pyplot as plt
from scipy.stats import norm

#self written modules
import MMD

#%%
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
torch.set_default_dtype(torch.float64)



#%%
# this is necessary to not just get the same values 150 times
seed = int(42)
torch.manual_seed(seed)
np.random.seed(seed)


# %%
lr = 0.1


#######################################################################################################################3
# setup parameter permutations

# #sample sizes
x_sample_sizes = [50]
y_sample_sizes = [50]

# true parameters
alpha_list = [0.05]

#dims
dim_list = [2]

# bandwidth
b_list = [0.01]

# number number iterations
nr_iterations = 800




#going through parameter permutation

for b_value in b_list:

	for alpha in alpha_list:

		for dim in dim_list:

			for x_sample_size in x_sample_sizes:

				for y_sample_size in y_sample_sizes:
		
					if x_sample_size >= y_sample_size:
		
			
					


			

						y = MMD.sample_multivariate_logistic(y_sample_size,dim,alpha,device)

						# check if b heuristic needs to be calculated
						if b_value == "AUTO":

							# starting point for bandwidth 
							b_params = {"alpha": 0.5}

							# b heuristic could also just be a value
							b = MMD.calc_b_heuristic(y, x_sample_size, "logistic", device, b_params).item()

						else:
							b = b_value

						#check if b needs to be iteratively updated
						if b_value == "AUTO":
							b_update = 100
						else:
							b_update = False
			
			

						#take start time
						now=datetime.now()

						#setup the parameters to optimize
						alpha_hat = torch.tensor([0.5]).to(device).requires_grad_()
						
						# setup optimizer
						optimizer = torch.optim.Adam([alpha_hat], lr=lr)

			
						# run the actual simulation
						simulated_df = MMD.training_loop_multi_logist(alpha_hat, y, nr_iterations, x_sample_size, device, b, optimizer, epoch_print_size=False,b_update=b_update)

			

			

# %%
simulated_df

# %%


# %%



