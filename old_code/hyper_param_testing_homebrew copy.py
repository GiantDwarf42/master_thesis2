#%%
import torch
import numpy as np
from datetime import datetime
import pandas as pd
import matplotlib.pyplot as plt

from scipy.stats import multivariate_normal
from scipy.stats import dirichlet 
from scipy.stats import norm
from torch.distributions.multivariate_normal import MultivariateNormal 

#self written modules
import MMD

#%%
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

#%%
seed = 42
torch.manual_seed(seed)
np.random.seed(seed)

#%%
######### 
# one dim case first generator

sample_size = 100
y_sample_size = 1000
nr_iterations = 3000
lr = 0.1
nr_simulations = 1



mu = -100
sigma = 4

folder_name= "home_brew_test"
y = torch.normal(mu,sigma,(y_sample_size,1))
y = y.squeeze(1)

# b heuristic could also just be a value
b = MMD.calc_b_heuristic_1d(y_sample_size, y)


simulation_results = []

#%%

#%%
for sim in np.arange(nr_simulations):
    now=datetime.now()

    mu_hat = torch.tensor([0.]).to(device).requires_grad_()
    sigma_hat = torch.tensor([1.]).to(device).requires_grad_()

#####
# setup optimizer
    optimizer = torch.optim.Adam([mu_hat,sigma_hat], lr=lr)

    simulation_index = pd.DataFrame({"sim_nr":np.repeat(sim, nr_iterations)})

    simulated_df = MMD.training_loop_gauss(mu_hat,sigma_hat, y, nr_iterations, sample_size, device, b, optimizer)

    simulated_df = pd.concat([simulated_df,simulation_index], axis=1)

    simulation_results.append(simulated_df)

    print(f"Simulation nr {sim}, done in {datetime.now()-now}")
    


#%%
stacked_results = pd.concat(simulation_results).reset_index(drop=True)


#%%
stacked_results.to_csv(f"{folder_name}/mu-2sigma3.csv")
#%%

#%%
for index,df in enumerate(simulation_results):

    plt.plot(df[["mu_hat", "sigma_hat", "MMD"]])
    plt.axhline(y=mu)
    plt.axhline(y=sigma)
    plt.title(f"Simulation nr {index}")
    plt.legend()
    plt.savefig(f"{folder_name}/plot_nr{index}")
    plt.show()
    

#%%

for index, df in enumerate(simulation_results):

    print(df.tail(100))
    print((df.MMD <0).value_counts())
    print(df.nsmallest(10, "MMD"))
#%%
simulation_results
#%%
y.mean()
#%%
y.std()
#%%
norm.fit(y)

#%%
simulation_results[0].tail()
#%%
#%%
b.shape

#%%

a = torch.normal(0,1,(10,1))

a
#%%
torch.cdist(a,a).diag()
#%%
a
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


