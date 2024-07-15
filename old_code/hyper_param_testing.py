#%%
import torch
import numpy as np
from datetime import datetime
import pandas as pd
import matplotlib.pyplot as plt

from scipy.stats import multivariate_normal
from scipy.stats import dirichlet 
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
nr_itterations = 10000
bandwidth_range = [1]

nr_simulations = 1

mu = -2
sigma = 3

folder_name= "kaggle_test"
y = torch.normal(mu,sigma,(sample_size,1))


simulation_results = []

#%%

#%%
for sim in np.arange(nr_simulations):
    now=datetime.now()

    mu_hat = torch.tensor([0.]).to(device)
    sigma_hat = torch.tensor([1.]).to(device).requires_grad_()

#####
# setup optimizer
    optimizer = torch.optim.Adam([sigma_hat])

    simulation_index = pd.DataFrame({"sim_nr":np.repeat(sim, nr_itterations)})

    simulated_df = MMD.training_loop_gaussian_kaggle(mu_hat, sigma_hat, y, nr_itterations,sample_size, device, bandwidth_range, optimizer)

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
    plt.savefig(f"{folder_name}/plot_nr{index}")
    plt.show()
    

#%%

(stacked_results.MMD < 0).any()
#%%
#%%


#%%

for index, df in enumerate(simulation_results):

    print(df.sort_values(by="MMD").head(20))
#%%

#%%

def MMD_equal_case(x,device,b=1):

    n = x.shape[0]
    
    # Expand x to enable broadcasting
    x_expanded = x.unsqueeze(0).expand(n,-1).to(device)  # Shape: [n, n]
    x_expanded_t = x_expanded.t().to(device)
    
    
    #calculate kernel values
    kernel_values = kernel_gauss(x_expanded,x_expanded_t,b)
    
    #get the diagonal
    diag_kernel_values = kernel_values.diagonal()

    #remove the diagonal j==i elements
    sum = torch.sum(kernel_values) - torch.sum(diag_kernel_values)
    
    res = 1/(n*(n-1)) * sum
    

    return res

#%%
def kernel_gauss(x, y, b=1):

    res = torch.exp(-b*((x-y)**2))

    return res
#%%

def MMD_mixed_case(x,y,device,b=1):

    n = x.shape[0]
    m = y.shape[0]
    # Expand x and y to enable broadcasting
    x_expanded = x.unsqueeze(0).expand(m,-1).to(device)  # Shape: [n, m]
    y_expanded = y.unsqueeze(0).expand(n,-1).t().to(device)  # Shape: [n, m]

 
    print(x_expanded)
    print(y_expanded)
    #calculate kernel values
    kernel_values = kernel_gauss(x_expanded,y_expanded,b)

    print(kernel_values)
    
    

    #get the sum
    sum = torch.sum(kernel_values) 

    #weighting the sum
    res = 2/(m*n) * sum
    
    

    return res



#%%
def calc_MMD_1d(x,y,device, b=1):

    x_case = MMD_equal_case(x,device, b)
    print(x_case)
    y_case = MMD_equal_case(y,device,b)
    print(y_case)

    xy_case = MMD_mixed_case(x,y,device, b)
    print(xy_case)

    MMD = x_case + y_case - xy_case

    return MMD

#%%
y = torch.arange(5000).to(torch.float)
x = torch.arange(5000).to(torch.float)
x
test = MMD.calc_MMD_1d(x,y,device,1)
#%%
test.shape
#%%
test
#%%
test
#%%



#%%

#%%

#%%

#%%
#%%
#%%
#%%

