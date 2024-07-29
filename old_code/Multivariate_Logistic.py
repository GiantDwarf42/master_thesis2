#%%
import torch
import numpy as np
from datetime import datetime
import pandas as pd
import matplotlib.pyplot as plt


#self written modules
import MMD

#%%
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
torch.set_default_dtype(torch.float64)

#%%
seed = 42
torch.manual_seed(seed)
np.random.seed(seed)

#%%

sample_size = 1000
y_sample_size = 1000
dims = 10
nr_iterations = 300
lr = 0.01
nr_simulations = 1
epoch_print_size = 100

alpha = 0.8

folder_name= "Logistic_Test"

y = MMD.sample_multivariate_logistic(y_sample_size,dims,alpha,device)

b_params = {"alpha": 0.5}

# b heuristic could also just be a value
b = MMD.calc_b_heuristic(y,sample_size,"logistic", device, b_params)

b_update = 100



folder_name= "Logistic_Test"
file_name = f"alpha{alpha}_b{b: .3f}_dims{dims}_xsize{sample_size}_y{y_sample_size}"
#file_name = f"alpha{alpha}_biter_dims{dims}_xsize{sample_size}_y{y_sample_size}"

simulation_results = []

#%%

#%%
for sim in np.arange(nr_simulations):
    now=datetime.now()

    alpha_hat = torch.tensor([0.5]).to(device).requires_grad_()

#####
# setup optimizer
    optimizer = torch.optim.Adam([alpha_hat], lr=lr)

    simulation_index = pd.DataFrame({"sim_nr":np.repeat(sim, nr_iterations)})
    


    simulated_df = MMD.training_loop_multi_logist(alpha_hat, y, nr_iterations, sample_size, device, b, optimizer, epoch_print_size, b_update)

    simulated_df = pd.concat([simulated_df,simulation_index], axis=1)

    simulation_results.append(simulated_df)

    print(f"Simulation nr {sim}, done in {datetime.now()-now}")
    


#%%
stacked_results = pd.concat(simulation_results).reset_index(drop=True)


#%%
stacked_results.to_csv(f"{folder_name}/{file_name}.csv")
#%%

#%%
for index,df in enumerate(simulation_results):

    plt.plot(df[["alpha_hat", "MMD"]])
    plt.axhline(y=alpha)
    plt.title(f"Simulation {file_name}")
    plt.legend(["alpha_hat", "MMD"])
    plt.savefig(f"{folder_name}/{file_name}.svg")
    plt.show()
    

#%%
stacked_results
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


