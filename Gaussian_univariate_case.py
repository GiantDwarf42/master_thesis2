#%%
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



#%%
seed = 42
torch.manual_seed(seed)
np.random.seed(seed)

#%%
######### 
# one dim case first generator

sample_size = 100
y_sample_size = 1000
nr_iterations = 300
lr = 0.1
nr_simulations = 1



mu = -10
sigma = 4


y = torch.normal(mu,sigma,(y_sample_size,1))

b_params = {"mu": 0,
            "sigma": 1}

# b heuristic could also just be a value
b = MMD.calc_b_heuristic(y, sample_size, "norm", device, b_params)

b_update = 100

folder_name= "Gaussian_Test"
file_name = f"mu{mu}sigma{sigma}xsize{sample_size}ysize{y_sample_size}bAUTO"


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

        #get MLE
    mu_hat_MLE, std_hat_MLE = norm.fit(y)
    #get Method of Moments Estimators
    mu_hat_MM = y.mean()
    std_hat_MM = y.std()

    estimators = pd.DataFrame({"mu_hat_MLE": np.repeat(mu_hat_MLE, nr_iterations),
                               "std_hat_MLE": np.repeat(std_hat_MLE, nr_iterations),
                               "mu_hat_MM": np.repeat(mu_hat_MM, nr_iterations),
                               "std_hat_MM": np.repeat(std_hat_MM, nr_iterations)})

    simulated_df = MMD.training_loop_gauss(mu_hat,sigma_hat, y, nr_iterations, sample_size, device, b, optimizer, b_update=b_update)


    simulated_df = pd.concat([simulated_df,estimators,simulation_index], axis=1)

    simulation_results.append(simulated_df)

    print(f"Simulation nr {sim}, done in {datetime.now()-now}")
    


#%%
stacked_results = pd.concat(simulation_results).reset_index(drop=True)


#%%
stacked_results.to_csv(f"{folder_name}/{file_name}.csv")
#%%

#%%
for index,df in enumerate(simulation_results):

    plt.plot(df[["mu_hat", "sigma_hat", "MMD"]])
    plt.axhline(y=mu)
    plt.axhline(y=sigma)
    plt.title(f"Simulation {file_name}")
    plt.legend(["mu_hat", "sigma_hat", "MMD"])
    plt.savefig(f"{folder_name}/{file_name}.svg")
    plt.show()
    

#%%
simulation_results
#%%

#%%
x_sample = [1,2,3,4,5]
y_sample = [1,2,3,4,5]



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


