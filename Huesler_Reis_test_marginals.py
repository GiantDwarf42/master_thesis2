
#%%

import sys


#%%
import torch
import numpy as np
from datetime import datetime
import pandas as pd
import matplotlib.pyplot as plt
import scipy.stats as stats
import seaborn as sns

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

#%%


#%%
folder_name= "marginals_Huesler_Reiss"
lr = 0.01

#%%
# test the margins of a single realisation
# setup variogram

alpha = 5.
p = 1

# create grids
grid = MMD.create_centered_grid(3)

y_sample_size = 100000
#%%
Vario_true_params = MMD.Vario(torch.tensor([alpha]),torch.tensor([p]))
#%%
#sample the response
y = MMD.sim_huesler_reis_ext(grid, Vario_true_params, device, no_simu=y_sample_size)

y_df = pd.DataFrame(y)
#%%

# Example DataFrame and column (replace this with your actual data)
for col in range(y_df.shape[1]):
# Extract the data column you want to check
    data_column = y_df[col]

    # Parameters for the Fréchet distribution
    alpha = 1  # shape parameter
    beta = 1   # scale parameter
    mu = 1     # location parameter

    # Create the theoretical Fréchet distribution
    x = np.linspace(min(data_column), max(data_column), 10000)
    frechet_pdf = stats.invweibull.pdf(x, c=alpha, loc=mu, scale=beta)

    ## Plot the KDE of the data
    sns.kdeplot(data_column, color='g', label='Empirical Data')


    # Superimpose the theoretical Fréchet distribution
    plt.plot(x, frechet_pdf, 'r-', lw=2, label=f'Fréchet PDF\n(α={alpha}, β={beta}, μ={mu})')

    plt.xlim([-10, 200])
    # Add labels and legend
    plt.xlabel('Value')
    plt.ylabel('Density')
    plt.legend(loc='best')
    plt.title('Empirical Data vs. Fréchet Distribution')


    file_name = f"Hüsler_Reiss_simu_col{col}_ext.jpg"
    plt.savefig(f"{folder_name}/{file_name}")
    plt.show()

#%%

#%%
#%%
#%%
#sample the response
y = MMD.sim_huesler_reiss(grid, Vario_true_params, device, no_simu=y_sample_size)

y_df = pd.DataFrame(y)
#%%

# Example DataFrame and column (replace this with your actual data)
for col in range(y_df.shape[1]):
# Extract the data column you want to check
    data_column = y_df[col]

    # Parameters for the Fréchet distribution
    alpha = 1  # shape parameter
    beta = 1   # scale parameter
    mu = 1     # location parameter

    # Create the theoretical Fréchet distribution
    x = np.linspace(min(data_column), max(data_column), 10000)
    frechet_pdf = stats.invweibull.pdf(x, c=alpha, loc=mu, scale=beta)

    ## Plot the KDE of the data
    sns.kdeplot(data_column, color='g', label='Empirical Data')


    # Superimpose the theoretical Fréchet distribution
    plt.plot(x, frechet_pdf, 'r-', lw=2, label=f'Fréchet PDF\n(α={alpha}, β={beta}, μ={mu})')

    plt.xlim([-10, 200])
    # Add labels and legend
    plt.xlabel('Value')
    plt.ylabel('Density')
    plt.legend(loc='best')
    plt.title('Empirical Data vs. Fréchet Distribution')


    file_name = f"Hüsler_Reiss_simu_col{col}_spectral.jpg"
    plt.savefig(f"{folder_name}/{file_name}")
    plt.show()
#%%

y

# %%
