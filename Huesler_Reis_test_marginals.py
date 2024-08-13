
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
folder_name= "marginals_Huesler_Reiss"
lr = 0.01

#%%
# test the margins of a single realisation
# setup variogram

alpha = 1.
p = 1.

# create grids
grid = MMD.create_centered_grid(2)

y_sample_size = 10
#%%
Vario_true_params = MMD.Vario(torch.tensor([alpha]),torch.tensor([p]))
#%%
#sample the response
loc = 0.
y = MMD.sim_huesler_reis_ext(grid, Vario_true_params, device, no_simu=y_sample_size, loc=loc)

y_df = pd.DataFrame(y)
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








































# Example DataFrame and column (replace this with your actual data)
for col in range(y_df.shape[1]):
# Extract the data column you want to check
    data_column = 1/y_df[col]
    

    # Parameters for the Fréchet distribution
    alpha = 1  # shape parameter
    beta = 1   # scale parameter
    loc = 0     # location parameter

    # Create the theoretical Fréchet distribution
    x = np.linspace(min(data_column), max(data_column), 10000)
    exp_pdf = stats.expon.pdf(x)

   
    # Superimpose the theoretical Fréchet distribution
    plt.plot(data_column, exp_pdf, 'r-', lw=2)

    plt.xlim([-5, 10])
    # Add labels and legend
    plt.xlabel('1/X_i')
    plt.ylabel('Standard Exponential Density')
    plt.title(f'1/X_i vs. Standard exponential col {col}')


    file_name = f"1overX_i vs standard Expo Density col{col}.jpg"
    print(f"{folder_name}/{file_name}")
    plt.savefig(f"{folder_name}/{file_name}")
  
    plt.show()

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


# Example DataFrame and column (replace this with your actual data)
for col in range(y_df.shape[1]):
# Extract the data column you want to check
    data_column = y_df[col]

    # Parameters for the Fréchet distribution
    alpha = 1  # shape parameter
    beta = 1   # scale parameter
    loc = 0     # location parameter

    # Create the theoretical Fréchet distribution
    x = np.linspace(min(data_column), max(data_column), 1000)
    frechet_pdf = stats.invweibull.pdf(x, c=alpha, loc=loc, scale=beta)

    ## Plot the KDE of the data
    sns.kdeplot(data_column, color='g', label='Empirical Data')


    # Superimpose the theoretical Fréchet distribution
    plt.plot(x, frechet_pdf, 'r-', lw=2, label=f'Fréchet PDF\n(α={alpha}, β={beta}, loc={loc})')

    plt.xlim([-5, 10])
    # Add labels and legend
    plt.xlabel('Value')
    plt.ylabel('Density')
    plt.legend(loc='best')
    plt.title('Empirical Data vs. Fréchet Distribution')


    file_name = f"Hüsler_Reiss_simu_col{col}_ext_loc0_scaled.jpg"
    #plt.savefig(f"{folder_name}/{file_name}")
    plt.show()
y_df[0].mean()
#%%
# Example DataFrame and column (replace this with your actual data)
for col in range(y_df.shape[1]):
# Extract the data column you want to check
    data_column = y_df[col]

    # Parameters for the Fréchet distribution
    alpha = 1  # shape parameter
    beta = 1   # scale parameter
    loc = 0     # location parameter

    # Create the theoretical Fréchet distribution
    x = np.linspace(-5, 12, 10000)
    frechet_pdf = stats.invweibull.pdf(x, c=alpha, loc=loc, scale=beta)

    ## Plot the KDE of the data
    sns.kdeplot(data_column, color='g', label='Empirical Data')


    # Superimpose the theoretical Fréchet distribution
    plt.plot(x, frechet_pdf, 'r-', lw=2, label=f'Fréchet PDF\n(α={alpha}, β={beta}, loc={loc})')

    plt.xlim([-5, 10])
    # Add labels and legend
    plt.xlabel('Value')
    plt.ylabel('Density')
    plt.legend(loc='best')
    plt.title('Empirical Data vs. Fréchet Distribution')


    file_name = f"Hüsler_Reiss_simu_col{col}_ext_loc0_unscaled.jpg"
    plt.savefig(f"{folder_name}/{file_name}")
    plt.show()


#%%
###################################################################################


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

alpha = 1.
p = 1.

# create grids
grid = MMD.create_centered_grid(3)

y_sample_size = 100000
#%%
Vario_true_params = MMD.Vario(torch.tensor([alpha]),torch.tensor([p]))
#%%
# Parameters for the Fréchet distribution
shape = 1 # shape parameter
scale = 1   # scale parameter
loc = 0     # location parameter

# Create the theoretical Fréchet distribution
x = np.linspace(-5, 12, 10000)

#%%
frechet_rv = stats.invweibull.rvs(c=1, loc=1, scale=0, size=10000)

frechet_rv.mean()

#=> mean 1.0
#%%

frechet_rv.mean()
#%%

frechet_rv = stats.invweibull.rvs(c=1, loc=0, scale=1, size=100000)

frechet_rv.mean()


#%%

y_df.mean()
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