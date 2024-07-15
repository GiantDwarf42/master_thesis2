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
######### 
# one dim case first generator

sample_size = 100
nr_itterations = 10000
bandwidth_range = [50]

mu = -2
sigma = 3



mu_hat = torch.tensor([0.]).to(device).requires_grad_()
mu_hat

sigma_hat = torch.tensor([1.]).to(device).requires_grad_()
sigma_hat


#####
# setup optimizer
optimizer = torch.optim.Adam([mu_hat,sigma_hat])

#%%
####
# setup generator function
x_1 = mu_hat + sigma_hat*torch.normal(0,1,(sample_size,1))


#######
#getting the response (fixed)
y_1 = torch.normal(mu,sigma,(sample_size,1))


#%%

################# Training loop ##########################
mu_hat_estimates = []
sigma_hat_estimates = []
MMD_values = []
now=datetime.now()
for epoch in np.arange(nr_itterations):

    # Sample from the generator
    sample = torch.normal(0,1,(sample_size,1))
    sample = mu_hat + sigma_hat*sample

    # Empty gradient
    optimizer.zero_grad()
    
    # Calculate Loss
    loss = MMD.MMD_gaussian_kaggle(sample, y_1, device, bandwidth_range=bandwidth_range)
    
    # Calculate gradient
    loss.backward()
    
    # Take one SGD step
    optimizer.step()

    mu_hat_estimates.append(mu_hat.detach().clone())
    sigma_hat_estimates.append(sigma_hat.detach().clone())
    MMD_values.append(loss.detach().clone())


    

    if epoch%500==0:
        print("epoch: ",epoch," loss=",loss)
        
################# end of training loop ##########################
print("Training time: ",datetime.now()-now)
#%%
optimizer.param_groups

#%%
mu_hat_estimates = torch.stack(mu_hat_estimates).detach().numpy().reshape([nr_itterations,])
sigma_hat_estimates = torch.stack(sigma_hat_estimates).detach().numpy().reshape([nr_itterations,])
MMD_values = torch.stack(MMD_values).detach().numpy()
#%%

df_results = pd.DataFrame({"mu_hat": mu_hat_estimates, 
              "sigma_hat": sigma_hat_estimates,
              "MMD": MMD_values})

#%%
df_results.plot()
#%%

#%%
plt.plot(df_results["mu_hat"])
plt.axhline(y=mu)
plt.show()
#%%
#%%
plt.plot(df_results["sigma_hat"])
plt.axhline(y=sigma)
plt.show()
#%%
df_results[df_results["MMD"] == df_results["MMD"].min()]
#%%
df_results.sort_values(by="MMD").head(20)
#%%
df_results.sort_values(by="MMD").head(20).mean()
#%%
plt.plot(df_results["MMD"])

plt.show()
#%%
#%%

################# Training loop ##########################
mu_hat_estimates = []
sigma_hat_estimates = []
MMD_values = []
now=datetime.now()
for epoch in np.arange(nr_itterations):

    # Sample from the generator
    sample = torch.normal(0,1,(sample_size,1))
    sample = mu_hat + sigma_hat*sample

    # Empty gradient
    optimizer.zero_grad()
    
    # Calculate Loss
    loss = MMD.MMD_homebrew_1d(sample, y_1, device)
    
    # Calculate gradient
    loss.backward()
    
    # Take one SGD step
    optimizer.step()

    mu_hat_estimates.append(mu_hat.detach().clone())
    sigma_hat_estimates.append(sigma_hat.detach().clone())
    MMD_values.append(loss.detach().clone())


    

    if epoch%500==0:
        print("epoch: ",epoch," loss=",loss)
        
################# end of training loop ##########################
print("Training time: ",datetime.now()-now)
#%%
sample = torch.normal(0,1,(sample_size,1))
sample = mu_hat + sigma_hat*sample

sample.shape

#%%
#%%
optimizer.param_groups

#%%
mu_hat_estimates = torch.stack(mu_hat_estimates).detach().numpy().reshape([nr_itterations,])
sigma_hat_estimates = torch.stack(sigma_hat_estimates).detach().numpy().reshape([nr_itterations,])
MMD_values = torch.stack(MMD_values).detach().numpy()
#%%

df_results = pd.DataFrame({"mu_hat": mu_hat_estimates, 
              "sigma_hat": sigma_hat_estimates,
              "MMD": MMD_values})

#%%
df_results.plot()
#%%

#%%
plt.plot(df_results["mu_hat"])
plt.axhline(y=mu)
plt.show()
#%%
#%%
plt.plot(df_results["sigma_hat"])
plt.axhline(y=sigma)
plt.show()
#%%
df_results[df_results["MMD"] == df_results["MMD"].min()]
#%%
df_results.sort_values(by="MMD").head(20).mean()
#%%
plt.plot(df_results["MMD"])

plt.show()
#%%
#%%
#%%
#%%
#%%
import torch

# Define the size of the vectors
n = 5
m = 5

# Create vectors x and y
x = torch.randn(n)
y = torch.randn(m)

# Define the function k
def k(x, y, bandwidth=10):

    res = torch.exp(-0.5 * (x-y)**2/bandwidth)

    return res

# Expand x and y to enable broadcasting
x_expanded = x.unsqueeze(1).expand(-1, m)  # Shape: [n, m]
y_expanded = y.unsqueeze(0).expand(n, -1)  # Shape: [n, m]

# Compute the element-wise function k for all pairs (i, j)
k_values = k(x_expanded, y_expanded)

# Create a mask to exclude the cases where j == i
mask = torch.ones(n, m, dtype=bool)
mask.fill_diagonal_(False)

# Apply the mask
masked_k_values = k_values[mask]

# Compute the sum
result = masked_k_values.sum()

print(f'The result of the vectorized sum is: {result}')
#%%
def kernel_gauss(x, y, bandwidth=10):

    res = torch.exp(-0.5 * (x-y)**2/bandwidth)

    return res

#%%



def MMD_homebrew_1d(x,y,bandwith=10):

    n = x.shape[0]
    m = y.shape[0]
    # Expand x and y to enable broadcasting
    x_expanded = x.unsqueeze(1).expand(-1, m)  # Shape: [n, m]
    y_expanded = y.unsqueeze(0).expand(n, -1)  # Shape: [n, m]

    # Compute the element-wise function k for all pairs (i, j)
    k_values = kernel_gauss(x_expanded, y_expanded)

    # Create a mask to exclude the cases where j == i
    mask = torch.ones(n, m, dtype=bool)
    mask.fill_diagonal_(False)

    # Apply the mask
    masked_k_values = k_values[mask]

    # Compute the sum
    result = masked_k_values.sum()

    return result
#%%



#%%
#%%
#%%

######################################################################
#test setup stuff 2dim

m = 200 # sample size
x_mean = torch.zeros(4)+1
y_mean = torch.zeros(4)
x_cov = 10*torch.eye(4) # IMPORTANT: Covariance matrices must be positive definite
y_cov = 30*torch.eye(4) - 1

px = MultivariateNormal(x_mean, x_cov)
qy = MultivariateNormal(y_mean, y_cov)
x = px.sample([m]).to(device)
y = qy.sample([m]).to(device)
#%%
x_cov
#%%

import numpy as np

from scipy.stats import multivariate_normal
from scipy.stats import dirichlet 
from torch.distributions.multivariate_normal import MultivariateNormal 



m = 20 # sample size


#response params
y_mean = torch.zeros(2)+3
y_cov = 3*torch.eye(2) - 1
qy = MultivariateNormal(y_mean, y_cov)

y = qy.sample([m]).to(device)
# getting response

#%%

# initializing generator function
x_mean = torch.zeros(2)
x_mean = x_mean.requires_grad_()

x_cov = torch.eye(2)
x_cov = x_cov.requires_grad_() # IMPORTANT: Covariance matrices must be positive definite

#%%
#sim dist
sim_mean = torch.zeros(2)
sim_cov = torch.eye(2)

px = MultivariateNormal(sim_mean, sim_cov)


#%%
x = px.sample([m]).to(device)

#x = x_mean + torch.mm(x_cov, x.t())

y = qy.sample([m]).to(device)

#%%
loss = MMD.MMD_gaussian_kaggle(x,y,device)

loss
#%%
result = MMD.gaus(x, y, kernel="multiscale")
result = MMD(x, x, kernel="multiscale")

print(f"MMD result of X and Y is {result.item()}")
#%%
x
#%%
#%%

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

def MMD(x, y, kernel):
    """Emprical maximum mean discrepancy. The lower the result
       the more evidence that distributions are the same.

    Args:
        x: first sample, distribution P
        y: second sample, distribution Q
        kernel: kernel type such as "multiscale" or "rbf"
    """
    xx, yy, zz = torch.mm(x, x.t()), torch.mm(y, y.t()), torch.mm(x, y.t())
    rx = (xx.diag().unsqueeze(0).expand_as(xx))
    ry = (yy.diag().unsqueeze(0).expand_as(yy))
    
    dxx = rx.t() + rx - 2. * xx # Used for A in (1)
    dyy = ry.t() + ry - 2. * yy # Used for B in (1)
    dxy = rx.t() + ry - 2. * zz # Used for C in (1)
    
    XX, YY, XY = (torch.zeros(xx.shape).to(device),
                  torch.zeros(xx.shape).to(device),
                  torch.zeros(xx.shape).to(device))
    
    if kernel == "multiscale":
        
        bandwidth_range = [0.2, 0.5, 0.9, 1.3]
        for a in bandwidth_range:
            XX += a**2 * (a**2 + dxx)**-1
            YY += a**2 * (a**2 + dyy)**-1
            XY += a**2 * (a**2 + dxy)**-1
            
    if kernel == "rbf":
      
        bandwidth_range = [10, 15, 20, 50]
        for a in bandwidth_range:
            XX += torch.exp(-0.5*dxx/a)
            YY += torch.exp(-0.5*dyy/a)
            XY += torch.exp(-0.5*dxy/a)
      
      

    return torch.mean(XX + YY - 2. * XY)
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
#%%