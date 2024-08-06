
#%%
import sys



print("main is running")

#print(f"job ID exists: {task_id}")
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
torch.set_default_dtype(torch.float64)


task_id = 42
#%%
# this is necessary to not just get the same values 150 times
seed = int(task_id)
torch.manual_seed(seed)
np.random.seed(seed)

#%%

grid_size = 4
grids = [MMD.create_centered_grid(i) for i in range(2, grid_size+1)]
#%%
#%%
folder_name= "Huesler_Reis_output"
lr = 0.01

#%%
#######################################################################################################################3
# setup parameter permutations

# create grids
grid_size = 4
#grids = [MMD.create_centered_grid(i) for i in range(2, grid_size+1)]
grids = [MMD.create_centered_grid(4)]


# #sample sizes
# x_sample_sizes = [50,100,200,500,1000]
# y_sample_sizes = [50,100,200,500,1000]

x_sample_sizes = [100]
y_sample_sizes = [100]

# true parameters
#alpha_list = [0.1, 0.3, 0.7, 0.9]
#p_list = [1.,2.,10.,1000.]

alpha_list = [3.]
p_list = [1.5]

# bandwidth
b_list = ["AUTO"]

# number number iterations
nr_iterations = 1000

b_update=100

for b_value in b_list:

    for grid in grids:

        for alpha in alpha_list:

            for p in p_list:
                 
                for x_sample_size in x_sample_sizes:

                    for y_sample_size in y_sample_sizes:
                    
                        if x_sample_size >= y_sample_size:

                            
                            grid_size = np.sqrt(grid.shape[0])

                            file_name = f"grid{grid_size}_alpha{alpha}_p{p}xsize{x_sample_size}_ysize{y_sample_size}_ID{task_id}"

                            # setup variogram
                            Vario_true_params = MMD.Vario(torch.tensor([alpha]),torch.tensor([p]))

                            #sample the response
                            y = MMD.sim_huesler_reis_ext(grid, Vario_true_params, device, no_simu=y_sample_size)


                            
                            # check if b heuristic needs to be calculated
                            if b_value == "AUTO":

                            # setting up initial b_params 
                                b_params = {"Vario": Vario_true_params,
                                            "grid": grid}

                            # b heuristic could also just be a value
                                b = MMD.calc_b_heuristic(y, x_sample_size, "huesler_reis", device, b_params).item()

                                #check if b needs to be iteratively updated
                                if b_value == "AUTO":
                                    b_update = 100
                                else:
                                    b_update = False


                                #take start time
                                now=datetime.now()

                                

                                #setup the parameters to optimize
                                alpha_hat = torch.tensor([1.]).to(device).requires_grad_()
                                p_hat = torch.tensor([1.]).to(device).requires_grad_()
                        
                                # setup optimizer
                                optimizer = torch.optim.Adam([alpha_hat, p_hat], lr=lr)
                                
                                #setup Vario object to optimize
                                Vario = MMD.Vario(alpha_hat,p_hat)

                                

                                simulated_df = MMD.training_loop_huesler_reis(Vario, y, grid, nr_iterations , x_sample_size, device, b, optimizer, epoch_print_size=50, b_update=b_update)


                                
                        

                                simulated_df.to_csv(f"{folder_name}/{file_name}.csv")

                                break


                            else:
                                continue



                                

                            
                        break
                    break
                break
    
            break
        break
    break  


#%%
simulated_df
#%%
print(f"{folder_name}/{file_name}.csv")
#%%
simulated_df
#%%
#%%
simulated_df
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

#going through parameter permutation

for b_value in b_list:

    for alpha in alpha_list:

        for dim in dim_list:

            for x_sample_size in x_sample_sizes:

                for y_sample_size in y_sample_sizes:
                    
                    if x_sample_size >= y_sample_size:
                    
                        
                        if b_value == "AUTO":
                            
                            file_name = f"alpha{alpha}_dim{dim}_bAUTO_xsize{x_sample_size}_ysize{y_sample_size}_ID{task_id}"
                        

                        elif isinstance(b_value, float):
                            
                            file_name = f"alpha{alpha}_dim{dim}_b{b_value}_xsize{x_sample_size}_ysize{y_sample_size}_ID{task_id}"


                        

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

                        

                        simulated_df.to_csv(f"{folder_name}/{file_name}.csv")
                    
                    else:
                        continue






























# #sample sizes
x_sample_sizes = [50,100,200,500,1000]
y_sample_sizes = [50,100,200,500,1000]




# true parameters
alpha_list = [0.1, 0.3, 0.7, 0.9]

#dims
dim_list = [1,2,5,10,20]

# bandwidth
b_list = ["AUTO", 0.01, 0.1]

# number number iterations
nr_iterations = 2500




#going through parameter permutation

for b_value in b_list:

    for alpha in alpha_list:

        for dim in dim_list:

            for x_sample_size in x_sample_sizes:

                for y_sample_size in y_sample_sizes:
                    
                    if x_sample_size >= y_sample_size:
                    
                        
                        if b_value == "AUTO":
                            
                            file_name = f"alpha{alpha}_dim{dim}_bAUTO_xsize{x_sample_size}_ysize{y_sample_size}_ID{task_id}"
                        

                        elif isinstance(b_value, float):
                            
                            file_name = f"alpha{alpha}_dim{dim}_b{b_value}_xsize{x_sample_size}_ysize{y_sample_size}_ID{task_id}"


                        

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

                        

                        simulated_df.to_csv(f"{folder_name}/{file_name}.csv")
                    
                    else:
                        continue
                    



