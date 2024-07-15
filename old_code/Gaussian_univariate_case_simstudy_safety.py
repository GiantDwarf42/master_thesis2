
import sys

def main(task_id):

    print("main is running")

    print(f"job ID exists: {task_id}")
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

    print(device)



    #%%
    seed = 42
    torch.manual_seed(seed)
    np.random.seed(seed)

    #%%

    #%%
    folder_name= "Gaussian_Test"
    lr = 0.1


    #######################################################################################################################3
    # setup parameter permutations

    # sample sizes
    # x_sample_sizes = [50,100,500,1000]
    # y_sample_sizes = [50,100,500,1000]

    # # true parameters
    # mu_list = [-40,-10,5,20]
    # sigma_list = [2,4,10,20]

    # # bandwidth
    # b_list = [0.001,0.01, "heuristic", "AUTO"]

    # # number number iterations
    # nr_iterations = 500


    # # nr simulations
    # nr_simulations = 150


    # sample sizes
    x_sample_sizes = [500]
    y_sample_sizes = [100]

    # true parameters
    mu_list = [5, -1]
    sigma_list = [10]

    # bandwidth
    b_list = [0.01, "heuristic", "AUTO"]

    # number number iterations
    nr_iterations = 1500


    # nr simulations
    nr_simulations = 2


    #%%



    for b in b_list:

        for mu in mu_list:

            for sigma in sigma_list:

                for x_sample_size in x_sample_sizes:

                    for y_sample_size in y_sample_sizes:
                        
                        #setting up the results list
                        simulation_results = []

                        if b == "heuristic":

                            file_name = f"mu{mu}sigma{sigma}xsize{x_sample_size}ysize{y_sample_size}bheuristic"

                        elif b == "AUTO":
                            file_name = f"mu{mu}sigma{sigma}xsize{x_sample_size}ysize{y_sample_size}bAUTO"

                        else:

                            file_name = f"mu{mu}sigma{sigma}xsize{x_sample_size}ysize{y_sample_size}b{b}"

                        for nr_simulation in range(nr_simulations):
                            
                            
                            y = torch.normal(mu,sigma,(y_sample_size,1))

                            # check if b heuristic needs to be calculated
                            if b == "heuristic" or b == "AUTO":

                                # starting point for bandwidth 
                                b_params = {"mu": 0,
                                        "sigma": 1}

                                # b heuristic could also just be a value
                                b = MMD.calc_b_heuristic(y, x_sample_size, "norm", device, b_params).item()

                            #check if b needs to be iteratively updated
                            if b == "AUTO":
                                b_update = 100
                            else:
                                b_update = False
                            
                            

                            #take start time
                            now=datetime.now()

                            #setup the parameters to optimize
                            mu_hat = torch.tensor([0.]).to(device).requires_grad_()
                            sigma_hat = torch.tensor([1.]).to(device).requires_grad_()
                            
                            # setup optimizer
                            optimizer = torch.optim.Adam([mu_hat,sigma_hat], lr=lr)

                            #get the simulation number
                            simulation_index = pd.DataFrame({"sim_nr":np.repeat(nr_simulation, nr_iterations)})

                            #get MLE
                            mu_hat_MLE, std_hat_MLE = norm.fit(y)
                            #get Method of Moments Estimators
                            mu_hat_MM = y.mean()
                            std_hat_MM = y.std()

                            #safe MLE and MM estimators
                            estimators = pd.DataFrame({"mu_hat_MLE": np.repeat(mu_hat_MLE, nr_iterations),
                                                    "std_hat_MLE": np.repeat(std_hat_MLE, nr_iterations),
                                                    "mu_hat_MM": np.repeat(mu_hat_MM, nr_iterations),
                                                    "std_hat_MM": np.repeat(std_hat_MM, nr_iterations)})

                            
                            # run the actual simulation
                            simulated_df = MMD.training_loop_gauss(mu_hat,sigma_hat, y, nr_iterations, x_sample_size, device, b, optimizer, epoch_print_size=False,b_update=b_update)

                            # get the simulation as a data frame
                            simulated_df = pd.concat([simulated_df,estimators,simulation_index], axis=1)

                            simulation_results.append(simulated_df)



                        stacked_results = pd.concat(simulation_results).reset_index(drop=True)

                        stacked_results.to_csv(f"{folder_name}/{file_name}_taskID_{task_id}.csv")

if __name__ == "__main__":
    if len(sys.argv) != 2:
        print("Usage: python my_calculation.py <task_id>")
        sys.exit(1)
    
    task_id = sys.argv[1]
    main(task_id)



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

print(device)



#%%
seed = 42
torch.manual_seed(seed)
np.random.seed(seed)

#%%

#%%
folder_name= "Gaussian_Test"
lr = 0.1


#######################################################################################################################3
# setup parameter permutations

# sample sizes
# x_sample_sizes = [50,100,500,1000]
# y_sample_sizes = [50,100,500,1000]

# # true parameters
# mu_list = [-40,-10,5,20]
# sigma_list = [2,4,10,20]

# # bandwidth
# b_list = [0.001,0.01, "heuristic", "AUTO"]

# # number number iterations
# nr_iterations = 500


# # nr simulations
# nr_simulations = 150


# sample sizes
x_sample_sizes = [500]
y_sample_sizes = [100]

# true parameters
mu_list = [5, -1]
sigma_list = [10]

# bandwidth
b_list = [0.01, "heuristic", "AUTO"]

# number number iterations
nr_iterations = 500


# nr simulations
nr_simulations = 2


#%%



for b in b_list:

    for mu in mu_list:

        for sigma in sigma_list:

            for x_sample_size in x_sample_sizes:

                for y_sample_size in y_sample_sizes:
                    
                    #setting up the results list
                    simulation_results = []

                    if b == "heuristic":

                        file_name = f"mu{mu}sigma{sigma}xsize{x_sample_size}ysize{y_sample_size}bheuristic"

                    elif b == "AUTO":
                        file_name = f"mu{mu}sigma{sigma}xsize{x_sample_size}ysize{y_sample_size}bAUTO"

                    else:

                        file_name = f"mu{mu}sigma{sigma}xsize{x_sample_size}ysize{y_sample_size}b{b}"

                    for nr_simulation in range(nr_simulations):
                        
                        
                        y = torch.normal(mu,sigma,(y_sample_size,1))

                        # check if b heuristic needs to be calculated
                        if b == "heuristic" or b == "AUTO":

                            # starting point for bandwidth 
                            b_params = {"mu": 0,
                                    "sigma": 1}

                            # b heuristic could also just be a value
                            b = MMD.calc_b_heuristic(y, x_sample_size, "norm", device, b_params).item()

                        #check if b needs to be iteratively updated
                        if b == "AUTO":
                            b_update = 100
                        else:
                            b_update = False
                        
                        

                        #take start time
                        now=datetime.now()

                        #setup the parameters to optimize
                        mu_hat = torch.tensor([0.]).to(device).requires_grad_()
                        sigma_hat = torch.tensor([1.]).to(device).requires_grad_()
                        
                        # setup optimizer
                        optimizer = torch.optim.Adam([mu_hat,sigma_hat], lr=lr)

                        #get the simulation number
                        simulation_index = pd.DataFrame({"sim_nr":np.repeat(nr_simulation, nr_iterations)})

                        #get MLE
                        mu_hat_MLE, std_hat_MLE = norm.fit(y)
                        #get Method of Moments Estimators
                        mu_hat_MM = y.mean()
                        std_hat_MM = y.std()

                        #safe MLE and MM estimators
                        estimators = pd.DataFrame({"mu_hat_MLE": np.repeat(mu_hat_MLE, nr_iterations),
                                                "std_hat_MLE": np.repeat(std_hat_MLE, nr_iterations),
                                                "mu_hat_MM": np.repeat(mu_hat_MM, nr_iterations),
                                                "std_hat_MM": np.repeat(std_hat_MM, nr_iterations)})

                        
                        # run the actual simulation
                        simulated_df = MMD.training_loop_gauss(mu_hat,sigma_hat, y, nr_iterations, x_sample_size, device, b, optimizer, epoch_print_size=False,b_update=b_update)

                        # get the simulation as a data frame
                        simulated_df = pd.concat([simulated_df,estimators,simulation_index], axis=1)

                        simulation_results.append(simulated_df)



                    stacked_results = pd.concat(simulation_results).reset_index(drop=True)

                    stacked_results.to_csv(f"{folder_name}/{file_name}.csv")

                

                        
                        



#%%

#%%
#%%
#%%

#%%

#%%

#%%
#%%
#%%

# %%
