
import sys

def main(task_id):

    print("main is running")

    print(f"job ID exists: {task_id}")
    #%%
    import torch
    import numpy as np
    from datetime import datetime
    import pandas as pd
    from scipy.stats import norm

    #self written modules
    import MMD

    #%%
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")




    #%%
    # this is necessary to not just get the same values 150 times
    seed = int(task_id)
    torch.manual_seed(seed)
    np.random.seed(seed)

    #%%

    #%%
    folder_name= "Gaussian_Result"
    lr = 0.1


    #######################################################################################################################3
    # setup parameter permutations

    # #sample sizes
    x_sample_sizes = [50,100,200,500,1000]
    y_sample_sizes = [50,100,200,500,1000]

    # true parameters
    mu_list = [-3, 50]
    sigma_list = [2,40]

    # bandwidth
    b_list = ["AUTO", 0.01, 0.1]

    # number number iterations
    nr_iterations = 2000

    
    # # #sample sizes
    # x_sample_sizes = [1000]
    # y_sample_sizes = [1000]

    # # true parameters
    # mu_list = [50]
    # sigma_list = [40]

    # # bandwidth
    # b_list = ["AUTO", 0.01, 0.1]

    # # number number iterations
    # nr_iterations = 500


   




    #%%



    for b_value in b_list:

        for mu in mu_list:

            for sigma in sigma_list:

                for x_sample_size in x_sample_sizes:

                    for y_sample_size in y_sample_sizes:
                        
                        if x_sample_size >= y_sample_size:
                        
                            


                            if b_value == "AUTO":
                                file_name = f"mu{mu}_sigma{sigma}_xsize{x_sample_size}_ysize{y_sample_size}_bAUTO"

                            else:

                                file_name = f"mu{mu}_sigma{sigma}_xsize{x_sample_size}_ysize{y_sample_size}_b{b_value}"

                            

                            y = torch.normal(mu,sigma,(y_sample_size,1))

                            # check if b heuristic needs to be calculated
                            if b_value == "AUTO":

                                # starting point for bandwidth 
                                b_params = {"mu": 0,
                                        "sigma": 1}

                                # b heuristic could also just be a value
                                b = MMD.calc_b_heuristic(y, x_sample_size, "norm", device, b_params).item()

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
                            mu_hat = torch.tensor([0.]).to(device).requires_grad_()
                            sigma_hat = torch.tensor([1.]).to(device).requires_grad_()
                            
                            # setup optimizer
                            optimizer = torch.optim.Adam([mu_hat,sigma_hat], lr=lr)


                            #get MLE
                            mu_hat_MLE, std_hat_MLE = norm.fit(y)
                            

                            #safe MLE and MM estimators
                            estimators = pd.DataFrame({"mu_hat_MLE": np.repeat(mu_hat_MLE, nr_iterations),
                                                    "sigma_hat_MLE": np.repeat(std_hat_MLE, nr_iterations)})

                            
                            # run the actual simulation
                            simulated_df = MMD.training_loop_gauss(mu_hat,sigma_hat, y, nr_iterations, x_sample_size, device, b, optimizer, epoch_print_size=False,b_update=b_update)

                            # get the simulation as a data frame
                            simulated_df = pd.concat([simulated_df,estimators], axis=1)

                            

                            simulated_df.to_csv(f"{folder_name}/{file_name}_ID{task_id}.csv")
                        
                        else:
                            continue
                        
if __name__ == "__main__":
    if len(sys.argv) != 2:
        print("Usage: python my_calculation.py <task_id>")
        sys.exit(1)
    
    task_id = sys.argv[1]
    main(task_id)



