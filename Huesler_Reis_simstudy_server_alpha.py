
#%%

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
    torch.set_default_dtype(torch.float64)


    #%%
    # this is necessary to not just get the same values 150 times
    seed = int(task_id)
    torch.manual_seed(seed)
    np.random.seed(seed)

    #%%


    #%%
    folder_name= "Huesler_Reiss_output_alpha"
    lr = 0.1

    #%%
    #######################################################################################################################3
    # setup parameter permutations

    # create grids
    #grids = [MMD.create_centered_grid(3), MMD.create_centered_grid(5), MMD.create_centered_grid(10)]

    grids = [
    {"grid":MMD.create_centered_grid(2),
     "name": "2x2"}]


    # #sample sizes
    x_sample_sizes = [1000]
    y_sample_sizes = [1000]


    # true parameters

    alpha_list = [2., 5., 10.]
    p_list = [1.]

    # bandwidth
    b_list = ["AUTO"]

    # number number iterations
    nr_iterations = 2500

    b_update=100

    for b_value in b_list:

        for grid in grids:

            for alpha in alpha_list:

                for p in p_list:
                    
                    for x_sample_size in x_sample_sizes:

                        for y_sample_size in y_sample_sizes:
                        
                            if x_sample_size >= y_sample_size:

                                grid_name = grid["name"]
                                grid_content = grid["grid"]
                                #grid_size = np.sqrt(grid.shape[0])

                                file_name = f"grid{grid_name}_alpha{alpha}_p{p}_xsize{x_sample_size}_ysize{y_sample_size}_ID{task_id}"

                                # setup variogram
                                Vario_true_params = MMD.Vario(torch.tensor([alpha]),torch.tensor([p]))

                                #sample the response
                                y = MMD.sim_huesler_reis_ext(grid_content, Vario_true_params, device, no_simu=y_sample_size)


                                
                                # check if b heuristic needs to be calculated
                                if b_value == "AUTO":

                                # setting up initial b_params 
                                    b_params = {"Vario": Vario_true_params,
                                                "grid": grid_content}

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
                                    p_hat = torch.tensor([1.]).to(device)
                            
                                    # setup optimizer
                                    optimizer = torch.optim.Adam([alpha_hat], lr=lr)
                                    
                                    #setup Vario object to optimize
                                    Vario = MMD.Vario(alpha_hat,p_hat)

                                    

                                    simulated_df = MMD.training_loop_huesler_reis(Vario, y, grid_content, nr_iterations , x_sample_size, device, b, optimizer, epoch_print_size=False, b_update=b_update)


                                    
                            

                                    simulated_df.to_csv(f"{folder_name}/{file_name}.csv")


                                else:
                                    continue

if __name__ == "__main__":
    if len(sys.argv) != 2:
        sys.exit(1)
    
    task_id = sys.argv[1]
    main(task_id)
                                    

