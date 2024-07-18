
import torch
import numpy as np
import pandas as pd
import time


def kernel_gauss_cdist(cdist:torch.tensor, b:int=1)->torch.tensor:
    """calculates the tensor (matrix) of gaussian kernel values calculated using cdist

    Args:
        cdist (torch.tensor): a 2d tensor of cdist output values
        b (int, optional): the bandwidth. Defaults to 1.

    Returns:
        torch.tensor: the resulting gaussian kernel value tensor 
    """

    res = torch.exp(-b*((cdist)**2))

    return res


def calc_b_heuristic(y:torch.tensor, n:int, case:str, device, params:dict)->torch.tensor:
    """calculates the median heuristic for b

    Args:
        y (torch.tensor): the target distribution tensor
        n (int): the number of rows the simulation distribution should
        case (str): the distribution for which to calculate b for cases: "norm", "logistic"
        device (_type_): the pytorch device object
        params (dict): a correctly named dictionary with the required parameters for the b calculation if "norm" => mu, sigma if "logistic" alpha
    Returns:
        torch.tensor: returns the bandwidth value as a 1x1 tensor
    """


    #case distinction
    if case == "norm":
        
        #this is a safety thing, during optimization it can incidentially happen
        # that sigma_hat becomes negative. It is very rare but it does.
        if params["sigma"] <= 0:
            params["sigma"] = 0.01
        
        x = torch.normal(params["mu"], params["sigma"],(n,y.shape[1]))
        b = 1/ torch.median(torch.cdist(x,y,p=2)**2)

    elif case == "logistic":
            
            #safety thing as above, might not be needed
            if params["alpha"] <= 0:
                params["alpha"] = 0.01
            elif params["alpha"] >= 1:
                params["alpha"] = 0.99

            x = sample_multivariate_logistic(n,y.shape[1], params["alpha"], device)
            b = 1/ torch.median(torch.cdist(x,y,p=2)**2)

    return b

def training_loop_gauss(mu_hat, sigma_hat,target_dist, nr_iterations , sample_size, device, b, optimizer, epoch_print_size=500, b_update=0)->pd.DataFrame:
    """workhorse function which does the actual gradient optimization in 
    the Gaussian case. This function is called for each simulation in a loop

    Args:
        mu_hat (_type_): The tensor for the mu parameter
        sigma_hat (_type_): the tensor for the sigma parameter
        target_dist (_type_): the target distribution vector
        nr_iterations (_type_): the nr of itterations per simulation (epochs)
        sample_size (_type_): the sample size for the simulation distribution
        device (_type_): the torch device object
        b (_type_): the bandwith parameter b
        optimizer (_type_): the setup optimizer object
        epoch_print_size (int, optional): the number of iterations after which a progress print should be made. Defaults to 500.
        b_update (int, optional) : the bandwith value will be adjusted every given number of iterations

    Returns:
        pd.DataFrame: the df with the results
    """
    mu_hat_estimates = []
    sigma_hat_estimates = []
    MMD_values = []
    b_values = []
    times = []

    start_time = time.time()

    #the MMD yy case is a constant.
    # no need to recalculate this for every epoch
    MMD_yy_case = MMD_equal_case(target_dist,device,b)
    
    for epoch in np.arange(nr_iterations):


        #iterative update of bandwidth value
        if b_update:

            if epoch % b_update == 0:

                params = {"mu": mu_hat.item(),
                          "sigma": sigma_hat.item()}
                
                
                b = calc_b_heuristic(target_dist, sample_size, "norm", device, params).item()

                MMD_yy_case = MMD_equal_case(target_dist,device,b)

        # Empty gradient
        optimizer.zero_grad()

        # Sample from the generator
        sample = torch.normal(0,1,(sample_size,target_dist.shape[1]))
        sample = mu_hat + sigma_hat*sample

        # Calculate Loss

        # sample case
        MMD_xx_case = MMD_equal_case(sample, device, b)
        MMD_xy_case = MMD_mixed_case(sample,target_dist,device, b)
        #loss
        loss = MMD_xx_case + MMD_yy_case - MMD_xy_case

        
        #optimizer.zero_grad()
        # Calculate gradient
        loss.backward()
        
        # Take one SGD step
        optimizer.step()

        mu_hat_estimates.append(mu_hat.detach().clone())
        sigma_hat_estimates.append(sigma_hat.detach().clone())
        MMD_values.append(loss.detach().clone())
        b_values.append(b)
        times.append(time.time()-start_time)


        
        if epoch_print_size:
            if epoch%epoch_print_size==0:
                print("epoch: ",epoch," loss=",loss)
            
    mu_hat_estimates = torch.stack(mu_hat_estimates).detach().numpy().reshape([nr_iterations,])
    sigma_hat_estimates = torch.stack(sigma_hat_estimates).detach().numpy().reshape([nr_iterations,])
    MMD_values = torch.stack(MMD_values).detach().numpy()

    df_results = pd.DataFrame({"mu_hat": mu_hat_estimates, 
              "sigma_hat": sigma_hat_estimates,
              "MMD": MMD_values,
              "b": b_values,
              "time": times})
    
    return df_results

def training_loop_multi_logist(alpha_hat, target_dist, nr_iterations , sample_size, device, b, optimizer, epoch_print_size=500, b_update=0):
   
    alpha_hat_estimates = []
    MMD_values = []
    b_values = []

    times = []
    start_time = time.time()

    #the MMD yy case is a constant.
    # no need to recalculate this for every epoch
    MMD_yy_case = MMD_equal_case(target_dist,device,b)
    
    for epoch in np.arange(nr_iterations):

        #iterative update of bandwidth value
        if b_update:

            if epoch % b_update == 0:

                params = {"alpha": alpha_hat.item()}
                
                
                b = calc_b_heuristic(target_dist, sample_size, "logistic", device, params).item()

                MMD_yy_case = MMD_equal_case(target_dist,device,b)


        # Empty gradient
        optimizer.zero_grad()

        # Sample from the generator
        sample = sample_multivariate_logistic(sample_size, target_dist.shape[1], alpha_hat, device)
        # Calculate Loss

        # sample case
        MMD_xx_case = MMD_equal_case(sample, device, b)
        MMD_xy_case = MMD_mixed_case(sample,target_dist,device, b)
        #loss
        loss = MMD_xx_case + MMD_yy_case - MMD_xy_case

        
        #optimizer.zero_grad()
        # Calculate gradient
        loss.backward()
        
        # Take one SGD step
        optimizer.step()

        alpha_hat_estimates.append(alpha_hat.detach().clone())
        MMD_values.append(loss.detach().clone())
        b_values.append(b)
        times.append(time.time()-start_time)


        

        if epoch_print_size:
            if epoch%epoch_print_size==0:
                print("epoch: ",epoch," loss=",loss)
            
    alpha_hat_estimates = torch.stack(alpha_hat_estimates).detach().numpy().reshape([nr_iterations,])
    MMD_values = torch.stack(MMD_values).detach().numpy()

    df_results = pd.DataFrame({"alpha_hat": alpha_hat_estimates,
              "MMD": MMD_values,
              "b": b_values,
              "time": times})
    
    return df_results


def MMD_mixed_case(x:torch.tensor,y:torch.tensor,device,b=1):

    """calculates the MMD between the sample and the target distribution 

    Args:
        x (torch.tensor): the current sample distribution 
        y (torch.tensor): the target distribution which we want to model
        device (_type_): device object for PyTorch
        b (int, optional): the bandwith value. Defaults to 1.

    Returns:
        _type_: the mixed case of the MMD which is a value
    """

    n = x.shape[0]
    m = y.shape[0]

    # calculates euclidean distance tensor
    cdist = torch.cdist(x,y)
 
    #calculate kernel values
    kernel_values = kernel_gauss_cdist(cdist,b).to(device)

  
    #get the sum
    sum = torch.sum(kernel_values) 

    #weighting the sum
    res = 2/(m*n) * sum

    return res

def MMD_equal_case(x:torch.tensor,device,b:float=1):
    """calculates the sample x sample and response x response case for the MMD

    Args:
        x (torch.tensor): the input tensor
        device (_type_): the PyTorch device object
        b (float, optional): the bandwith value . Defaults to 1.

    Returns:
        _type_: the MMD value for the equal case (a single value)
    """

    n = x.shape[0]
    
    cdist = torch.cdist(x,x,p=2).to(device)
    
    
    #calculate kernel values
    kernel_values = kernel_gauss_cdist(cdist,b)
    
    #get the diagonal
    diag_kernel_values = kernel_values.diagonal()

    #remove the diagonal j==i elements
    sum = torch.sum(kernel_values) - torch.sum(diag_kernel_values)
    
    res = 1/(n*(n-1)) * sum
    

    return res


def sample_multivariate_logistic(n:int, m:int, alpha:float, device)->torch.tensor:

    # Step 1: 
    # simulate from a positive stable distribution
    S = sample_PS(n, m, alpha, device)

    # Step 2:
    # sample random standard exponential variables independent of S
    W = torch.zeros([n,m]).exponential_(lambd=1).to(device)
    X = (S/W)**alpha

    return X


def sample_PS(n:int, m:int, alpha:float, device)-> torch.tensor:

    # REMEMBER: unclear if S is rowwise identical. 
    # I assume so because of lack of index

    W = torch.zeros([n,1]).exponential_(1).to(device)

    U = torch.zeros([n,1]).uniform_(0,torch.pi).to(device)

    exponent = (1-alpha)/alpha

    comp1 = torch.sin((1-alpha)*U)/W

    comp2 = torch.sin(alpha*U)/(torch.sin(U)**(1-alpha))

    S = comp1**exponent * comp2

    S = S.repeat(1,m)

    return S

