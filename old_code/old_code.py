def training_loop_gauss_resample_y(mu_hat, sigma_hat,target_dist_sample_size, nr_iterations , sample_size_x, device, b, optimizer, epoch_print_size=500):
    mu_hat_estimates = []
    sigma_hat_estimates = []
    MMD_values = []
   
    for epoch in np.arange(nr_iterations):

        # Empty gradient
        #optimizer.zero_grad()

        # Sample from the generator
        sample = torch.normal(0,1,(sample_size_x,1))
        sample = mu_hat + sigma_hat*sample
        # going from [sample_size,1] to [sample_size] tensor
        sample = sample.squeeze(1)

        # sample target dist
        target_dist = torch.normal(-2,3,(target_dist_sample_size,1))
        # going from [sample_size,1] to [sample_size] tensor
        target_dist = target_dist.squeeze(1)

        

        # Empty gradient
        optimizer.zero_grad() #location???



        # Calculate Loss
        loss = calc_MMD_1d(sample, target_dist, device, b)
        
    
        #optimizer.zero_grad()
        # Calculate gradient
        loss.backward()
        
        # Take one SGD step
        optimizer.step()

        mu_hat_estimates.append(mu_hat.detach().clone())
        sigma_hat_estimates.append(sigma_hat.detach().clone())
        MMD_values.append(loss.detach().clone())


        

        if epoch%epoch_print_size==0:
            print("epoch: ",epoch," loss=",loss)
            
    mu_hat_estimates = torch.stack(mu_hat_estimates).detach().numpy().reshape([nr_iterations,])
    sigma_hat_estimates = torch.stack(sigma_hat_estimates).detach().numpy().reshape([nr_iterations,])
    MMD_values = torch.stack(MMD_values).detach().numpy()

    df_results = pd.DataFrame({"mu_hat": mu_hat_estimates, 
              "sigma_hat": sigma_hat_estimates,
              "MMD": MMD_values})
    
    return df_results

def MMD_equal_case(x,device,b=1):

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

def MMD_equal_case_1d(x,device,b=1):

    n = x.shape[0]
    
    # Expand x to enable broadcasting
    x_expanded = x.unsqueeze(0).expand(n,-1).to(device)  # Shape: [n, n]
    x_expanded_t = x_expanded.t().to(device)
    
    
    #calculate kernel values
    kernel_values = kernel_gauss(x_expanded,x_expanded_t,b)
    
    #get the diagonal
    diag_kernel_values = kernel_values.diagonal()

    #remove the diagonal j==i elements
    sum = torch.sum(kernel_values) - torch.sum(diag_kernel_values)
    
    res = 1/(n*(n-1)) * sum
    

    return res

def MMD_mixed_case_1d(x,y,device,b=1):

    n = x.shape[0]
    m = y.shape[0]
    # Expand x and y to enable broadcasting
    x_expanded = x.unsqueeze(0).expand(m,-1).to(device)  # Shape: [n, m]
    y_expanded = y.unsqueeze(0).expand(n,-1).t().to(device)  # Shape: [n, m]

 

    #calculate kernel values
    kernel_values = kernel_gauss(x_expanded,y_expanded,b)

  
    
    

    #get the sum
    sum = torch.sum(kernel_values) 

    #weighting the sum
    res = 2/(m*n) * sum
    
    

    return res



def MMD_equal_case_1d(x,device,b=1):

    n = x.shape[0]
    
    # Expand x to enable broadcasting
    x_expanded = x.unsqueeze(0).expand(n,-1).to(device)  # Shape: [n, n]
    x_expanded_t = x_expanded.t().to(device)
    
    
    #calculate kernel values
    kernel_values = kernel_gauss(x_expanded,x_expanded_t,b)
    
    #get the diagonal
    diag_kernel_values = kernel_values.diagonal()

    #remove the diagonal j==i elements
    sum = torch.sum(kernel_values) - torch.sum(diag_kernel_values)
    
    res = 1/(n*(n-1)) * sum
    

    return res

def MMD_mixed_case_1d(x,y,device,b=1):

    n = x.shape[0]
    m = y.shape[0]
    # Expand x and y to enable broadcasting
    x_expanded = x.unsqueeze(0).expand(m,-1).to(device)  # Shape: [n, m]
    y_expanded = y.unsqueeze(0).expand(n,-1).t().to(device)  # Shape: [n, m]

 

    #calculate kernel values
    kernel_values = kernel_gauss(x_expanded,y_expanded,b)

  
    
    

    #get the sum
    sum = torch.sum(kernel_values) 

    #weighting the sum
    res = 2/(m*n) * sum
    
    

    return res

def calc_MMD_1d(x,y,device, b=1):

    x_case = MMD_equal_case_1d(x,device, b)

    y_case = MMD_equal_case_1d(y,device,b)
    

    xy_case = MMD_mixed_case_1d(x,y,device, b)

    y_case = MMD_equal_case_1d(y,device,b)
    

    MMD = x_case + y_case - xy_case

    

    return MMD

def kernel_gauss(x, y, b=1):

    res = torch.exp(-b*((x-y)**2))

    return res