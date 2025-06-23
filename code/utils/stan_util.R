# Make data list required for stan model
make_stan_data = function(ID, type, df=df_total){
  df_ind = df %>% filter(SubjID == ID)
  if('Task' %in% colnames(df)){df_ind = df_ind %>% filter(Task == 2)}
  if(type == 'TSS'){
    # case1: TSS model
    f_idx = as.integer(round(((df_ind$CE - df_ind$x2)/(df_ind$x1 - df_ind$x2) + 0.02)/0.04))
    c_idx = as.integer(round(floor((f_idx-1)/5)*5+3))
    datalist = list(
      N = nrow(df_ind), p = df_ind$P_true, x1 = df_ind$x1, x2 = df_ind$x2,
      CE = df_ind$CE, c_idx = c_idx, f_idx = f_idx
    )
  }else{
    # case2: continuous model
    datalist = list(
      N = nrow(df_ind), p = df_ind$P_true, x1 = df_ind$x1, x2 = df_ind$x2,
      CEobs = df_ind$CE
    )
  }
  return(datalist)
}

# Make stan data for hierarchical model
make_stan_data_h = function(subj_vec, Tmax, df=df_total){
  if('Task' %in% colnames(df)){df = df %>% filter(Task == 2)}
  N = length(subj_vec) ## N
  Tsubj = rep(Tmax, N) ## Tsubj[N]
  p = matrix(0, nrow=N, ncol=Tmax) ## p[N, T]
  x1 = matrix(0, nrow=N, ncol=Tmax) ## x1[N, T]
  x2 = matrix(0, nrow=N, ncol=Tmax) ## x2[N, T]
  CE = matrix(0, nrow=N, ncol=Tmax) ## CE[N, T]
  c_idx = matrix(0, nrow=N, ncol=Tmax) ## c_idx[N, T]
  f_idx = matrix(0, nrow=N, ncol=Tmax) ## f_idx[N, T]
  for(i in 1:N){
    df_ind = df %>% filter(SubjID == subj_vec[i])
    T_i = min(Tmax, nrow(df_ind))
    Tsubj[i] = T_i
    p[i, 1:T_i] = df_ind[1:T_i, 'P_true']
    x1[i, 1:T_i] = df_ind[1:T_i, 'x1']
    x2[i, 1:T_i] = df_ind[1:T_i, 'x2']
    CE[i, 1:T_i] = df_ind[1:T_i, 'CE']
    f_idx_i = as.integer(round(((df_ind$CE - df_ind$x2)/(df_ind$x1 - df_ind$x2) + 0.02)/0.04))
    c_idx_i = as.integer(round(floor((f_idx_i-1)/5)*5+3))
    f_idx[i,1:T_i] = f_idx_i[1:T_i]
    c_idx[i,1:T_i] = c_idx_i[1:T_i]
  }
  data_list = list(
    N=N, T=Tmax, Tsubj=Tsubj, p=p, x1=x1, x2=x2, CE=CE, c_idx=c_idx, f_idx=f_idx
  )
  return(data_list)
}



# Extract posterior mean & convergence metric(Rhat, n_eff) & loo
MCMC_summary = function(fit_list, variables, convergence= TRUE, modelcomp=FALSE){
  # make empty matrix
  n = length(fit_list)
  colnum = length(variables)
  colnum = colnum + ifelse(convergence, 2, 0) + ifelse(modelcomp, 1, 0)
  output = matrix(0, ncol=colnum, nrow=n)
  # fill the matrix
  for(i in 1:n){
    fit = fit_list[[i]]
    output[i,1:length(variables)] = unlist(lapply(extract(fit, pars=variables), mean))
    if(convergence){
      fit_summary = summary(fit)$summary
      output[i,length(variables)+1] = max(fit_summary[variables,'Rhat'])
      output[i,length(variables)+2] = max(fit_summary[variables,'n_eff'])
    }
    if(modelcomp){
      log_lik_mat = extract_log_lik(fit, parameter_name = "log_lik")
      output[i, colnum] = loo(log_lik_mat)$estimates[[1]]
    }
  }
  output = data.frame(output, row.names=1:n)
  col_names = variables
  if(convergence){col_names = c(col_names, 'max.Rhat', 'min.n_eff')}
  if(modelcomp){col_names = c(col_names, 'loo')}
  colnames(output) = col_names
  return(output)
}

# Extract posterior mean of hierarchical model
hB_summary = function(fit, variables){
  posterior = extract(fit)
  df_summary = list()
  for(i in 1:length(variables)){
    df_summary[[variables[i]]] = apply(posterior[[variables[i]]], 2, mean)
  }
  return(as.data.frame(df_summary))
}
