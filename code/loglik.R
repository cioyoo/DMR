# 1. Auxilary functions =====================================================
logit =  function(p) log(p / (1 - p))
inv_logit = function(z){return(1/(1+exp(-z)))}
softmax = function(v, beta=1) {
  v_scaled = beta * v
  v_shifted = v_scaled - max(v_scaled)
  probs = exp(v_shifted)
  return(probs / sum(probs))
}

# 2. TSS model minus LL =======================================================
TSS.lik = function(gamma=1, alpha_gain=1, alpha_loss=alpha_gain, lambda=1, data){
  output=0
  for(i in 1:data$N){
    dx = data$x1[i] - data$x2[i]
    base = data$x2[i]
    # coarse selection
    options_c = base + dx * c(0.1, 0.3, 0.5, 0.7, 0.9)
    pip = inv_logit(gamma * logit(data$p[i]))
    V_c = pip * (data$x1[i] - options_c)^alpha_gain - lambda * (1 - pip) * (options_c - data$x2[i])^alpha_loss
    V_c = -abs(V_c)
    c_action = floor((data$c_idx[i])/5)+1
    c_lik = V_c[c_action] - max(V_c) - log(sum(exp(V_c - max(V_c))))
    # fine selection
    options_f = base + dx * (-0.02 + 0.04 * ((data$c_idx[i]) + (-2:2)))
    V_f = pip * (data$x1[i] - options_f)^alpha_gain - lambda * (1 - pip) * (options_f - data$x2[i])^alpha_loss
    V_f = -abs(V_f)
    f_action = data$f_idx[i] - data$c_idx[i] + 3
    f_lik = V_f[f_action] - max(V_f) - log(sum(exp(V_f - max(V_f))))
    # add minus LL
    output = output + c_lik + f_lik
  }
  return(-output)
}

# TSS lambda (<-> TSS5.stan)
# * param = c(gamma, alpha, lambda)
TSS_lambda.lik = function(param, data){
  gamma = exp(param[1])
  alpha = exp(param[2])
  lambda = exp(param[3])
  output = TSS.lik(gamma=gamma, alpha_gain=alpha, alpha_loss=alpha, lambda=lambda, data)
  return(output)
}

# TSS alpha (<-> TSS7.stan)
# * param = c(gamma, alpha_raw[1], alpha_raw[2])
TSS_alpha.lik = function(param, data){
  gamma = exp(param[1])
  alpha_gain = exp(param[2])
  alpha_loss = exp(param[2]+param[3])
  output = TSS.lik(gamma=gamma, alpha_gain=alpha_gain, alpha_loss=alpha_loss, lambda=1, data)
  return(output)
}

# TSS full (<-> TSS8.stan)
# * param = c(gamma, alpha_raw[1], alpha_raw[2], lambda)
TSS_full.lik = function(param, data){
  gamma = exp(param[1])
  alpha_gain = exp(param[2])
  alpha_loss = exp(param[2] + param[3])
  lambda = exp(param[4])
  output = TSS.lik(gamma=gamma, alpha_gain=alpha_gain, alpha_loss=alpha_loss, lambda=lambda, data)
  return(output)
}


# 3. continuous model minus LL =================================================
# BLO model reparam
# ** param = c(center_raw, Delta_raw, tau_raw, kappa_raw, L0_raw, alpha_raw, sigma_raw)
BLO.reparam = function(param){
  center = (2*inv_logit(param[1])-1)*4.59
  Delta = inv_logit(param[2])*4.59
  tau = exp(param[3])
  kappa = exp(param[4])
  L0 = (2*inv_logit(param[5])-1)*10
  alpha = exp(param[6])
  sigma = exp(param[7])
  return(data.frame(center=center, Delta=Delta, tau=tau, kappa=kappa, L0=L0, alpha=alpha, sigma=sigma))
}

# BLO model: minus LL
# ** param = c(center_raw, Delta_raw, tau_raw, kappa_raw, L0_raw, sigma_raw)
BLO.lik = function(param, data){
  # reparametrization
  center = (2*inv_logit(param[1])-1)*4.59
  Delta = inv_logit(param[2])*4.59
  tau = exp(param[3])
  kappa = exp(param[4])
  L0 = (2*inv_logit(param[5])-1)*10
  alpha = exp(param[6])
  sigma = exp(param[7])
  # calculate pi(p); this is vector
  L = tau * pmin(pmax(logit(data$p)-center, -Delta), Delta) # lambda(p)
  w_p = 1/(1+ kappa * (data$p * (1-data$p))) # w_p
  pip = inv_logit(w_p*L + (1-w_p)*L0) # pi(p)
  # calculate CEpred and -LL
  CEpred = (pip * (data$x1)^alpha + (1-pip) * (data$x2)^alpha)^(1/alpha)
  log_lik = sum(dnorm(CEpred, mean = data$CEobs, sd = sigma, log = TRUE))
  return(-log_lik)
}

# LLO model reparam
# ** param = c(gamma_raw, L0, alpha_raw, sigma_raw)
LLO.reparam = function(param){
  gamma = exp(param[1])
  L0 = param[2]
  alpha = exp(param[3])
  sigma = exp(param[4])
  return(data.frame(gamma=gamma, L0=L0, alpha=alpha, sigma=sigma))
}

# LLO model: minus LL
# ** param = c(gamma_raw, L0, alpha_raw, sigma_raw)
LLO.lik = function(param, data){
  # reparametrization
  gamma = exp(param[1])
  L0 = param[2]
  alpha = exp(param[3])
  sigma = exp(param[4])
  # calculate CEpred and -LL
  pip = inv_logit(gamma * logit(data$p) + (1-gamma)*L0)
  CEpred = (pip * (data$x1)^alpha + (1-pip) * (data$x2)^alpha)^(1/alpha)
  log_lik = sum(dnorm(CEpred, mean = data$CEobs, sd = sigma, log = TRUE))
  return(-log_lik)
}

# simpleLO model: minus LL
# ** param = c(gamma_raw, alpha_raw, sigma_raw)
simpleLO.lik = function(param, data){
  # reparametrization
  gamma = exp(param[1])
  alpha = exp(param[2])
  sigma = exp(param[3])
  # calculate CEpred and -LL
  pip = inv_logit(gamma * logit(data$p))
  CEpred = (pip * (data$x1)^alpha + (1-pip) * (data$x2)^alpha)^(1/alpha)
  log_lik = sum(dnorm(CEpred, mean = data$CEobs, sd = sigma, log = TRUE))
  return(-log_lik)
}


# 4. MLE summary with estimated parameter & AIC/BIC=============================
MLE_summary = function(fit_list, variables, N_vec=c(rep(330, 51), rep(165, 24))){
  K = length(variables)
  df_summary = matrix(0, nrow=length(fit_list), ncol=K+2)
  for(i in 1:length(fit_list)){
    df_summary[i,1:K] = fit_list[[i]]$par
    AIC = 2 * fit_list[[i]]$value + 2 * K
    BIC = 2 * fit_list[[i]]$value + K * log(N_vec[i])
    df_summary[i,K+1:2] = c(AIC, BIC)
  }
  df_summary = as.data.frame(df_summary)
  colnames(df_summary) = c(variables, 'AIC', 'BIC')
  return(df_summary)
}
