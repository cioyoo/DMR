
# plot_simul : make plot simulated data from the model

# * obs_point_ID --> add points of observed data of the given ID, it doesn't add any point by default
# * simul_func = function(p, x1, x2, ...)
# * simul_func_arg = list(...)
plot_simul = function(simul_func, simul_func_arg=list(), obs_point_ID=0,  q_vec=c(.05, .2, .8, .95), ...){
  x1_vec = c(25, 50, 75, 100, 150, 200, 400, 800, 50, 75, 100, 150, 150, 200, 200)
  x2_vec = c(0, 0, 0, 0, 0, 0, 0, 0, 25, 50, 50, 50, 100, 100, 150)
  p_vec = c(.01, .05, .10, .25, .4, .5, .6, .75, .9, .95, .99)
  plot_list = list()
  # Make simulated data
  for(i in 1:length(x1_vec)){
    x1_val = x1_vec[i]
    x2_val = x2_vec[i]
    df_sim = matrix(0, nrow=length(p_vec), ncol= 6)
    df_sim[1:length(p_vec),1] = p_vec
    for(j in 1:length(p_vec)){
      p_val = p_vec[j]
      full_arg = c(list(p=p_val, x1=x1_val, x2=x2_val), simul_func_arg)
      sim = do.call(simul_func, full_arg)
      df_sim[j,2] = mean(sim)
      df_sim[j,3] = quantile(sim, q_vec[1])
      df_sim[j,4] = quantile(sim, q_vec[2])
      df_sim[j,5] = quantile(sim, q_vec[3])
      df_sim[j,6] = quantile(sim, q_vec[4])
    }
    df_sim = data.frame(df_sim)
    colnames(df_sim) = c('P_true', 'CE_mean', 'q1', 'q2', 'q3', 'q4')
    df_sim = df_sim %>% mutate(expected_value = P_true*x1_val + (1-P_true)*x2_val)
    # Make a plot
    p= df_sim %>% ggplot(aes(x=P_true))+
      geom_ribbon(aes(ymin = q1, ymax = q4), fill = "grey70") +
      geom_ribbon(aes(ymin = q2, ymax = q3), fill = "grey50") +
      geom_line(aes(y = CE_mean))+
      geom_line(aes(y = expected_value),color = "red",linetype = "dashed")
    # add points of observed data
    if(obs_point_ID != 0){
      df_ind = df_total %>% filter(SubjID==obs_point_ID, Task==2, x1==x1_val, x2==x2_val)
      p = p + geom_point(data=df_ind, aes(x=P_true, y=CE))
    }
    p = p + theme_classic() +
      labs(title=paste0('X1, X2 = ',x1_val,', ',x2_val))+xlab('p1')+ylab('CE')
    plot_list[[i]] = p
  }
  return(plot_list)
}

# Softmax function
softmax = function(v, beta=1) {
  v_scaled = beta * v
  v_shifted = v_scaled - max(v_scaled)
  probs = exp(v_shifted)
  return(probs / sum(probs))
}


# TSS simulation
# * V_func = function(y, p, x1, x2, ...)
TSS_simul = function(p, x1, x2, V_func, V_func_arg = list(), beta=1, iter=100) {
  Y_total = x2 + (x1-x2)*seq(0.02, 0.98, by=0.04)
  V_total = rep(0, 25)
  for(i in 1:25){
    full_arg = c(list(y=Y_total[i], p=p, x1=x1, x2=x2), V_func_arg)
    V_total[i] = do.call(V_func,full_arg)
  }
  c_idx_vec = rep(-1, iter)
  f_idx_vec = rep(-1, iter)
  CEsim = rep(-1, iter)
  # Coarse selection
  # Select among 0.1, 0.3, ... , 0.9 <-> 3, 8, 13, 18, 23, 28
  # Select the action with lowest absolute value
  V_dist = abs(V_total[3+5*0:4])
  c_idx_vec = sample(3+5*0:4, size= iter, prob = softmax(-V_dist, beta), replace=TRUE)
  for(i in 1:iter){
    c_idx = c_idx_vec[i]
    V_dist = abs(V_total[c_idx+ -2:2])
    f_idx = sample(c_idx+ -2:2, size=1, prob = softmax(-V_dist, beta))
    f_idx_vec[i] = f_idx
    CEsim[i] = Y_total[f_idx]
  }
  return(CEsim)
}

# Auxilary function for V_pt
logit =  function(p) log(p / (1 - p))
inv_logit = function(z){return(1/(1+exp(-z)))}

# Prospect theory value function
V_pt = function(y, p, x1, x2, gamma=1, alpha_gain=1, alpha_loss=alpha_gain, lambda=1){
  pip = inv_logit(gamma*logit(p))
  V = pip*(x1-y)^alpha_gain - lambda * (1-pip)*(y-x2)^alpha_loss
  return(V)
}