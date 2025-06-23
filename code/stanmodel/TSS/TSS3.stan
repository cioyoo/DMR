functions {
  real pip(real p, real gamma) {
    return inv_logit(gamma*logit(p));
  }
  real compute_V(real y, real p, real x1, real x2, real gamma, real alpha) {
    real w = pip(p, gamma);
    real gain = w * (x1 - y);
    real loss = (1 - w) * (y - x2);
    return pow(gain, alpha) - pow(loss, alpha);
  }
}

data {
  int<lower=1> N;
  vector[N] p;
  vector[N] x1;
  vector[N] x2;
  vector[N] CE;
  int c_idx[N];
  int f_idx[N];
}

parameters {
  real beta_raw;
  real gamma_raw;
  real alpha_raw;
}

transformed parameters {
  real<lower=0> beta = exp(beta_raw);
  real<lower=0> gamma = exp(gamma_raw);
  real<lower=0> alpha = exp(alpha_raw);
}


model{
  beta_raw ~ normal(0,1);
  gamma_raw~ normal(0,1);
  alpha_raw ~ normal(0,1);
  
  for(i in 1:N){
    vector[5] V_c;
    vector[5] V_f;
    // Coarse selection
    for(j in 1:5){
      real Y_c = x2[i] + (x1[i] - x2[i])*(-0.02 + 0.04*(5*(j-1)+3));
      V_c[j] = -abs(beta * compute_V(Y_c, p[i], x1[i], x2[i], gamma, alpha));
    }
    int action_c = (c_idx[i]-1) / 5 + 1;
    action_c ~ categorical_logit(V_c);
    // Fine selection
    for(j in 1:5){
      real Y_f = x2[i] + (x1[i] - x2[i])*(-0.02 + 0.04*(c_idx[i]-3+j));
      V_f[j] = -abs(beta * compute_V(Y_f, p[i], x1[i], x2[i], gamma, alpha));
    }
    int action_f = f_idx[i] - c_idx[i] + 3;
    action_f ~ categorical_logit(V_f);  
    
  }
}

generated quantities {
  vector[N] log_lik;

  for (i in 1:N) {
    vector[5] V_c;
    vector[5] V_f;

    for (j in 1:5) {
      real Y_c = x2[i] + (x1[i] - x2[i]) * (-0.02 + 0.04 * (5 * (j - 1) + 3));
      V_c[j] = -abs(beta * compute_V(Y_c, p[i], x1[i], x2[i], gamma, alpha));
    }
    int action_c = c_idx[i] / 5 + 1;

    for (j in 1:5) {
      real Y_f = x2[i] + (x1[i] - x2[i]) * (-0.02 + 0.04 * (c_idx[i] - 3 + j));
      V_f[j] = -abs(beta * compute_V(Y_f, p[i], x1[i], x2[i], gamma, alpha));
    }
    int action_f = f_idx[i] - c_idx[i] + 3;

    log_lik[i] =
      categorical_logit_lpmf(action_c | V_c) +
      categorical_logit_lpmf(action_f | V_f);
  }
}
