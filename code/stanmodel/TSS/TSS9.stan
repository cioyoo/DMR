functions {
  real pip (real p, real c, real D, real gamma) {
    // hard clipping & scaling
    real L = fmin(fmax(logit(p)-c, -D), D);
    // variational compensation
    return inv_logit(gamma * L);
  }
  real compute_V(real y, real p, real x1, real x2, real gamma, real alpha_gain, real alpha_loss, real center, real Delta) {
    real w = pip(p, center, Delta, gamma);
    real gain = w * pow(x1-y, alpha_gain);
    real loss = (1 - w) * pow(y-x2, alpha_loss);
    return (gain - loss);
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
  real center_raw;
  real Delta_raw;
  real alpha_raw;
  //vector[2] alpha_raw;
  real gamma_raw;
  //real lambda_raw;
}

transformed parameters {
  real<lower=-4.59, upper=4.59> center = (2*inv_logit(center_raw)-1)*4.59;
  real<lower=0, upper=4.59> Delta = inv_logit(Delta_raw)*4.59;
  real<lower=0> gamma = exp(gamma_raw);
  real<lower=0, upper=1> alpha = inv_logit(alpha_raw);
  //real<lower=0, upper=1> alpha_gain = inv_logit(alpha_raw[1]);
  //real<lower=0, upper=1> alpha_loss = inv_logit(alpha_raw[1]+alpha_raw[2]);
  //real<lower=0> lambda = exp(lambda_raw);
}


model{
  center_raw ~ normal(0,1); 
  Delta_raw ~ normal(0,1);
  gamma_raw~ normal(0,1);
  alpha_raw ~ normal(0,1);
  //lambda_raw ~ normal(0,1);
  
  for(i in 1:N){
    vector[5] V_c;
    vector[5] V_f;
    // Coarse selection
    for(j in 1:5){
      real Y_c = x2[i] + (x1[i] - x2[i])*(-0.02 + 0.04*(5*(j-1)+3));
      V_c[j] = -abs(compute_V(Y_c, p[i], x1[i], x2[i], gamma, alpha, alpha,  center, Delta));
    }
    int action_c = (c_idx[i]-1) / 5 + 1;
    action_c ~ categorical_logit(V_c);
    // Fine selection
    for(j in 1:5){
      real Y_f = x2[i] + (x1[i] - x2[i])*(-0.02 + 0.04*(c_idx[i]-3+j));
      V_f[j] = -abs(compute_V(Y_f, p[i], x1[i], x2[i], gamma, alpha, alpha, center, Delta));
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
      V_c[j] = -abs(1 * compute_V(Y_c, p[i], x1[i], x2[i], gamma, alpha, alpha, center, Delta));
    }
    int action_c = c_idx[i] / 5 + 1;

    for (j in 1:5) {
      real Y_f = x2[i] + (x1[i] - x2[i]) * (-0.02 + 0.04 * (c_idx[i] - 3 + j));
      V_f[j] = -abs(1 * compute_V(Y_f, p[i], x1[i], x2[i], gamma, alpha, alpha, center, Delta));
    }
    int action_f = f_idx[i] - c_idx[i] + 3;

    log_lik[i] =
      categorical_logit_lpmf(action_c | V_c) +
      categorical_logit_lpmf(action_f | V_f);
  }
}
