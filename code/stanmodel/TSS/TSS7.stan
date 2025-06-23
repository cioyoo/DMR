functions {
  vector compute_V(vector Y, real p, real x1, real x2, real gamma, real alpha_gain, real alpha_loss) {
    real w = inv_logit(gamma*logit(p));
    return w * pow(x1-Y, alpha_gain) - (1-w) * pow(Y-x2, alpha_loss);
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

transformed data {
  array[N] vector[5] Y_c;
  array[N] vector[5] Y_f;

  for (j in 1:N) {
    for (k in 1:5) {
      Y_c[j][k] = x2[j] + (x1[j] - x2[j]) * (-0.02 + 0.04 * (5 * (k - 1) + 3));
      Y_f[j][k] = x2[j] + (x1[j] - x2[j]) * (-0.02 + 0.04 * (c_idx[j] - 3 + k));
    }
  }

}

parameters {
  vector[2] alpha_raw;
  real gamma_raw;
}

transformed parameters {
  real<lower=0> gamma = exp(gamma_raw);
  real<lower=0> alpha_gain = exp(alpha_raw[1]);
  real<lower=0> alpha_loss = exp(alpha_raw[1]+alpha_raw[2]);
}


model{
  gamma_raw~ normal(0,1);
  alpha_raw ~ normal(0,1);
  
  for(j in 1:N){
    // Coarse selection
    vector[5] V_c = -fabs(compute_V(Y_c[j], p[j], x1[j], x2[j], gamma, alpha_gain, alpha_loss));
    int action_c = (c_idx[j]) / 5 + 1;
    action_c ~ categorical_logit(V_c);
    // Fine selection
    vector[5] V_f = -fabs(compute_V(Y_f[j], p[j], x1[j], x2[j], gamma, alpha_gain, alpha_loss));
    int action_f = f_idx[j] - c_idx[j] + 3;
    action_f ~ categorical_logit(V_f);  
  }
}

generated quantities {
  vector[N] log_lik;

  for (j in 1:N) {
    vector[5] V_c = -fabs(compute_V(Y_c[j], p[j], x1[j], x2[j], gamma, alpha_gain, alpha_loss));
    int action_c = (c_idx[j]) / 5 + 1;
    real ll_c = categorical_logit_lpmf(action_c | V_c);
    vector[5] V_f = -fabs(compute_V(Y_f[j], p[j], x1[j], x2[j], gamma, alpha_gain, alpha_loss));
    int action_f = f_idx[j] - c_idx[j] + 3;
    real ll_f = categorical_logit_lpmf(action_f | V_f);
    log_lik[j] = ll_c + ll_f;
  }
}
