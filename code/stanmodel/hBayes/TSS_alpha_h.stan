functions {
  vector compute_V(vector Y, real p, real x1, real x2, real gamma, real alpha_gain, real alpha_loss) {
    real w = inv_logit(gamma*logit(p));
    return w * pow(x1-Y, alpha_gain) - (1-w) * pow(Y-x2, alpha_loss);
  }
}

data {
  int<lower=1> N; // subject number
  int<lower=1> T; // maximum trial number
  int<lower=1, upper=T> Tsubj[N]; // trial number for each subject
  real p[N,T]; // P_true
  real x1[N,T]; // x1
  real x2[N,T]; // x2
  real CE[N,T]; // CE
  int c_idx[N,T]; // cooarse(first) selection
  int f_idx[N,T]; // fine(second) selection
}

transformed data {
  array[N, T] vector[5] Y_c;
  array[N, T] vector[5] Y_f;
  for (i in 1:N) {
    for (j in 1:Tsubj[i]) {
      for (k in 1:5) {
        Y_c[i, j][k] = x2[i, j] + (x1[i, j] - x2[i, j]) * (-0.02 + 0.04 * (5 * (k - 1) + 3));
        Y_f[i, j][k] = x2[i, j] + (x1[i, j] - x2[i, j]) * (-0.02 + 0.04 * (c_idx[i, j] - 3 + k));
      }
    }
  }
}

parameters {
  // group level parameters
  vector[3] mu_p;
  vector<lower=0>[3] sigma;
  
  // subject level raw parameters (for Matt trick)
  vector[N] gamma_pr;
  vector[N] alpha_1_pr;
  vector[N] alpha_2_pr;
}

transformed parameters {
  // subject level parameters
  vector<lower=0>[N] gamma = exp(mu_p[1]+sigma[1]*gamma_pr);
  vector<lower=0>[N] alpha_gain = exp(mu_p[2]+sigma[2]*alpha_1_pr);
  vector<lower=0>[N] alpha_loss = exp(mu_p[2]+sigma[2]*alpha_1_pr + mu_p[3]+sigma[3]*alpha_2_pr);
}


model{
  mu_p ~ normal(0,1);
  sigma ~ normal(0,1);
  gamma_pr ~ normal(0,1);
  alpha_1_pr ~ normal(0,1);
  alpha_2_pr ~ normal(0,1);
  
  for(i in 1:N){
    for(j in 1:Tsubj[i]){
      // Coarse selection
      vector[5] V_c = -fabs(compute_V(Y_c[i,j], p[i,j], x1[i,j], x2[i,j], gamma[i], alpha_gain[i], alpha_loss[i]));
      int action_c = (c_idx[i,j]) / 5 + 1;
      action_c ~ categorical_logit(V_c);
      // Fine selection
      vector[5] V_f = -fabs(compute_V(Y_f[i,j], p[i,j], x1[i,j], x2[i,j], gamma[i], alpha_gain[i], alpha_loss[i]));
      int action_f = f_idx[i,j] - c_idx[i,j] + 3;
      action_f ~ categorical_logit(V_f);  
    }
  }
}

generated quantities {
  real mu_gamma = exp(mu_p[1]);
  real mu_alpha_gain = exp(mu_p[2]);
  real mu_alpha_loss = exp(mu_p[2] + mu_p[3]);
  matrix[N, T] log_lik;
  log_lik = rep_matrix(0, N, T);

  for (i in 1:N) {
    for (j in 1:Tsubj[i]) {
      vector[5] V_c = -fabs(compute_V(Y_c[i,j], p[i,j], x1[i,j], x2[i,j], gamma[i], alpha_gain[i], alpha_loss[i]));
      int action_c = (c_idx[i,j] - 1) / 5 + 1;
      real ll_c = categorical_logit_lpmf(action_c | V_c);

      vector[5] V_f = -fabs(compute_V(Y_f[i,j], p[i,j], x1[i,j], x2[i,j], gamma[i], alpha_gain[i], alpha_loss[i]));
      int action_f = f_idx[i,j] - c_idx[i,j] + 3;
      real ll_f = categorical_logit_lpmf(action_f | V_f);

      log_lik[i, j] = ll_c + ll_f;
    }
  }
}
