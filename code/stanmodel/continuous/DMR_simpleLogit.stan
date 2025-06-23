data {
  int<lower=1> N;               // number of data points
  vector[N] p;              // probabilities (on [0,1])
  vector[N] x1;            // gambling options: (x1, p) vs. (x2, 1-p)
  vector[N] x2; 
  vector[N] CEobs;
}

parameters {
  real gamma_raw;
  real alpha_raw;
  real<lower=0> sigmaCE;
}

transformed parameters {
  real<lower=0> gamma = exp(gamma_raw);
  real<lower=0> alpha = exp(alpha_raw);
}

model {
  gamma_raw ~ normal(0,1);
  alpha_raw ~ normal(0,1);
  sigmaCE ~ cauchy(0,5);
  vector[N] pi_p = inv_logit(gamma * logit(p));
  vector[N] CEpred = pow(pi_p .* pow(x1, alpha) + (1-pi_p) .* pow(x2, alpha), 1/alpha);
  CEobs ~ normal(CEpred, sigmaCE);
}

generated quantities {
  vector[N] log_lik;

  for (i in 1:N) {
    real pip_i = inv_logit(gamma * logit(p[i]));
    real CEpred_i = pow(pip_i * pow(x1[i], alpha) + (1 - pip_i) * pow(x2[i], alpha), 1 / alpha);
    log_lik[i] = normal_lpdf(CEobs[i] | CEpred_i, sigmaCE);
  }
}

