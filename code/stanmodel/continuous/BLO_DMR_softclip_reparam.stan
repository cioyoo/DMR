data {
  int<lower=1> N;               // number of data points
  vector[N] p;              // probabilities (on [0,1])
  vector[N] x1;            // gambling options: (x1, p) vs. (x2, 1-p)
  vector[N] x2; 
  vector[N] CEobs;
}

parameters {
  real center_raw;
  real Delta_raw;
  real tau_raw;
  real kappa_raw;
  real L0_raw;
  real alpha_raw;        
  real<lower=0> sigmaCE;
}

transformed parameters {
  // reparmetrization
  real<lower=-4.59, upper=4.59> center = (2*inv_logit(center_raw)-1)*4.59;
  real<lower=0, upper=4.59> Delta = inv_logit(Delta_raw)*4.59;
  real<lower=0> tau = exp(tau_raw);
  real<lower=0> kappa = exp(kappa_raw);
  real<lower=-10, upper=10> L0 = (2*inv_logit(L0_raw)-1)*10;
  real<lower=0> alpha = exp(alpha_raw); 
  // calculate CEpred
  vector[N] L = tau * Delta * tanh((logit(p) - center)/Delta);
  vector[N] w = 1 ./ (1+kappa * p .* (1-p));
  vector[N] pip = inv_logit(w .* L + (1-w) * L0);
  vector[N] CEpred = pow(pip .* pow(x1, alpha) + (1-pip) .* pow(x2, alpha),1/alpha);
}

model {
  center_raw ~ normal(0,1); 
  Delta_raw ~ normal(0,1);
  tau_raw ~ normal(0,1);
  kappa_raw ~ normal(0,1);
  L0_raw ~ normal(0,1);
  alpha_raw ~ normal(0,1);
  sigmaCE ~ cauchy(0,5);
  CEobs ~ normal(CEpred, sigmaCE);
}

generated quantities {
  vector[N] log_lik;
  for (i in 1:N){
    log_lik[i] = normal_lpdf(CEobs[i] | CEpred[i], sigmaCE);
  }
}
