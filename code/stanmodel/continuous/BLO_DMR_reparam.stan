functions {
  real pi_p (real p, real c, real D, real tau, real kappa, real L0) {
    // hard clipping & scaling
    real L = tau * fmin(fmax(logit(p)-c, -D), D);
    // variational compensation
    real w = 1 / (1+kappa * p * (1-p));
    return inv_logit(w * L + (1-w) * L0);
  }
}

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
  real<lower=-4.59, upper=4.59> center;
  real<lower=0, upper=4.59> Delta;
  real<lower=0> tau;
  real<lower=0> kappa;
  real<lower=-10, upper=10> L0;
  real<lower=0> alpha;        
  // reparmetrization
  center = (2*inv_logit(center_raw)-1)*4.59;
  Delta = inv_logit(Delta_raw)*4.59;
  tau = exp(tau_raw);
  kappa = exp(kappa_raw);
  L0 = (2*inv_logit(L0_raw)-1)*10;
  alpha = exp(alpha_raw);
}

model {
  center_raw ~ normal(0,1); 
  Delta_raw ~ normal(0,1);
  tau_raw ~ normal(0,1);
  kappa_raw ~ normal(0,1);
  L0_raw ~ normal(0,1);
  alpha_raw ~ normal(0,1);
  sigmaCE ~ cauchy(0,5);
  vector[N] CEpred;
  vector[N] pip;
  for(i in 1:N){
    pip[i] = pi_p(p[i], center, Delta, tau, kappa, L0);
    CEpred[i] = pow(pip[i]*pow(x1[i], alpha) + (1-pip[i])*pow(x2[i], alpha), 1/alpha);
  }
  CEobs ~ normal(CEpred, sigmaCE);
}

generated quantities {
  vector[N] log_lik;
  vector[N] CEpred;
  vector[N] pip;
  for (i in 1:N){
    pip[i] = pi_p(p[i], center, Delta, tau, kappa, L0);
    CEpred[i] = pow(pip[i]*pow(x1[i], alpha) + (1-pip[i])*pow(x2[i], alpha), 1/alpha);
    log_lik[i] = normal_lpdf(CEobs[i] | CEpred[i], sigmaCE);
  }
}
