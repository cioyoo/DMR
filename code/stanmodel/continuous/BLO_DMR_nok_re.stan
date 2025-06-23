functions {
  real pi_p (real p, real center, real Delta, real tau) {
    // soft clipping & linear transform to [-tau*Delta, tau*Delta]
    return inv_logit(tau*Delta*tanh((logit(p) - center) /Delta) + center);
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
  real center;
  real<lower=0> Delta;
  real tau_raw;
  real alpha_raw; 
  real<lower=0> sigmaCE;
}

transformed parameters {
  real<lower=0> tau = exp(tau_raw);
  real<lower=0> alpha = exp(alpha_raw);
  vector[N] subp;
  vector[N] subq;
  vector[N] CEpred;
  
  for(i in 1:N){
    // subjective probability of p
    subp[i] = pi_p(p[i], center, Delta, tau);
    // subjective probability of 1-p
    subq[i] = pi_p(1-p[i], center, Delta, tau);
    // CEpred(the mean of predicted CE)
    CEpred[i] = pow(subp[i]*pow(x1[i], alpha) + subq[i]*pow(x2[i], alpha), 1/alpha);
  }
}

model {
  center ~ normal(0, 4.59); // alternative of uniform(-4.59, 4.59)
  Delta ~ normal(2.295, 2.295); // alternative of uniform(0, 4.59)
  tau_raw ~ normal(0,1);
  alpha_raw ~ normal(0,1);
  sigmaCE ~ cauchy(0,5);
  CEobs ~ normal(CEpred, sigmaCE);
}

generated quantities {
  vector[N] log_lik;
  for (i in 1:N)
    log_lik[i] = normal_lpdf(CEobs[i] | CEpred[i], sigmaCE);
}
