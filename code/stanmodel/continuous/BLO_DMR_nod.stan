functions {
  real pi_p (real p, real A, real B, real L0) {
    // variance compensation
    real w = 1/(exp(A)+exp(B)*p*(1-p));
    return inv_logit(w*(logit(p)-L0) + L0);
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
  real A;
  real B;
  real<lower=-10, upper=10> L0;
  real alpha_raw;
  real<lower=0> sigmaCE;
}

transformed parameters {
  real<lower=0> tau = exp(-exp(A));
  real<lower=0> kappa = exp(exp(B)-exp(A));
  real<lower=0> alpha = exp(alpha_raw);
  vector[N] subp;
  vector[N] subq;
  vector[N] CEpred;
  
  for(i in 1:N){
    // subjective probability of p
    subp[i] = pi_p(p[i], exp(A), exp(B), L0);
    // subjective probability of 1-p
    subq[i] = pi_p(1-p[i], exp(A), exp(B), L0);
    // CEpred(the mean of predicted CE)
    CEpred[i] = pow(subp[i]*pow(x1[i], alpha) + subq[i]*pow(x2[i], alpha), 1/alpha);
  }
}

model {
  A~ normal(0,1);
  B ~ normal(0,1);
  L0 ~ normal(0,10); // alternative of uniform(-10,10)
  alpha_raw ~ normal(0,1);
  sigmaCE ~ cauchy(0,5);
  CEobs ~ normal(CEpred, sigmaCE);
}

generated quantities {
  vector[N] log_lik;
  for (i in 1:N)
    log_lik[i] = normal_lpdf(CEobs[i] | CEpred[i], sigmaCE);
}
