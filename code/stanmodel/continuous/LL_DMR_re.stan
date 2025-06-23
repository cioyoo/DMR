functions {
  real pi_p (real p, real a, real b) {
    return inv_logit(a*logit(p)+b);
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
  real a;
  real b;   
  real alpha_raw;
  real<lower=0> sigmaCE;
}

transformed parameters {
  real<lower=0> alpha = exp(alpha_raw);
  vector[N] CEpred;
  for(i in 1:N){
    // CEpred(the mean of predicted CE)
    CEpred[i] = pow(pi_p(p[i],a,b)*pow(x1[i], alpha) + pi_p(1-p[i],a,b)*pow(x2[i], alpha), 1/alpha);
  }
}

model {
  a ~ normal(0,1);
  b ~ normal(0,1);
  alpha_raw ~ normal(0,1);
  sigmaCE ~ cauchy(0,5);
  CEobs ~ normal(CEpred, sigmaCE);
}

generated quantities {
  vector[N] log_lik;
  for (i in 1:N)
    log_lik[i] = normal_lpdf(CEobs[i] | CEpred[i], sigmaCE);
}
