data {
  int<lower=1> N;               // number of data points
  vector[N] p;              // probabilities (on [0,1])
  vector[N] x1;            // gambling options: (x1, p) vs. (x2, 1-p)
  vector[N] x2; 
  vector[N] CEobs;
}

parameters {
  real<lower=-4.59, upper=4.59> DeltaMinus;
  real<lower=0, upper=4.59> Delta;
  real tau_raw;
  real alpha_raw;  
  real<lower=-10, upper=10> L0;
  real<lower=0> sigmaCE;
}

transformed parameters {
  real<lower=0> tau = exp(tau_raw);
  real<lower=0> alpha = exp(alpha_raw);
  real DeltaPlus = DeltaMinus + 2*Delta;
  vector[N] subp;
  vector[N] CEpred;
  
  for(i in 1:N){
    // **soft** clipping between Delta & transform to +-Psi
    subp[i]  = tau * Delta * tanh((p[i] - DeltaMinus - Delta) /Delta);
    // subp = subjective prob.
    subp[i] = inv_logit(subp[i]+L0);
    // CEpred(the mean of predicted CE)
    CEpred[i] = pow(subp[i]*pow(x1[i], alpha) + (1-subp[i])*pow(x2[i], alpha), 1/alpha);
  }
}

model {
  DeltaMinus ~ normal(0, 4.59); // alternative of uniform(-4.59, 4.59)
  Delta ~ normal(2.295, 2.295); // alternative of uniform(0, 4.59)
  tau_raw ~ normal(0,1);
  alpha_raw ~ normal(0,1);
  L0 ~ normal(0,10); // alternative of uniform(-10,10)
  sigmaCE ~ cauchy(0,5);
  CEobs ~ normal(CEpred, sigmaCE);
}

generated quantities {
  vector[N] log_lik;
  for (i in 1:N)
    log_lik[i] = normal_lpdf(CEobs[i] | CEpred[i], sigmaCE);
}
