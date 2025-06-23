data {
  int<lower=1> N;               // number of data points
  vector[N] p;              // probabilities (on [0,1])
  vector[N] x1;            // gambling options: (x1, p) vs. (x2, 1-p)
  vector[N] x2; 
  vector[N] CEobs;
}

parameters {
  real<lower=0> sigmaCE;
  real<lower=0> alpha;            
}

transformed parameters {
  vector[N] CEpred;
  for(i in 1:N){
    // CEpred(the mean of predicted CE)
    CEpred[i] = pow(p[i]*pow(x1[i], alpha) + (1-p[i])*pow(x2[i], alpha), 1/alpha);
  }
}

model {
  sigmaCE ~ cauchy(0,5);
  alpha ~ cauchy(0,5);
  CEobs ~ normal(CEpred, sigmaCE);
}

generated quantities {
  vector[N] log_lik;
  for (i in 1:N)
    log_lik[i] = normal_lpdf(CEobs[i] | CEpred[i], sigmaCE);
}

