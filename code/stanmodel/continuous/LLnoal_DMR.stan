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
  real<lower=0> sigmaCE;
}

transformed parameters {
  vector[N] CEpred;
  vector[N] pi_p;
  for(i in 1:N){
    // subjective probability
    pi_p[i] = inv_logit(a * logit(p[i]) + b);
    // CEpred(the mean of predicted CE)
    CEpred[i] = pi_p[i]*x1[i]+(1-pi_p[i])*x2[i];
  }
}

model {
  a ~ normal(0,1);
  b ~ normal(0,1);
  sigmaCE ~ cauchy(0,5);
  CEobs ~ normal(CEpred, sigmaCE);
}

generated quantities {
  vector[N] log_lik;
  for (i in 1:N)
    log_lik[i] = normal_lpdf(CEobs[i] | CEpred[i], sigmaCE);
}
