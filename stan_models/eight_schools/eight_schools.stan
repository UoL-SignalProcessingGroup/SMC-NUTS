data {
  int<lower=0> J;
  array[J] real y;
  array[J] real<lower=0> sigma;
  real<lower=0, upper=1> phi;
}
parameters {
  real mu;
  real<lower=0> tau;
  array[J] real theta_tilde;
}
transformed parameters {
  array[J] real theta;
  for (j in 1 : J) 
    theta[j] = mu + tau * theta_tilde[j];
}
model {
  mu ~ normal(0, 5);
  tau ~ cauchy(0, 5);
  theta_tilde ~ normal(0, 1);
//  y ~ normal(theta, sigma);
  target += normal_lpdf(y | theta, sigma);
}