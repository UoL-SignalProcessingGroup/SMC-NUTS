// ARMA(1, 1)

data {
  int<lower=1> T; // number of observations
  array[T] real y; // observed outputs
  real<lower=0, upper=1> phi; // Tempering parameter is passed through as data
}
parameters {
  real mu; // mean coefficient
  real beta; // autoregression coefficient
  real theta; // moving average coefficient
  real<lower=0> sigma; // noise scale
}
model {
  vector[T] nu; // prediction for time t
  vector[T] err; // error for time t

  target += normal_lpdf(mu | 0, 10);
  target += normal_lpdf(beta | 0, 2);
  target += normal_lpdf(theta | 0, 2);
  target += cauchy_lpdf(sigma | 0, 2.5);

  nu[1] = mu + beta * mu; // assume err[0] == 0
  err[1] = y[1] - nu[1];
  for (t in 2 : T) {
    nu[t] = mu + beta * y[t - 1] + theta * err[t - 1];
    err[t] = y[t] - nu[t];
  }

  target += phi * normal_lpdf(err | 0, sigma);
}
