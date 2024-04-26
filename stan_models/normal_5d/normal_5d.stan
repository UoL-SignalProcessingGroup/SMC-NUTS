data {
    int<lower=0> K;
    real df[K];
    real mu[K];
    real<lower=0, upper=1> phi;
}
parameters {
    real x[K];
}
model {
    for(k in 1:K) {
        target += phi * normal_lpdf(x[k] | mu[k], df[k]);
    }
}