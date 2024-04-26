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
    real lpdf = 0;
    for(k in 1:K) {
        target += lgamma((df[k]+1)/2);
        target += - lgamma(df[k]/2);
        target += - log( sqrt(df[k]*pi()) );
        target += - ((df[k]+1)/2)*log(1 + (1/df[k])*(x[k] - mu[k])*(x[k] - mu[k])) * phi;
        
    }
}