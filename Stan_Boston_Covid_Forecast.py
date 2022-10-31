import nest_asyncio
nest_asyncio.apply()
import stan
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd


model_code = """
data {
  int<lower=2> T;  // number of observations
  int<lower=0> y[T];     // observation vector
  int<lower=0> N; //Total population
  int<lower=2> Tf;
}
parameters {
  real<lower=0,upper=1> B; //Beta
  real<lower=0,upper=1> gam;//Gamma
  real<lower=0,upper=1> mu;//Mu
  real<lower=0> I_init;
  real<lower=0> E_init;
}
transformed parameters {
  vector[Tf] S;
  vector[Tf] E;
  vector[Tf] I;
  vector[Tf] R;
  vector[Tf] lam;
  
  S[1] = N - I_init - E_init;
  E[1] = E_init;
  I[1] = I_init;
  R[1] = 0;
  lam[1] = mu*E[1];

  for (t in 2:Tf){
    S[t] = S[t-1] - B*S[t-1]*I[t-1] / N; 
    E[t] = E[t-1] + B*S[t-1]*I[t-1] / N - mu*E[t-1];
    I[t] = I[t-1] + mu*E[t-1] - gam*I[t-1];
    R[t] = R[t-1] + gam*I[t-1];
    lam[t] = mu*E[t];
  }
}

model {
  gam ~ lognormal(log(1./8.),.2);
  B ~ lognormal(log(.4),.5);
  mu ~ lognormal(log(1),.5);
  I_init ~cauchy(0,100);
  E_init ~cauchy(0,100);
  
  for (t in 1:T){
    y[t] ~ poisson(lam[t]); 
  }
}
generated quantities {
  vector[Tf] forecast; // forecast cases from T to Tf
  forecast=lam;
}
"""

boston_covid = pd.read_csv("boston_covid.csv", header=0, parse_dates=[0])
boston_covid.set_index("Date", inplace=True)
boston_covid = boston_covid.to_numpy()
boston_covid_14 = boston_covid[0:14]
boston_covid_28 = boston_covid[0:42]
boston_covid_14 = boston_covid_14.astype(int).flatten()
boston_covid_28 = boston_covid_28.astype(int).flatten()


T = 14
Tf = 14+28
N = 68900
boston_data = {"T": T, "y": boston_covid_14, "N": N, "Tf": Tf}

boston_posterior = stan.build(model_code, data=boston_data, random_seed=1)
boston_fit = boston_posterior.sample(num_chains=1, num_samples=1000)

boston_fit

forecast_mean = np.mean(boston_fit["forecast"], axis=1)
fig, ax = plt.subplots()
ax.plot(boston_covid_14, label="observed")
ax.plot(forecast_mean[:14], label="forecast")
ax.set_title("Boston Covid-19 Forecast")
ax.set_xlabel("time in days")
ax.set_ylabel("new infections")
ax.legend()
plt.legend()
plt.show()

fig, ax = plt.subplots()
plt.plot(np.mean(boston_fit['forecast'], axis=1), label='forecast')
ax.set_title("Boston SEIR Forecast")
plt.xlabel("time in days")
plt.ylabel("new infections")
plt.fill_between(np.arange(0,42), np.percentile(boston_fit["forecast"],2.5,axis=1), 
                 np.percentile(boston_fit["forecast"],97.5,axis=1), color='b', alpha=.1)
plt.plot(boston_covid_28, color="r", label="observed")
plt.legend()
plt.show()