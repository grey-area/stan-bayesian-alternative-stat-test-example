import pystan
import numpy as np
import pickle
import os.path
import arviz as az
import matplotlib.pyplot as plt


code = """
data {
    int<lower=0> N1;  // number of data points for set 1
    int<lower=0> N2;  // number of data points for set 2
    vector[N1] y1;  // set 1
    vector[N2] y2;  // set 2
    real mu_prior_mean;  // parameters for our priors
    real mu_prior_std;
    real sigma_prior_low;
    real sigma_prior_high;
}
parameters {
    real group1_mu;
    real group2_mu;
    real<lower=0> group1_sigma;
    real<lower=0> group2_sigma;
    real<lower=1> nu;
}
model {
    group1_mu ~ normal(mu_prior_mean, mu_prior_std);
    group2_mu ~ normal(mu_prior_mean, mu_prior_std);
    group1_sigma ~ uniform(sigma_prior_low, sigma_prior_high);
    group2_sigma ~ uniform(sigma_prior_low, sigma_prior_high);
    nu ~ gamma(2, 0.1);

    // model data as coming from two student t distributions
    // with shared degrees of freedom parameter
    y1 ~ student_t(nu, group1_mu, group1_sigma);
    y2 ~ student_t(nu, group2_mu, group2_sigma);
}
"""

# Create our data
N1 = 30
y1 = np.random.normal(3, 1, size=N1)
N2 = 14
y2 = np.random.normal(2.5, 0.8, size=N2)
pooled_data = np.hstack((y1, y1))
pooled_data_mean = np.mean(pooled_data)
pooled_data_std = np.std(pooled_data)

data = {
    'N1': N1,
    'N2': N2,
    'y1': y1,
    'y2': y2,
    'mu_prior_mean': pooled_data_mean,
    'mu_prior_std': 1e3 * pooled_data_std,
    'sigma_prior_low': 1e-3 * pooled_data_std,
    'sigma_prior_high': 1e3 * pooled_data_std
}

# If we've already created and pickled the model, load it
# otherwise, compile it
if os.path.isfile('model.pkl'):
    sm = pickle.load(open('model.pkl', 'rb'))
else:
    sm = pystan.StanModel(model_code=code)
    with open('model.pkl', 'wb') as f:
        pickle.dump(sm, f)

fit = sm.sampling(data=data, iter=10000, chains=4)
print(fit)

samples = fit.extract(permuted=True)
group1_mean = samples['group1_mu']
group2_mean = samples['group2_mu']
print(
    f'Probability group 1 mean greater than group 2 mean:',
    np.mean(group1_mean > group2_mean)
)

az.plot_forest(samples, var_names=['group1_mu', 'group2_mu'])
plt.show()

az.plot_posterior(fit, var_names=['group1_mu', 'group2_mu'])
plt.show()