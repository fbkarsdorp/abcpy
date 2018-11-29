import numpy as np
import pylab as plt
# import pyGPs
# from QuadMean import Quadratic
#
# model = pyGPs.GPR()
#
# x = np.random.uniform(-5,5,10).reshape(10,1)
# z = np.linspace(-6,6,100).reshape(100,1)
# y = (1/100) * (pow(x,6) - 30 * pow(x , 4) + 234 * pow(x , 2) + 14 * x + 100 + np.sin(17 * x) + np.cos(11 * x))
#
# a = np.random.uniform(-5,5,20).reshape(10,2)
# L = np.hstack((a,pow(a,2)))
# print(L)
# m1 = Quadratic( D=x.shape[1] * 2 ) + pyGPs.mean.Const()
# k = pyGPs.cov.RBF()
# model.setPrior(mean=m1, kernel=k)
# model.getPosterior(x, y) # fit default model (mean zero & rbf kernel) with data
# model.optimize(x, y)
# # optimize hyperparamters (default optimizer: single run minimize)
# ym, ys2, fm, fs2, lp = model.predict(z)
# # predict test cases
# plt.figure()
# plt.plot(x, y, 'b*')
# plt.plot(z, ym, 'k-')
# plt.show()

# define observation for true parameters mean=170, std=15
height_obs = [160.82499176]

# define prior
from abcpy.continuousmodels import Uniform

mu = Uniform([[150], [200]], name='mu')
sigma = Uniform([[5], [25]], name='sigma')

# define the model
from abcpy.continuousmodels import Normal

height = Normal([mu, sigma], name = 'height')


# define statistics
from abcpy.statistics import Identity
statistics_calculator = Identity(degree=2, cross=True)

# define backend
# Note, the dummy backend does not parallelize the code!
# from abcpy.backends import BackendDummy as Backend
from abcpy.backends import BackendDummy as Backend
backend = Backend()

## GPEMCEE with synlik
# Define the likelihood function
from abcpy.approx_lhd import SynLiklihood
likfun = SynLiklihood(statistics_calculator)

from abcpy.inferences import GPEMCEE
steps, n_samples, n_samples_per_param, n_design, n_burnin = 4, 1000, 100, 100, 10
sampler = GPEMCEE([height], [likfun], backend, seed=1)
print('GPEMCEE Inferring')
journal, samples, design_points, lhd, aaa = sampler.sample([height_obs], steps, n_samples, n_samples_per_param, n_design, n_burnin)
print('GPEMCEE done')

print(samples.shape)
print(design_points.shape)

plt.figure()
plt.plot(samples[:,0],samples[:,1],'*b')
plt.xlim([100, 200])
plt.ylim([1, 25])
plt.show()

# print(design_points.shape)
# #print(lhd)
# #mu_sample = np.array(journal.get_parameters()['mu'])
# #sigma_sample = np.array(journal.get_parameters()['sigma'])
#
# # Visualization of samples
