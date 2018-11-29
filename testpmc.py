import numpy as np
import pylab as plt
from abcpy.continuousmodels import Uniform
from Model import StochLorenz95
from Statistics import HakkarainenLorenzStatistics

# Define Graphical Model
theta1 = Uniform([[0], [1]], name='theta1')
theta2 = Uniform([[0], [1]], name='theta2')
timestep = 120
lorenz = StochLorenz95([theta1, theta2, timestep], name='lorenz')

# Example to Generate Data to check it's correct
data_obs = lorenz.forward_simulate([.3, .6, 1200], 1)

# define backend
# Note, the dummy backend does not parallelize the code!
#from abcpy.backends import BackendDummy as Backend
from abcpy.backends import BackendMPI as Backend
backend = Backend()

## GPEMCEE with synlik
# Define the likelihood function
# Define Statistics
statistics_calculator = HakkarainenLorenzStatistics(degree=2, cross=True)
from abcpy.approx_lhd import SynLiklihood
likfun = SynLiklihood(statistics_calculator)

from abcpy.inferences import GPEMCEE
steps, n_samples, n_samples_per_param, n_design, n_burnin = 2, 1000, 100, 100, 1000
sampler = GPEMCEE([lorenz], [likfun], backend, seed=1)
print('GPEMCEE Inferring')
journal, samples, design_points, lhd, aaa = sampler.sample([data_obs], steps, n_samples, n_samples_per_param, n_design, n_burnin)
print('GPEMCEE done')

print(samples.shape)
print(design_points.shape)

plt.figure()
plt.plot(samples[:,0],samples[:,1],'*b')
plt.xlim([0, 1])
plt.ylim([0, 1])
plt.show()



# from abcpy.inferences import PMC
# steps, n_samples, n_samples_per_param = 2, 100, 10
# sampler = PMC([lorenz], [likfun], backend, seed=1)
# print('PMC Inferring')
# journal = sampler.sample([data_obs], steps, n_samples, n_samples_per_param, covFactors=np.array([.1, .1]), iniPoints=None)
# print('PMC Done')
# print('PMC done')
# mu_sample = np.array(journal.get_parameters()['mu'])
# sigma_sample = np.array(journal.get_parameters()['sigma'])
# import matplotlib.pyplot as plt
# plt.figure()
# plt.plot(mu_sample,sigma_sample,'.')
# plt.savefig('pmcpenlogred.eps',format='eps', dpi=1000)
#
#
# print(design_points.shape)
# #print(lhd)
# #mu_sample = np.array(journal.get_parameters()['mu'])
# #sigma_sample = np.array(journal.get_parameters()['sigma'])
#
# # Visualization of samples
# import matplotlib.pyplot as plt
#
# fig, (ax1, ax2) = plt.subplots(nrows=2)
#
# ax1.tricontour(design_points[:,0].reshape(-1,),design_points[:,1].reshape(-1,), -lhd.reshape(-1,), 14, linewidths=0.5, colors='k')
# cntr1 = ax1.tricontourf(design_points[:,0].reshape(-1,),design_points[:,1].reshape(-1,), -lhd.reshape(-1,), 14, cmap="RdBu_r")
# fig.colorbar(cntr1, ax=ax1)
# ax1.plot(design_points[:,0].reshape(-1,),design_points[:,1].reshape(-1,), 'ko', ms=3)
#
# plt.show()