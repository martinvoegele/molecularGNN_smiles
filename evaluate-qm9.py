import numpy as np
import matplotlib.pyplot as plt
%matplotlib inline

targets = np.loadtxt('qm9_targets.dat',dtype=str)[:,1]
factors = [1., 1., 27.2114, 27.2114, 27.2114, 1., 27211.4, 1., 1., 1., 1., 1., 0.043363, 0.043363, 0.043363, 0.043363, 1., 1., 1., 1., 0.043363]

assert len(factors) == len(targets)

seeds = ['11','22','33']

mae_avg = [] 
mae_std = []

for target in targets:
    
    results = []
    
    for seed in seeds:
        data = np.loadtxt('output/result--qm9-'+target+'--radius1--dim50--layer_hidden6--layer_output6--batch_train32--batch_test32--lr1e-4--lr_decay0.99--decay_interval10--iteration1000--seed'+seed+'.txt', 
                          skiprows=1)
        epoch, time, loss_train, mae_dev, mae_test = data.T
        result = mae_test[np.argmin(mae_dev)]
        results.append(result)
        
    mae_avg.append(np.mean(results))
    mae_std.append(np.std(results))
    
for i in range(len(targets)):
    print('%9s %8.4f %8.4f'%(targets[i], mae_avg[i]*factors[i], mae_std[i]*factors[i]))
    
    
    