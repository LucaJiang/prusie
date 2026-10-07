"""Run the packaged example from any working directory, without R or downloads."""
import numpy as np
import prusie

example = prusie.load_example()
fit = prusie.susie_rss(**example['inputs'], **example['parameters'])
lead = int(np.argmax(fit.pip))
print(f'{fit.variant_ids[lead]}: PIP={fit.pip[lead]:.3f}')
print('Converged:', fit.converged, 'Iterations:', fit.niter)
for k, component in enumerate(fit.sets['cs_index']):
    print('Component:', int(component),
          'Variants:', fit.variant_ids[fit.sets['cs'][k]].tolist(),
          'Posterior mass:', round(float(fit.sets['coverage'][k]), 3))
