# Frozen stopping regression

`cycle_seed312.npz` is a five-variable software regression input generated with
CPython 3.12.2 and NumPy 2.2.6. It contains no participant data. SHA256:
`33ee7614d545f49e6d48dac7a310e2f6c790914590dfef47bb224722dd55b950`.

The generator is the exact reported seed, including the unused draw:

```python
import numpy as np
g = np.random.default_rng(312)
p = int(g.integers(5, 40))
g.standard_normal((p, p))
R = np.corrcoef(g.standard_normal((300, 3)) @ g.standard_normal((3, p))
               + 0.3 * g.standard_normal((300, p)), rowvar=False)
R_gwas = np.corrcoef(g.standard_normal((300, 3)) @ g.standard_normal((3, p))
                    + 0.3 * g.standard_normal((300, p)), rowvar=False)
b = np.zeros(p)
b[g.choice(p, 3, replace=False)] = g.normal(0, 6, 3)
z = R_gwas @ b + g.standard_normal(p)
```

The test reads the frozen arrays, so a change in random generation or BLAS
rounding does not silently change the regression. The input is a numerical
counterexample with mismatched LD, not a biological analysis or simulation study.
