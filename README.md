# Comparing JAX & numba for Glicko calculation

This repository follows on from a [twitter
thread](https://twitter.com/xenophar/status/1134799173738303489) where I
mentioned that numba appeared to work faster than JAX for a particular code
example. I fully expect that things can be recoded to be fast in JAX, too, but
I'm not quite sure how to do it! I suspect the main culprit might be changing
shapes in the arguments, which force JAX to recompile things frequently (?). I
am hoping that the improvements required to make it fast could be illustrative
for new JAX users.

The repository contains the following files:

* `glicko_jax.py` and `glicko_numba.py` are exactly the same code (see diff
  below), except that the JAX file has `import jax.numpy as np` and uses `@jit`
  decorators rather than numba's `@jit(nopython=True)`.

* `Benchmark.ipynb` generates some fake data and runs both with timings.

### Notes

* I noticed that JAX ran a bit more quickly without the `@jit` decorators for
  the functions `calculate_mu_prime` and
  `calculate_approximate_approximate_likelihood`. I've left them in for now
  though to have the code as similar to `numba` as possible.

* The naive approach of just substituting the `jit` lines clearly doesn't work
  well, as JAX runs very slowly (20 s vs 121 ms for numba).

### Diff

Result of running `diff glicko_jax.py glicko_numba.py`:

```diff
1,2c1,2
< import jax.numpy as np
< from jax import jit
---
> import numpy as np
> from numba import jit
9c9
< @jit
---
> @jit(nopython=True)
18c18
< @jit
---
> @jit(nopython=True)
39c39
< @jit
---
> @jit(nopython=True)
61c61
< @jit
---
> @jit(nopython=True)
82c82
< @jit
---
> @jit(nopython=True)
120c120
< @jit
---
> @jit(nopython=True)
```
