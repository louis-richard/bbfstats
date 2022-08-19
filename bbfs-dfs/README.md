# BBFs dipolarization fronts

## Organisation
- [`bbfs-dfs_search.py`](./bbfs-dfs_search.py) contains the code to find large amplitude magnetic field changes and DF at the jet fronts using S11 and F12 methods.
- [`bbfs-dfs_examples.py`](./bbfs-dfs_examples.py) contains the code to plot two characteristic examples of jet fronts.
- [`bbfs-dfs_occurrence.py`](./bbfs-dfs_occurrence.py) contains the code to compute and plot the occurrence of: quiet jet fronts, jet fronts with DFs and turbulent jet fronts.

## Datasets used
- The magnetic field measured by the Flux Gate Magnetometer (FGM) ([Russell et al. 2016](https://link.springer.com/article/10.1007/s11214-014-0057-3))
 
|              | Data rate  | level |
|--------------|:----------:|------:|
| $`B`$ (GSE)  | srvy, brst | l2    |

- The electric field measured by the Electric field Double Probe (EDP) ([Ergun et al. 2016](https://link.springer.com/article/10.1007/s11214-014-0115-x),
  [Lindqvist et al. 2016](https://link.springer.com/article/10.1007/s11214-014-0116-9))
 
|              | Data rate | level |
|--------------|:---------:|------:|
| $`E`$ (GSM)  | brst      | l2    |

- The thermal ion (proton) moments are computed using the moments of the velocity distribution functions measured by the Fast Plasma Investigation (FPI) ([Pollock et al. 2016](https://link.springer.com/article/10.1007/s11214-016-0245-4)) removing the penetrating radiation.

|                | Data rate  | level |
|----------------|:----------:|:------|
| $`n_i`$        | fast      | l2    |
| $`V_i`$ (GSM)  | fast      | l2    |
| $`t_i`$        | fast      | l2    |

> **_NOTE:_** The spintone in removed from the bulk velocity

## Reproducibility
Find large 
  amplitude magnetic field changes and DF at the jet fronts using S11 and F12 methods.

```bash
python3.9 bbfs-dfs_search.py
```

Plot two characteristic examples of jet fronts.
```bash
python3.9 bbfs-dfs_examples.py
```

Compute and plot the occurrence of: quiet jet fronts, jet fronts with DFs and 
turbulent jet fronts. 
```bash
python3.9 bbfs_compile.py
```



