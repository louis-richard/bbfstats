# BBFs distribution

## Organisation
- [`bbfs_search.py`](./bbfs_search.py) contains the code to find Earthward or tailward BBFs on a given day. This code is run through [`bbfs_search.sh`](./bbfs_search.sh) which searches for BBFs during the tail seasons.
- [`bbfs_merge.py`](./bbfs_merge.py) contains the code to merge the output of [`bbfs_search.py`](./bbfs_search.py) (time intervals).
- [`bbfs_compile.py`](./bbfs_compile.py) contains the code to load and save various quantities to characterise the BBFS.
- [`bbfs-dist_2d.py`](./bbfs-dist_2d.py) contains the code to plot the spatial distribution of BBFs on the equatorial and meridional planes.
- [`bbfs-dist_1d.py`](./bbfs-dist_1d.py) contains the code to plot the earth-tail distribution of the BBFs, the BBFs duration with respect to distance to Earth, and the distribution of distance to NS.

## Datasets used
- The magnetic field measured by the Flux Gate Magnetometer (FGM) ([Russell et al. 2016](https://link.springer.com/article/10.1007/s11214-014-0057-3))
 
|              | Data rate | level |
|--------------|:---------:|------:|
| $`B`$ (GSE)  | srvy      | l2    |

- The thermal ion (proton) moments are computed using the moments of the velocity distribution functions measured by the Fast Plasma Investigation (FPI) ([Pollock et al. 2016](https://link.springer.com/article/10.1007/s11214-016-0245-4)) removing the penetrating radiation.

|                | Data rate | level |
|----------------|:---------:|:------|
| $`n_i`$        | fast      | l2    |
| $`V_i`$ (GSM)  | fast      | l2    |
| $`t_i`$        | fast      | l2    |

> **_NOTE:_** The spintone in removed from the bulk velocity

## Reproducibility
Find BBFs in the tail seasons

```bash
./bbfs_search
```

Merge time intervals 
```bash
python3.9 bbfs_merge.py tints ../database/ ../data/
```

Compile the database
```bash
python3.9 bbfs_compile.py
```

Plot the Figures 1 and 2
```bash
python3.9 bbfs-dist_2d.py
python3.9 bbfs-dist_1d.py
```




