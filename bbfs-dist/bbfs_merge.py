import argparse
import os

import h5py as h5
import pandas as pd
import numpy as np
import xarray as xr

from pyrfu import pyrf


def _fname(year, month, day, suff):
    return os.path.join(
        year, month, day, f"{year}{month}{day}_000000_235959_{suff}.csv"
    )


def _merge_tints(args):
    bbfs_tailward, bbfs_earthward = [], []
    for year in ["2017", "2018", "2019", "2020", "2021"]:
        bbfs_t, bbfs_e = [], []
        for month in np.sort(os.listdir(os.path.join(args.inp_path, year))):
            m_path = os.path.join(args.inp_path, year, month)
            days = np.sort(os.listdir(m_path))
            days = filter(lambda d: os.listdir(os.path.join(m_path, d)), days)
            days = list(days)

            try:
                file_ew = _fname(year, month, days[0], "earthward")
                file_ew = os.path.join(args.inp_path, file_ew)
                file_tw = _fname(year, month, days[0], "tailward")
                file_tw = os.path.join(args.inp_path, file_tw)
                bbfs_e.append(pd.read_csv(file_ew, header=None))
                bbfs_t.append(pd.read_csv(file_tw, header=None))
            except FileNotFoundError:
                pass

            for day in days[1:]:
                try:
                    file_ew = _fname(year, month, day, "earthward")
                    file_ew = os.path.join(args.inp_path, file_ew)
                    file_tw = _fname(year, month, day, "tailward")
                    file_tw = os.path.join(args.inp_path, file_tw)
                    tmp_e = pd.read_csv(file_ew, header=None)
                    tmp_t = pd.read_csv(file_tw, header=None)
                    bbfs_e[-1] = bbfs_e[-1].append(tmp_e)
                    bbfs_t[-1] = bbfs_t[-1].append(tmp_t)
                except FileNotFoundError:
                    pass

        out_e = bbfs_e[0]
        out_t = bbfs_t[0]
        for e, t in zip(bbfs_e[1:], bbfs_t[1:]):
            out_e = out_e.append(e)
            out_t = out_t.append(t)

        out_e = out_e.dropna(axis=0, how="any").drop_duplicates()
        out_t = out_t.dropna(axis=0, how="any").drop_duplicates()
        bbfs_earthward.append(out_e)
        bbfs_tailward.append(out_t)

        # out.to_csv(f"bbfs_database_tailward_{year}.csv", index=False, header=False)

    bbfs_all = [bbfs_earthward[0], bbfs_tailward[0]]

    for bbfs_e, bbfs_t in zip(bbfs_earthward, bbfs_tailward):
        bbfs_all[0] = bbfs_all[0].append(bbfs_e)
        bbfs_all[1] = bbfs_all[1].append(bbfs_t)

    bbfs_all[0] = bbfs_all[0].dropna(axis=0, how="any").drop_duplicates()
    bbfs_all[1] = bbfs_all[1].dropna(axis=0, how="any").drop_duplicates()

    bbfs = bbfs_all[0].append(bbfs_all[1])
    bbfs = bbfs.dropna(axis=0, how="any").drop_duplicates().sort_values(by=0)
    bbfs = bbfs.set_index(np.arange(len(bbfs)))

    return bbfs


def main(args):
    if args.which.lower() == "tints":
        bbfs = _merge_tints(args)
        bbfs.to_csv(
            os.path.join(args.out_path, "mms_bbfs_db.csv"), index=False, header=False
        )
    elif args.which.lower() == "data":
        bbfs = _merge_tints(args)
        # Load saved quantities (.nc)
        data = xr.load_dataset(os.path.join(args.inp_path, "bbfs_database_all.nc"))
        start = np.array(bbfs.values[:, 0]).astype(np.datetime64)
        stop = np.array(bbfs.values[:, 1]).astype(np.datetime64)

        with h5.File(os.path.join(args.out_path, "mms_bbfs_db.h5"), "a") as f:
            _ = f.create_dataset("start", data=pyrf.datetime642unix(start))
            _ = f.create_dataset("stop", data=pyrf.datetime642unix(stop))

            for k in data:
                _ = f.create_dataset(k, data=data[k])

    else:
        raise NotImplementedError


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("which", type=str, choices=["tints", "data"])
    parser.add_argument("inp_path", type=str)
    parser.add_argument("out_path", type=str)
    main(parser.parse_args())
