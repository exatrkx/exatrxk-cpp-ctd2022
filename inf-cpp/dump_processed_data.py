#!/usr/bin/env python 

import os
import numpy as np
import pandas as pd
import torch

def dump(indir, outdir, num_evts):
    """take events from the indir and dump csv to the outdir"""
    nfiles = os.listdir(indir)
    print("process {} events from {}.".format(num_evts, len(nfiles)))
    
    for idx in range(num_evts):
        file = os.path.join(indir, nfiles[idx])
        data = torch.load(file)
        df = pd.DataFrame(data.x.numpy())
        outname = "evt_{}.csv".format(idx)
        df.to_csv(os.path.join(outdir, outname), header=False, index=False)


if __name__ == '__main__':
    import argparse
    parser = argparse.ArgumentParser(description='dump processed data')
    add_arg = parser.add_argument
    add_arg('indir', help='input directory')
    add_arg("outdir", help="output directory")
    add_arg('-n', "--num_evts", type=int, default=1, help="number of events to dump")
    args = parser.parse_args()

    dump(args.indir, args.outdir, args.num_evts)
    