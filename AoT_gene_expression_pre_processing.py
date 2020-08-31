import pandas as pd
import argparse
import warnings
import sys as system
import subprocess
import os

parser = argparse.ArgumentParser(description="takes gene expression matrix and pre-processes for ML")
parser.add_argument("--expressionfile", action="store", help='Gene expression data file (rows genes, columns samples)', required=True)
parser.add_argument("--expressiontype", action="store", help='Format of gene expression values', choices=['FPKM', 'RPKM', 'TMM', 'TPM', 'Log2FC', 'COUNTS', 'OTHER'], required=True, type=str)
parser.add_argument("--Filtergenes", action="store", help='Remove genes unless they have a gene expression value over X in Y or more samples: usage --Filter-genes X Y', default=[0, 1], nargs=2)
parser.add_argument("--Filtersamples", action="store", help='Remove samples if no of genes with coverage is >x std from the mean across all samples', default=1000000)
parser.add_argument("--output", action="store", help='Processed output file name (.csv)', required=True)

args = parser.parse_args()

if args.expressiontype == "COUNTS":
    os.system('%s %s %s %s %s %s %s %s %s %s %s' % (
    "R", "--slave", "--vanilla", "--args", args.expressionfile, args.Filtergenes[0], args.Filtergenes[1],
    args.Filtersamples, args.output, "<", "Run_TMM.R"))

elif args.expressiontype == "FPKM" or args.expressiontype == "RPKM" or args.expressiontype == "TPM" or args.expressiontype == "TMM":
    os.system('%s %s %s %s %s %s %s %s %s %s %s' % (
    "R", "--slave", "--vanilla", "--args", args.expressionfile, args.Filtergenes[0], args.Filtergenes[1],
    args.Filtersamples, args.output, "<", "Run_others.R"))

elif args.expressiontype == "Log2FC" or args.expressiontype == "OTHER":
    os.system('%s %s %s %s %s %s %s %s %s %s %s' % (
    "R", "--slave", "--vanilla", "--args", args.expressionfile, args.Filtergenes[0], args.Filtergenes[1],
    args.Filtersamples, args.output, "<", "Run_LO.R"))