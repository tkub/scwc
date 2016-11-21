# scwc
sCWC: very fast feature selection for nominal data

## Quick Start

This program accepts the following options
```sh
$ sbt
> run --help
scwc 0.8.0
Usage: scwc [options] input_file [output_file]

  --help                  prints this usage text
  input_file              input file in the ARFF/CSV/LIBSVM format
  output_file             output file with extension {arff, csv, libsvm}
  -s, --sort {mi|su|mcc}  sorting measure for forward selection (default: mi)
  -v, --verbose           display selection process information
  -l, --log               output log file
  -o, --overwrite         overwrite output file
  -r, --remove <range>    attribute indices to remove e.g.) 1,3-5,8
```

Also it accepts an input file in dense/sparse ARFF, CSV, and LIBSVM
formats.  It outputs a new data file with the selected features in the
same format if no output file is given.
```sh
> run -v data/sparse.arff
```
In the above example, `data/sparse.out.arff` is generated.
When an output file is specified, the output format is given by
the file extension.
```sh
> run -s su -l data/dense.arff data/dense.csv
```
If you want to remove the 1st column, run
```sh
> run -v -l data/sample.csv -r 1
```

## Reference

* K. Shin, T. Kuboyama, T. Hashimoto, D. Shepard: Super-CWC and super-LCC: Super fast feature selection algorithms. Big Data 2015: 61-67
