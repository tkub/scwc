# scwc
sCWC: very fast feature selection for nominal data

## Quick Start

To build `scwc.jar` file in the `./bin` directory, run
```sh
$ sbt assembly
```
Command `bin/scwc` accepts a number of command line options.
```sh
$ bin/scwc --help
scwc 0.8.0
Usage: scwc [options] inputfile [outputfile]

  --help                  prints this usage text
  inputfile               input file in the ARFF/CSV/LIBSVM format
  outputfile              output file with extension {arff, csv, libsvm}
  -s, --sort {mi|su|mcc}  sorting measure for forward selection (default: mi)
  -v, --verbose           display selection process information
  -l, --log               output log file
  -o, --overwrite         overwrite output file
  -r, --remove <range>    attribute indices to remove e.g.) 1,3-5,8
```
It requires an input file in the dense/sparse ARFF, CSV, or LIBSVM
formats. Unless an output file is specified, it creates a new data
file in the same format as the input file by removing the unselected
features from the input file.
```sh
$ bin/scwc -v data/sparse.arff
```
In the above example, `data/sparse.out.arff` is created.
When an output file is specified, the output format is given by
its file extension (now `arff`, `csv`, and `libsvm` are available).
```sh
$ bin/scwc -s su -l data/dense.arff data/dense.csv
```
Also the log file `data/dense.out.log` is created by the option `-l`.
If you want to remove the 1st column in the input file, 
run with the option `-r 1`.
```sh
$ scwc -v -l data/sample.csv -r 1
```

## Reference

* K. Shin, T. Kuboyama, T. Hashimoto, D. Shepard: Super-CWC and
  super-LCC: Super fast feature selection algorithms. Big Data 2015:
  61-67
