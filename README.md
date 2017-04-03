# sCWC

sCWC: very fast feature selection for nominal data

## Quick Start

Install [sbt](http://www.scala-sbt.org/release/docs/Setup.html).
Clone or download this repository.
```
$ git clone https://github.com/tkub/scwc.git
```

To build `scwc.jar` file in the `./bin` directory, run
```
$ cd scwc
$ sbt assembly
```
Command `bin/scwc` accepts a number of command line options.
```
$ bin/scwc --help
scwc 0.8.2
Usage: scwc [options] inputfile [outputfile]

  --help                  prints this usage text
  inputfile               input file in the ARFF/CSV/LIBSVM format
  outputfile              output file with extension {arff, csv, libsvm}
  -s, --sort {mi|su|icr|mcc}  
                          sorting measure for forward selection (default: mi)
  -v, --verbose           display selection process information
  -l, --log               output log file
  -o, --overwrite         overwrite output file
  -r, --remove <range>    attribute indices to remove e.g.) 1,3-5,8
```
It requires an input file in the dense/sparse ARFF, CSV, or LIBSVM
formats. Unless an output file is specified, it creates a new data
file in the same format as the input file after removing the unselected
features from the input file.
```
$ bin/scwc -v data/sparse.arff
```
In the above example, `data/sparse.out.arff` is created.
When an output file is specified, the output format is given by
its file extension (now `arff`, `csv`, and `libsvm` are available).
```
$ bin/scwc -s su -l data/dense.arff data/dense.csv
```
Also the log file `data/dense.out.log` is created by the option `-l`.
If you want to remove the 1st column in the input file, 
run with the option `-r 1`.
```
$ scwc -v -l data/sample.csv -r 1
```

## Format of input files

You can use not only binary values {0,1} but also any nominal values,
for example, {a,b,c,d} available in the ARFF format.

If you want to use ordinal variables, use the dummy variables to
encode the values.  For example, values 1,2, and 3 can be encoded as
[0,0], [1,0] and [1,1] respectively. This encoding procedure will be
incorporated in the future version.

## Reference

* K. Shin, T. Kuboyama, T. Hashimoto, D. Shepard: Super-CWC and
  super-LCC: Super fast feature selection algorithms. Big Data 2015:
  61-67

## Acknowledgements

This work was partially supported by JSPS KAKENHI Grant [No.26280090](https://kaken.nii.ac.jp/en/grant/KAKENHI-PROJECT-26280090/)

