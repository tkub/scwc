package jp.ac.gakushuin.cc.tk.scwc

/**
 * sCWC: Feature Selection Algorithm CWC for sparse data set
 * 
 * @author Tetsuji Kuboyama <ori-scwc@tk.cc.gakushuin.ac.jp>
 * @version 0.8.3, 2018-01-03
 * 
 * License: http://www.apache.org/licenses/LICENSE-2.0 Apache-2.0
 * 
 * Acknowledgements:
 * This work was partially supported by JSPS KAKENHI Grant No. 26280090
 */

import scopt.OptionParser
import scala.collection.mutable.{Set,OpenHashMap,ArrayBuffer,ListBuffer}
import scala.annotation.tailrec
import math.Ordered.orderingToOrdered // for tuple comparison
import java.io.{File, PrintWriter}
import utils._

object SortMeasure extends Enumeration {
  type SortMeasure = Value // to identify SortMeasure as SortMeasure.Value
  val MI, SU, ICR, MCC = Value
}
import SortMeasure._

case class Config(
  inputFileName:  String      = "",
  outputFileName: String      = "",
  sortMeasure:    SortMeasure = MI,
  verbose:        Boolean     = false,
  overWrite:      Boolean     = false,
  log:            Boolean     = false,
  removeIndices:  String      = ""
  // parallel:      Boolean     = false
)


object Main {

  val VERSION = "0.8.3"

  def main(args: Array[String]) {

    val parser = new OptionParser[Config]("scwc") {
      head("scwc", VERSION)
      help("help").text("prints this usage text")

      arg[String]("inputfile").action{ (x, c) =>
        c.copy(inputFileName = x)
      }.text("input file in the arff/csv/libsvm format")

      arg[String]("outputfile").optional.action{ (x, c) =>
        c.copy(outputFileName = x)
      }.text("output file with extension {arff, csv, libsvm}")

      opt[String]('s',"sort").valueName("{mi|su|icr|mcc}").action{ (x, c) =>
        val sm = x match {
          case "mi"  => MI  // mutual information
          case "su"  => SU  // symmetric uncertainty
          case "icr" => ICR // inconsistency rate (Bayesian risk)
          case "mcc" => MCC // matthew's correlation coefficient
        }
        c.copy(sortMeasure = sm)
      }.text("sorting measure for forward selection (default: mi)")

      opt[Unit]('v', "verbose").action{ (_, c) =>
        c.copy(verbose = true)
      }.text("display selection process information")

      opt[Unit]('l', "log").action{ (_, c) =>
        c.copy(log = true)
      }.text("output log file")

      opt[Unit]('o', "overwrite").action{ (_, c) =>
        c.copy(overWrite = true)
      }.text("overwrite output file")

      opt[String]('r', "remove").valueName("<range>").
        action{ (x, c) => c.copy(removeIndices = x)
      }.text("attribute indices to remove e.g.) 1,3-5,8")

//    opt[Unit]('p', "parallel").action{ (_, c) =>
//      c.copy(parallel = true)
//    }.text("Parallelize consistency check")
    }

    parser.parse(args, Config()) match {
      case Some(config) => featureSelection(config)
      case None         => println(parser.usage)
    }
  }

  def featureSelection(config: Config) {
    // dataIO -> Stat -> Data
    println("Reading input file...")
    val (dataIO, timeRead) = time { new DataIO(config) }

    dataIO.logging(s"# scwc version $VERSION log\n")
    dataIO.logging(s"# Time: ${java.util.Calendar.getInstance.getTime}\n")
    dataIO.logging("#\n### Configuration\n")
    dataIO.logging(s"# Input file:  ${config.inputFileName}\n")
    dataIO.logging(s"# Output file: ${dataIO.outputFileName}\n")
    dataIO.logging(s"# Sort measure: ${config.sortMeasure}\n")
    dataIO.logging(s"# Removed indices: ${config.removeIndices}\n")

    println("Scanning input file...")
    val (stat, timeScan) = time { new Stat(dataIO.instances) }

    println("Data preprocessing...")
    val (data, timePreprocess) = time { new Data(stat, config) }

    println(s"# Number of instances: ${stat.nRows}")
    if (data.rows.size < stat.nRows) {
      print("# Number of instances after aggregation: ")
      println( data.rows.size+"/"+stat.nRows)
    }
    if (data.numOfMinorInstances>0) {
      println("# Data is NOT consistent.")
      println("# Number of deleted instances to make consistent: "+
               data.numOfMinorInstances)
    }
    println(s"# Number of features: ${stat.featureSet.size}")
    if (data.sortedFeatures.size < stat.featureSet.size) {
      print("# Number of features after trimming: ")
      println(s"${data.sortedFeatures.size}/${stat.featureSet.size}")
    }
    if ( stat.nC.keys.size == 1 ) {
      print("All instances have the same single class label...Nothing to do")
      // nothing to do. selected features = emptyset
    } else {
      val fs = new FeatureSelection(data, config)
      print("Selecting features...")
      if (config.verbose) println
      val (selected, timeMain) = time { fs.select() }
      println
      if (config.verbose)
        println(s"# Number of consistency checks: ${fs.numOfConsistencyCheck}")
      print("# Number of features selected: ")
      println(selected.size+"/"+stat.featureSet.size)
      if (config.verbose) {
        println
        println(selected.map{data.sortedFeatures(_).name}.mkString("{",",","}"))
      }
      dataIO.logging("#\n### Data stats\n")
      dataIO.logging(s"# Number of instances: ${stat.nRows}\n")
      dataIO.logging(s"# Number of instances after aggregation: ")
      dataIO.logging(s"${data.rows.size}/${stat.nRows}\n")
      dataIO.logging(s"# Number of features: |F| = ${stat.featureSet.size}\n")
      dataIO.logging( "# Number of features after trimming: ")
      dataIO.logging(s"${data.sortedFeatures.size}/${stat.featureSet.size}\n")
      dataIO.logging(s"# Number of class labels: |C| = ${stat.nC.keys.size}\n")
      dataIO.logging(f"# H(C) = ${data.sm.hC}%.4f\n")
      dataIO.logging(s"# Data consistency: ${data.isConsistent}\n")
      dataIO.logging(s"# Number of deleted instances: ${data.numOfMinorInstances}\n")

      dataIO.logging( "#\n### Computation stats\n")
      dataIO.logging(s"# Number of consistency checks: ${fs.numOfConsistencyCheck}\n")
      dataIO.logging(f"# File read time:         $timeRead%10.4f msec\n")
      dataIO.logging(f"# File scan time:         $timeScan%10.4f msec\n")
      dataIO.logging(f"# Preprocess time:        $timePreprocess%10.4f msec\n")
      dataIO.logging(f"# Feature selection time: $timeMain%10.4f msec\n")
      dataIO.logging( "#\n### Selected features\n")
      dataIO.logging( "# Number of features selected: ")
      dataIO.logging(s"${selected.size}/${stat.featureSet.size}\n")
      val measures = SortMeasure.values.toSeq
      dataIO.logging(measures.map( s => f"$s%6s" ).mkString(" ")+
                     " feature\n")
      for ( fi <- selected; f = data.sortedFeatures(fi) ) {
      // for ( f <- stat.featureSet ) { // outpus all features
        val vals = measures.map(m =>f"${data.sm(m)(f)}%6.3f").mkString(" ")
        dataIO.logging(vals+" "+f.name+"\n")
      }
      dataIO.loggingClose

      dataIO.output(selected, stat, data)

      sys.exit(0)
    }
  }
}

class DataIO(config: Config) {

  import weka.core.converters.{ArffLoader,CSVLoader,LibSVMLoader}
  import weka.core.converters.{ArffSaver, CSVSaver, LibSVMSaver}
  import weka.filters.unsupervised.attribute.Remove
  import weka.filters.Filter

  val inputFileName  = config.inputFileName
  val Array(fileBaseName, inputFileExt) = inputFileName.split("\\.(?=[^.]+$)")

  val outputFileName = if (config.outputFileName.isEmpty) {
                         fileBaseName + ".out." + inputFileExt
                       } else {
                         config.outputFileName
                       }
  val Array(outputFileBaseName, outputFileExt) = outputFileName.split("\\.(?=[^.]+$)")

  val logFileName = outputFileBaseName + ".log"

  val (inputFile, outputFile) = checkFiles(inputFileName, outputFileName)
  val log = if (config.log) new PrintWriter(new File(logFileName)) else null
  val instances = readFile(inputFile, inputFileExt)

  def logging(msg: String) {
    if (config.log) log.print(msg)
  }
  def loggingClose {
    if (config.log) log.close
  }

  def output(selected: ListBuffer[Int], stat: Stat, data: Data) {

    def genSelectedIndices = {
      val numSelected = selected.size - (if (data.isConsistent) 0 else 1)
      val selectedIndices = new Array[Int](numSelected + 1)
      for ( i <- 0 until numSelected;
              cwcIndex = selected(i);
              attr     = data.sortedFeatures(cwcIndex) ) {
        selectedIndices(i) = stat.attr2index(attr)
      }
      val classIndex = instances.numAttributes - 1
      selectedIndices(numSelected) = classIndex
      selectedIndices
    }

    val selectedIndices = genSelectedIndices
    val filter = new Remove
    filter.setAttributeIndicesArray(selectedIndices)
    filter.setInvertSelection(true)
    filter.setInputFormat(instances)
    val newInstances = Filter.useFilter(instances, filter)

    var saver = outputFileExt match {
      case "csv"    => new CSVSaver
      case "arff"   => new ArffSaver
      case "libsvm" => new LibSVMSaver
      case ext      =>
        Console.err.println(s"\nUnknown output file type '${ext}'")
        sys.exit(1)
    }
    println(s"scwc: ${outputFileName}: output selected features")

    saver.setInstances(newInstances)
    saver.setFile(outputFile)
    saver.writeBatch

  }

  private def readFile(inputFile: File, inputFileExt: String) = {

    var loader = inputFileExt match {
      case "csv"    => new CSVLoader
      case "arff"   => new ArffLoader
      case "libsvm" => new LibSVMLoader
      case ext      =>
        Console.err.println(s"\nUnknown input file type '${ext}'")
        sys.exit(1)
    }

    loader.setSource(inputFile)
    var instances = loader.getDataSet
    if (!config.removeIndices.isEmpty) {
      val filter = new Remove
      filter.setOptions(Array("-R",config.removeIndices))
      filter.setInputFormat(instances)
      instances = Filter.useFilter(instances, filter)
    }
    instances
  }

  private def checkFiles(inputFileName: String, outputFileName: String) = {
    val inputFile  = new File(inputFileName)
    val outputFile = new File(outputFileName)

    if (!inputFile.exists) {
      Console.err.println(s"\nscwc: ${inputFileName}: No such file exists")
      sys.exit(1)
    }
    if (outputFile.exists && 
                     (inputFileName == outputFileName || !config.overWrite)) {
      Console.err.print(s"\nscwc: ${outputFileName}: output file already exists ")
      Console.err.println("(use '-o' to overwrite)")
      sys.exit(1)
    } 
    (inputFile, outputFile)
  }
}

class Stat(instances: weka.core.Instances) {

  val featureSet  = Set.empty[Symbol]
  val classLabels = ArrayBuffer.empty[Int]
  val rowFs       = ArrayBuffer.empty[ArrayBuffer[Symbol]]
  val rowVs       = ArrayBuffer.empty[ArrayBuffer[Int]]
  val nFV         = NestedMap[Symbol](NestedMap[Int](0))
  val nCFV        = NestedMap[Int](NestedMap[Symbol](NestedMap[Int](0)))
  val nC          = NestedMap[Int](0)
  val tp,tn,fp,fn = NestedMap[Symbol](0)
  val attr2index  = getAttr2index

  tabulateData(instances)
  val nRows = instances.numInstances

  private def getAttr2index = {
    val attr2index = OpenHashMap.empty[Symbol,Int]
    for (idx <- 0 until instances.numAttributes;
      attr = Symbol(instances.attribute(idx).name)) {
      attr2index(attr) = idx
    }
    attr2index
  }

  private def tabulateData(instances: weka.core.Instances) {

    val numInstances  = instances.numInstances
    val numAttrs      = instances.numAttributes
    val enumInstances = instances.enumerateInstances

    while (enumInstances.hasMoreElements) {
      val instance = enumInstances.nextElement
      val rowF = ArrayBuffer.empty[Symbol]
      val rowV = ArrayBuffer.empty[Int]

      val classIndex = numAttrs - 1
      val classLabel = instance.value(classIndex).toInt
      nC(classLabel) += 1
      classLabels += classLabel

      for ( i <- 0 until instance.numValues;
        idx = instance.index(i); v = instance.value(idx).toInt
        if (v != 0) && (idx < classIndex)
      ) {
        val attr  = Symbol(instance.attribute(idx).name)
        val value = instance.value(idx).toInt
        featureSet += attr
        nFV(attr)(value) += 1
        nCFV(classLabel)(attr)(value) += 1
        rowF += attr
        rowV += value
      }
      rowFs += rowF
      rowVs += rowV
    }

    // complete zero feature values for sparse format
    for ( f <- featureSet ) {
      nFV(f)(0) = numInstances - nFV(f).values.sum
      for ( (classLabel, freq) <- nC ) {
        nCFV(classLabel)(f)(0) = freq - nCFV(classLabel)(f).values.sum
      }
      //  f \ c |   0      v
      //  ----------------------
      //    0   |  (tn)  (fn)
      //    v   |   fp    tp
      //  ----------------------
      //          nC(0) nC(v)
      tn(f) = nCFV(0)(f)(0)
      fn(f) = nCFV.keys.filter(_>0).map(nCFV(_)(f)(0)).sum
      fp(f) = nC(0) - tn(f)
      tp(f) = numInstances - tn(f) - fn(f) - fp(f)
    }
  }
}



class Data(stat: Stat, config: Config) {

  val sm             = new SortMeasures(stat)
  val sortedFeatures = trimAndSortFeatures(stat.featureSet, sm) // i2f
  val f2i            = getReverseIndex(sortedFeatures)

  private val sortedRows  = sortEachRowByRenumFeatures(sm)
  val rows                = aggregate(sortedRows)
  val numOfMinorInstances = makeConsistentData(rows) // destructive to rows
  val isConsistent = (numOfMinorInstances == 0)

  private def getReverseIndex(features: ArrayBuffer[Symbol]) = {
    val f2i = OpenHashMap.empty[Symbol,Int]
    for ( (f, i) <- features.zipWithIndex ) f2i(f)=i
    f2i
  }

  private def trimAndSortFeatures(featureSet: Set[Symbol], 
                                          sm: SortMeasures): ArrayBuffer[Symbol] = {
    // 1. remove features s.t. H(F)=0 from featureSet
    // 2. return features sorted by a specific sort measure
    featureSet.filter(sm.hF(_)>0).to[ArrayBuffer].
      sortBy( v => -(sm(config.sortMeasure)(v).abs) ) // descendant order
  }

  private def sortEachRowByRenumFeatures(sm: SortMeasures): ArrayBuffer[Instance] = {
    // 1. remove feature values s.t. H(F)=0 from each row
    // 2. renumbering feature names in the order of importance
    // 3. sorting feature values (columns) by *renumbered* features
    // 4. sorting rows with lexicographical order of features
    val newRows = ArrayBuffer.empty[Instance]
    for ( (rowF, i) <- stat.rowFs.zipWithIndex ) {
      val vals =
        (for ( (f, v) <- rowF zip stat.rowVs(i) if sm.hF(f) > 0 ) yield 
          (f2i(f), v)).sortBy(_._1)  // renumbering by f2i
      newRows += new Instance(vals, stat.classLabels(i)) //, sortedFeatures)
      // instance is generated only here
    }
    newRows.sorted // ** to be revised by Radix Sort **
  }

  private def makeConsistentData(rows: ArrayBuffer[Instance]): Int = {
    // instances have been already SORTED and AGGREGATED
    val labelMax = stat.nC.keys.max
    val minorInstances = ArrayBuffer[Int]()
    var count = 0
    var j = 0
    var prev = rows(j)

    for (i <- 1 until rows.size; cur = rows(i)) {
      if (prev.classLabel < labelMax &&
          prev.classLabel != cur.classLabel &&
          prev.vals == cur.vals
      ) {
        // 'cur' is inconsistent with 'prev'
        if (prev.freq < cur.freq) {
          minorInstances += j
          prev = cur; j = i
        } else {
          minorInstances += i
        }
        count += 1
      } else {
        prev = cur
        j = i
      }
    }
    minorInstances.foreach { i => rows.remove(i) }
    count
  }

  private def aggregate(rows: ArrayBuffer[Instance]) = {
    val n = rows.size
    var i = 0
    val rowsAggregated = ArrayBuffer.empty[Instance]

    for ( j <- 1 until n ) {
      if (rows(i) == rows(j)) {
        rows(i).freq += 1
      } else {
        rowsAggregated += rows(i)
        i = j
      }
    }
    rowsAggregated += rows(i)
    rowsAggregated
  }
}



class SortMeasures(s: Stat) {

  val nRows = s.nRows.toDouble
  private val log  = (b: Int) => (x: Double) =>
                     if (x>0) math.log(x)/math.log(b) else 0.0
  private val log2 = log(2)
  private val entropy = (counts: Iterable[Int]) => {
    // H = - \sum_n ( n/n_rows * log2( n/n_rows ) )
    val log2ni = log2(nRows)
    counts.map{ n => n*(log2ni - log2(n)) }.sum / nRows
  }

  val hC  = entropy(s.nC.values)             // H(C)
  val hF  = OpenHashMap.empty[Symbol,Double] // H(F)
  val hFC = OpenHashMap.empty[Symbol,Double] // H(C,F)
  val measure  = NestedMap[SortMeasure](NestedMap[Symbol](0.0))

  def apply(sm: SortMeasure) = measure(sm)

  for (f <- s.featureSet) {
    hF(f)  = entropy(s.nFV(f).values)
    hFC(f) = entropy(s.nCFV.values.flatMap(_(f).values))
    val tn = s.tn(f)
    val fn = s.fn(f)
    val fp = s.fp(f)
    val tp = s.tp(f)
    measure(MI)(f)  = hC + hF(f) - hFC(f)
    measure(SU)(f)  = 2*measure(MI)(f) / (hC + hF(f))
    // complement of ICR (Bayesian Risk)
    measure(ICR)(f) = 1 - s.nFV(f).map{ case (v,freq) 
                         => freq - s.nCFV.keys.map{ s.nCFV(_)(f)(v) }.max
                       }.sum / nRows
    // to avoid overflow
    val mccDenominator = Seq(tp+fp,tp+fn,tn+fp,tn+fn).map(math.sqrt(_)).product
    measure(MCC)(f) = if (mccDenominator == 0) 0
                      else (tp*tn-fp*fn)/mccDenominator
  }

}



class FeatureSelection(data: Data, config: Config) {
  
  var numOfConsistencyCheck = 0
  var parWorlds = ArrayBuffer(new Instances(data.rows)) // start with single world
 
  def select() = {
    val selected = ListBuffer.empty[Int]
    var cfront = data.sortedFeatures.size - 1 
    // extract from the feature set { 0,.., cf }
    while ( cfront >= 0 ) {
      cfront = searchConsistentFront( cfront )
      parWorlds = divideWorldsBy( cfront )
      if (config.verbose) print( s"${cfront}(${parWorlds.size})" )
      cfront +=: selected
      cfront -= 1
    }
    if (config.verbose) print("\n...")
    selected
  }

  private def isConsistentUpto(f: Int): Boolean =  {
    // assuming that each parWorld includes more than one instance
    if (config.verbose) print(".")
    numOfConsistencyCheck += 1
    //if (config.parallel && parWorlds.size >= 2)
    //  parWorlds.par.forall(_.isConsistentUpto(f))
    //else
    parWorlds.forall(_.isConsistentUpto(f))
  }

  private def searchConsistentFront(f: Int): Int = {
    // assuming that [0, f] is consistent in each parWorld
    @tailrec 
    def binSearch(lo: Int, hi: Int): Int = {
      if ( lo >= hi ) 
        lo // lo is the least f for which isConsistentUpto(f) is true
      else {
        val mid = lo + (hi - lo) / 2
        if ( isConsistentUpto(mid) )
          binSearch(lo, mid)
        else
          binSearch(mid + 1, hi)
      }
    }
    // linear search:  (-1 to f).find(isConsistentUpto(_))
    if (0 < f && !isConsistentUpto(f - 1)) {
      f // Do not use this part for inconsistent data
    } else {
      binSearch(0, f)
      /*
       val consistentFront = binSearch(0, f)
       //this happens if data is inconsistent
       if (!isConsistentUpto(consistentFront)) {
       println("\n\n####### Warning: This data is inconsistent!")
       -1
       } else {
       consistentFront
      }
     */
    }
  }

  private def divideWorldsBy(f: Int) = {

    def hasSingleClass(instances: ArrayBuffer[Instance]): Boolean = {
      (instances.size == 1) || {
        val cl = instances(0).classLabel
          (1 until instances.size).forall( cl == instances(_).classLabel )
      }
    }

    for (instances <- parWorlds; wd <- instances.divideBy(f)
      if !hasSingleClass(wd) ) yield new Instances(wd)
  }
}


class Instances(instances: ArrayBuffer[Instance]) {
  // assuming that instances are sorted and aggregated
  def size = instances.size
  def apply(i: Int) = instances(i)

  def isConsistentUpto(f: Int): Boolean = {
    var prev = instances(0)
    for ( i <- 1 until size; cur = instances(i) ) {
      if (prev.classLabel != cur.classLabel && 
          prev.prefixIsIdenticalUpto(cur, f))
        return false
      prev = cur
    }
    true
  }

  def divideBy(f: Int) = instances.groupBy{ _.getVal(f) }.values

  def p: String = instances.map(_.p).mkString("\n")
  override def toString: String = instances.map(_.toString).mkString("\n")
}



class Instance(val vals: ArrayBuffer[(Int,Int)], 
               val classLabel: Int) extends Ordered[Instance] {
  // vals = ArrayBuffer((#f1, v1), ..., (#f_n, v_n))
  var freq: Int = 1

  def size = vals.size

  def getVal(f: Int): Int = {
    @tailrec
    def binSearch(lo: Int, hi: Int): Int = {
      if (lo > hi) 0
      else {
        val mid = lo + (hi - lo) / 2
        vals(mid)._1 match {
          case `f`         => vals(mid)._2
          case k if k <= f => binSearch(mid + 1, hi)
          case _           => binSearch(lo, mid - 1)
        }
      }
    }
    binSearch(0, (size - 1) min f)
  }

  def prefixIsIdenticalUpto(other: Instance, f: Int): Boolean = {
    val m = size
    val n = other.size

    @tailrec
    def identical(i: Int): Boolean = i match {
      case `m` if m == n => true
      case `m` => other.vals(i)._1 > f // i == m < n
      case `n` =>       vals(i)._1 > f // m > n == i
      case _   =>
        val (f1, v1) = vals(i)
        val (f2, v2) = other.vals(i)
        if (f1 == f2 && f1 <= f && v1 == v2)
          identical(i+1)
        else
          f1 > f && f2 > f
    }
    identical(0)
  }

  def compare(other: Instance): Int = {
    val m = size
    val n = other.size

    @tailrec
    def cmp(i: Int): Int = i match {
      case `m` if m == n => classLabel compare other.classLabel
      case `m` => -1 // i == m < n
      case `n` =>  1 // m > n == i
      case _   => 
        // note that (0,1) > (1,1)       in sparse form
        // because   (0,1) > (0,0),(1,1) in dense form
        val (f1, v1) = vals(i)
        val (f2, v2) = other.vals(i)
        if (f1 == f2) {
          if (v1 == v2) cmp(i+1) else v1 compare v2
        } else {
          f2 compare f1
        }
    }
    cmp(0)
  }

  override def equals(other: Any): Boolean = other match {
    case other: Instance =>
      if (classLabel != other.classLabel) false
      else
        vals == other.vals
    case _ => false
  }

  override def toString: String = 
    classLabel+" "+vals.map{ case (f,v) => f+":"+v }.mkString(" ")+" ("+freq+")"

  def p: String =
    // classLabel+" "+vals.map{ case (f,v) => features(f)+":"+v }.mkString(" ")+" ("+freq+")"
    classLabel+" "+vals.map{ case (f,v) => f+":"+v }.mkString(" ")+" ("+freq+")"
}
