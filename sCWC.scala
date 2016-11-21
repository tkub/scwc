package jp.ac.gakushuin.cc.tk.scwc

/**
 * sCWC: Feature Selection Algorithm CWC for sparse data set
 * 
 * @author Tetsuji Kuboyama (ori-scwc@tk.cc.gakushuin.ac.jp)
 * @version 0.8.0, 2016-11-20
 * 
 */

import scopt.OptionParser
import scala.collection.mutable.{Set,OpenHashMap,ArrayBuffer,ListBuffer}
import scala.util.Sorting.{quickSort,stableSort}
import scala.runtime.ScalaRunTime.stringOf
import scala.annotation.tailrec
import math.Ordered.orderingToOrdered // for tuple comparison
import java.io.{File, PrintWriter}
import scwc._

object SortMeasure extends Enumeration {
  type SortMeasure = Value // to identify SortMeasure as SortMeasure.Value
  val MI, SU, MCC = Value
}
import SortMeasure._

object Constant {
  val PatchFeature = 'patchFeature
}
import Constant._

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

  val VERSION = "0.8.0"

  def main(args: Array[String]) {

    val parser = new OptionParser[Config]("scwc") {
      head("scwc", VERSION)
      help("help").text("prints this usage text")

      arg[String]("input_file").action{ (x, c) =>
        c.copy(inputFileName = x)
      }.text("input file in the ARFF/CSV/LIBSVM format")

      arg[String]("output_file").optional.action{ (x, c) =>
        c.copy(outputFileName = x)
      }.text("output file with extension {arff, csv, libsvm}")

      opt[String]('s',"sort").valueName("{mi|su|mcc}").action{ (x, c) =>
        val sm = x match {
          case "mi"  => MI  // mutual information
          case "su"  => SU  // symmetric uncertainty
          case "mcc" => MCC // matthew's correlation coefficient
          // case "cen" => CEN // confusion entropy
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
    // dataIO -> Table -> Data
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
    val (table, timeScan) = time { new Table(dataIO.instances) }

    println("Data preprocessing...")
    val (data, timePreprocess) = time { new Data(table, config) }

    println(s"# Number of instances: ${table.nRows}")
    if (data.rows.size < table.nRows) {
      print("# Number of instances after aggregation: ")
      println( data.rows.size+"/"+table.nRows)
    }
    println(s"# Number of features: ${table.featureSet.size}")
    if (data.sortedFeatures.size < table.featureSet.size) {
      print("# Number of features after trimming: ")
      println(s"${data.sortedFeatures.size}/${table.featureSet.size}")
    }
    if ( table.nC.keys.size == 1 ) {
      print("All instances have the same single class label...")
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
      println(selected.size+"/"+table.featureSet.size)
      if (config.verbose) {
        println
        println(selected.map{data.sortedFeatures(_).name}.mkString("{",",","}"))
      }
      dataIO.output(selected, table, data)

      dataIO.logging("#\n### Data stats\n")
      dataIO.logging(s"# Number of instances: ${table.nRows}\n")
      dataIO.logging(s"# Number of instances after aggregation: ")
      dataIO.logging(s"${data.rows.size}/${table.nRows}\n")
      dataIO.logging(s"# Number of features: |F| = ${table.featureSet.size}\n")
      dataIO.logging( "# Number of features after trimming: ")
      dataIO.logging(s"${data.sortedFeatures.size}/${table.featureSet.size}\n")
      dataIO.logging(s"# Number of class labels: |C| = ${table.nC.keys.size}\n")
      dataIO.logging(f"# H(C) = ${data.sm.hC}%.4f\n")
      dataIO.logging(s"# Data consistency: ${data.isConsistent}\n")
      dataIO.logging(s"# Number of patched instances: ${data.numOfPatchedInstances}\n")

      dataIO.logging( "#\n### Computation stats\n")
      dataIO.logging(s"# Number of consistency checks: ${fs.numOfConsistencyCheck}\n")
      dataIO.logging(f"# File read time:         $timeRead%10.4f msec\n")
      dataIO.logging(f"# File scan time:         $timeScan%10.4f msec\n")
      dataIO.logging(f"# Preprocess time:        $timePreprocess%10.4f msec\n")
      dataIO.logging(f"# Feature selection time: $timeMain%10.4f msec\n")
      dataIO.logging( "#\n### Selected features\n")
      dataIO.logging( "# Number of features selected: ")
      dataIO.logging(s"${selected.size}/${table.featureSet.size}\n")

      dataIO.logging(" MI     SU     MCC   feature\n")
      for ( fi <- selected; f = data.sortedFeatures(fi) ) {
        val vals = Seq(MI,SU,MCC).map(m => f"${data.sm(m)(f)}%6.3f").mkString(" ")
        dataIO.logging(vals+" "+f.name+"\n")
      }

      dataIO.loggingClose
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

  def output(selected: ListBuffer[Int], table: Table, data: Data) {

    def genSelectedIndices = {
      val numSelected = selected.size - (if (data.isConsistent) 0 else 1)
      val selectedIndices = new Array[Int](numSelected + 1)
      for ( i <- 0 until numSelected;
              cwcIndex = selected(i);
              attr     = data.sortedFeatures(cwcIndex) ) {
        selectedIndices(i) = table.attr2index(attr)
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
    if (outputFile.exists && (inputFileName == outputFileName || !config.overWrite)) {
      Console.err.print(s"\nscwc: ${outputFileName}: output file already exists ")
      Console.err.println("(use '-o' to overwrite)")
      sys.exit(1)
    } 
    (inputFile, outputFile)
  }
}



class Table(instances: weka.core.Instances) {

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



class Data(table: Table, config: Config) {

  val sm             = new SortMeasures(table)
  val sortedFeatures = trimAndSortFeatures(table.featureSet, sm) // i2f
  val f2i            = getReverseIndex(sortedFeatures)

  private val sortedRows    = sortEachRowByRenumFeatures(sm)
  val rows                  = aggregate(sortedRows)
  val numOfPatchedInstances = makeConsistentData(rows) // destructive to rows
  val isConsistent = (numOfPatchedInstances == 0)

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
      sortBy(v => -(sm(config.sortMeasure)(v).abs) ) // descendant order
  }

  private def sortEachRowByRenumFeatures(sm: SortMeasures): ArrayBuffer[Instance] = {
    // 1. remove feature values s.t. H(F)=0 from each row
    // 2. renumbering feature names in the order of importance
    // 3. sorting feature values (columns) by *renumbered* features
    // 4. sorting rows with lexicographical order of features
    val newRows = ArrayBuffer.empty[Instance]
    for ( (rowF, i) <- table.rowFs.zipWithIndex ) {
      val vals =
        (for ( (f, v) <- rowF zip table.rowVs(i) if sm.hF(f) > 0 ) yield 
          (f2i(f), v)).sortBy(_._1)  // renumbering by f2i
      newRows += new Instance(vals, table.classLabels(i), sortedFeatures)
      // instance is generated only here
    }
    newRows.sorted
    //quickSort(newRows)  // destructive sort
    //stableSort(newRows) // not destructive
  }

  private def makeConsistentData(rows: ArrayBuffer[Instance]): Int = {
    // instances have been already sorted and aggregated
    val patchFeatureIndex  = sortedFeatures.size
    val cLmax = table.nC.keys.max
    var count = 0
    var prev = rows(0)
    var cl = 1

    for (i <- 1 until rows.size; cur = rows(i)) {
      if (prev.classLabel < cLmax && 
          prev.classLabel != cur.classLabel &&
          prev.rowIsIdenticalTo(cur)
      ) {
        cur.row += ((patchFeatureIndex, cl))
        cl += 1
        count += 1
      } else {
        cl = 1
        prev = cur
      }
    }
    if (count > 0) { // inconsistent
      println("\n# Data is NOT consistent. Number of modified instances: "+count)
      sortedFeatures += PatchFeature // patch feature index
      f2i(PatchFeature) = patchFeatureIndex
      println(s"# Added feature '${PatchFeature.name}' at ${patchFeatureIndex}")
    }
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
    if (i == n - 1) rowsAggregated += rows(i)
    rowsAggregated
  }
}



class SortMeasures(table: Table) {

  val nRows = table.nRows
  private val log  = (b: Int) => (x: Double) => if (x>0) math.log(x)/math.log(b) else 0.0
  private val log2 = log(2)
  private val entropy = (counts: Iterable[Int]) => {
    // H = - \sum_n ( n/n_rows * log2( n/n_rows ) )
    val log2ni = log2(nRows)
    counts.map{ n => n*(log2ni - log2(n)) }.sum / nRows
  }

  val hC  = entropy(table.nC.values)         // H(C)
  val hF  = OpenHashMap.empty[Symbol,Double] // H(F)
  val hFC = OpenHashMap.empty[Symbol,Double] // H(C,F)
  val measure  = NestedMap[SortMeasure](NestedMap[Symbol](0.0))

  def apply(sm: SortMeasure) = measure(sm)

  for (f <- table.featureSet) {
    hF(f)  = entropy(table.nFV(f).values)
    hFC(f) = entropy(table.nCFV.values.flatMap(_(f).values))
    val tn = table.tn(f)
    val fn = table.fn(f)
    val fp = table.fp(f)
    val tp = table.tp(f)
    measure(MI)(f)  = hC + hF(f) - hFC(f)
    measure(SU)(f)  = 2*measure(MI)(f) / (hC + hF(f))

    // to suppress overflow
    val mccDenominator = Seq(tp+fp,tp+fn,tn+fp,tn+fn).map(math.sqrt(_)).product
    measure(MCC)(f) = if (mccDenominator == 0) 0 
                      else (tp*tn-fp*fn)/mccDenominator
    //measure(CEN)(f) = (fn+fp)*log2(nRows^2-(tp-tn)^2) / (2*nRows) - 
    //                  (fn*log2(fn)+fp*log2(fp))/nRows
  }

}



class FeatureSelection(data: Data, config: Config) {
  
  var numOfConsistencyCheck = 0
  var parWorlds = ArrayBuffer(new Instances(data.rows)) // start with single world
 
  def select() = {
    val selected = ListBuffer.empty[Int]
    var cfront = data.sortedFeatures.size - 1 
    // extract from the feature set { 0,.., cf }
    while (cfront >= 0) {
      cfront = searchConsistentFront( cfront )
      if (cfront >= 0) {
        parWorlds = divideWorldsBy( cfront )
        if (config.verbose) print( s"${cfront}(${parWorlds.size})" )
        cfront +=: selected
        cfront -= 1
      }
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
      if (instances.size == 1) {
        true
      } else {
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
      if (prev.classLabel != cur.classLabel && prev.prefixIsIdenticalUpto(cur, f)) 
        return false
      prev = cur
    }
    true
  }

  def divideBy(f: Int) = instances.groupBy{ _.getVal(f) }.values

  def p: String = instances.map(_.p).mkString("\n")
  override def toString: String = instances.map(_.toString).mkString("\n")
}



class Instance(val row: ArrayBuffer[(Int,Int)], 
               val classLabel: Int,
               features: ArrayBuffer[Symbol]) extends Ordered[Instance] {
  var freq: Int = 1

  def size = row.size

  def getVal(f: Int): Int = {
    // assume that this part is called just when dividing the worlds
    // that's why "pref" works
    @tailrec
    def binSearch(lo: Int, hi: Int): Int = {
      if (lo > hi) 0
      else {
        val mid = lo + (hi - lo) / 2
        row(mid)._1 match {
          case `f`         => row(mid)._2
          case k if k <= f => binSearch(mid + 1, hi)
          case _           => binSearch(lo, mid - 1)
        }
      }
    }
    binSearch(0, (size - 1) min f)
  }

  def rowIsIdenticalTo(other: Instance): Boolean =
    if (size == other.size) 
      ( 0 until size ).forall { i => row(i) == other.row(i) }
    else 
      false

  def prefixIsIdenticalUpto(other: Instance, f: Int): Boolean = {
    var i = 0
    var consistent = true
    val imax = size min other.size

    while ( i < imax && consistent && row(i)._1 <= f ) {
      consistent = ( row(i) == other.row(i) )
      i += 1
    }
    if      ( size > other.size &&       row(i)._1 <= f )
      false
    else if ( size < other.size && other.row(i)._1 <= f )
      false
    else
      consistent
  }

  def compare(other: Instance): Int = {
    val m = size
    val n = other.size

    @tailrec
    def cmp(i: Int): Int = i match {
      case `m` if m == n => classLabel compare other.classLabel
      case `m`           => -1 // i == m < n
      case `n`           =>  1 // m > n == i
      case _             =>
        row(i) compare other.row(i) match {
          case 0 => cmp(i + 1)
          case c => c
      }
    }
    cmp(0)
  }

  override def equals(other: Any): Boolean = other match {
    case other: Instance =>
      if (classLabel != other.classLabel) false
      else {
        if (size == other.size) 
          (0 until size).forall { i => row(i) == other.row(i) }
        else
          false
      }
    case _ => false
  }

  override def toString: String = 
    classLabel+" "+row.map{ case (f,v) => f+":"+v }.mkString(" ")+" ("+freq+")"

  def p: String =
    classLabel+" "+row.map{ case (f,v) => features(f)+":"+v }.mkString(" ")+" ("+freq+")"
}
