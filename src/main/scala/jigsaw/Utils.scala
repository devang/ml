/* Copyright 2019 Devang Mistry. All rights reserved.
 * You may not remove this notice, but you may add
 * your name if you've made software modifications.
 * Software provided as-is. Available under the
 * terms of the AGPL v3 license available here:
 * https://www.gnu.org/licenses/agpl-3.0.en.html
 */

package jigsaw

import org.apache.spark.mllib.evaluation.{MulticlassMetrics, MultilabelMetrics}
import org.apache.spark.rdd.RDD

import scala.collection.mutable.ArrayBuffer

object Utils {

  def runMLMetrics(scoreAndLabels: RDD[(Array[Double], Array[Double])]): Unit = {
    // Instantiate metrics object
    val mlmetrics = new MultilabelMetrics(scoreAndLabels)

    // Summary stats
    println(s"Recall = ${mlmetrics.recall}")
    println(s"Precision = ${mlmetrics.precision}")
    println(s"F1 measure = ${mlmetrics.f1Measure}")
    println(s"Accuracy = ${mlmetrics.accuracy}")

    // Individual label stats
    mlmetrics.labels.foreach(label =>
      println(s"Class $label precision = ${mlmetrics.precision(label)}"))
    mlmetrics.labels.foreach(label => println(s"Class $label recall = ${mlmetrics.recall(label)}"))
    mlmetrics.labels.foreach(label => println(s"Class $label F1-score = ${mlmetrics.f1Measure(label)}"))

    // Micro stats
    println(s"Micro recall = ${mlmetrics.microRecall}")
    println(s"Micro precision = ${mlmetrics.microPrecision}")
    println(s"Micro F1 measure = ${mlmetrics.microF1Measure}")

    // Hamming loss
    println(s"Hamming loss = ${mlmetrics.hammingLoss}")

    // Subset accuracy
    println(s"Subset accuracy = ${mlmetrics.subsetAccuracy}")
  }

  def runMCMetrics(predictionAndLabels: RDD[(Double, Double)]): Unit = {
    val metrics = new MulticlassMetrics(predictionAndLabels)

    // Confusion matrix
    println("Confusion matrix:")
    println(metrics.confusionMatrix)

    // Overall Statistics
    val accuracy = metrics.accuracy
    println("Summary Statistics")
    println(s"Accuracy = $accuracy")

    // Precision by label
    val labels = metrics.labels
    labels.foreach { l =>
      println(s"Precision($l) = " + metrics.precision(l))
    }

    // Recall by label
    labels.foreach { l =>
      println(s"Recall($l) = " + metrics.recall(l))
    }

    // False positive rate by label
    labels.foreach { l =>
      println(s"FPR($l) = " + metrics.falsePositiveRate(l))
    }

    // F-measure by label
    labels.foreach { l =>
      println(s"F1-Score($l) = " + metrics.fMeasure(l))
    }

    // Weighted stats
    println(s"Weighted precision: ${metrics.weightedPrecision}")
    println(s"Weighted recall: ${metrics.weightedRecall}")
    println(s"Weighted F1 score: ${metrics.weightedFMeasure}")
    println(s"Weighted false positive rate: ${metrics.weightedFalsePositiveRate}")
  }

  def getCat(v: Int*): Int = {
    var a = 1
    var num = 0
    for (i <- v) {
      num += i*a
      a *= 2
    }
    Math.abs(num)
  }

  def getClasses(value: Double) : Array[Double] = {
    val ord = Array(1, 2, 3, 4, 5, 6)
    var array: ArrayBuffer[Double] = ArrayBuffer()

    ord.zip(value.toInt.toBinaryString.reverse).map { case (x: Int, y: Char) => if (y == '1') array += x else None }

    if (array.isEmpty)
      Array[Double](0)
    else
      array.toArray
  }

}
