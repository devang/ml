/* Copyright 2019 Devang Mistry. All rights reserved.
 * You may not remove this notice, but you may add
 * your name if you've made software modifications.
 * Software provided as-is. Available under the
 * terms of the AGPL v3 license available here:
 * https://www.gnu.org/licenses/agpl-3.0.en.html
 */

package jigsaw

import org.apache.log4j.{Level, Logger}
import org.apache.spark.ml.Pipeline
import org.apache.spark.ml.classification.NaiveBayes
import org.apache.spark.ml.evaluation.MulticlassClassificationEvaluator
import org.apache.spark.ml.feature._
import org.apache.spark.ml.fpm.FPGrowth
import org.apache.spark.ml.tuning.{CrossValidator, ParamGridBuilder}
import org.apache.spark.rdd.RDD
import org.apache.spark.sql.SparkSession
import org.apache.spark.sql.functions._
import org.apache.spark.sql.types.DoubleType

case class LabeledComment(id: String, comment_text: String, toxic: Boolean, severe_toxic: Boolean, obscene: Boolean, threat: Boolean, insult: Boolean, identity_hate: Boolean)
case class TestComment(id: String, comment_text: String)
case class LabeledTestComment(id: String, toxic: Boolean, severe_toxic: Boolean, obscene: Boolean, threat: Boolean, insult: Boolean, identity_hate: Boolean)

object JigsawNB {

  def jigsaw(): Unit = {
    val spark = SparkSession.builder.config("spark.executor.memory", "8G").master("local[*]").getOrCreate()

    Logger.getLogger("org").setLevel(Level.ERROR)
    Logger.getLogger("akka").setLevel(Level.ERROR)

    import spark.implicits._

    val jigsawDF = spark.read.format("csv")
      .option("mode","FAILFAST")
      .option("escape", "\"")
      .option("multiLine", "true")
      .option("inferSchema", "true")
      .option("header", "true")
      .load("/kaggle/input/jigsaw-toxic-comment-classification-challenge/train.csv")
      .as[LabeledComment]
    println(jigsawDF.show(2))
    jigsawDF.describe().show()
    jigsawDF.dtypes.iterator

    val jigsawTestCommentDF = spark.read.format("csv")
      .option("mode","FAILFAST")
      .option("escape", "\"")
      .option("multiLine", "true")
      .option("inferSchema", "true")
      .option("header", "true")
      .load("/kaggle/input/jigsaw-toxic-comment-classification-challenge/test.csv")
      .as[TestComment]
    println(jigsawTestCommentDF.show(2))

    val jigsawTestLabelsDF = spark.read.format("csv")
      .option("mode","FAILFAST")
      .option("escape", "\"")
      .option("multiLine", "true")
      .option("inferSchema", "true")
      .option("header", "true")
      .load("/kaggle/input/jigsaw-toxic-comment-classification-challenge/test_labels.csv")
      .as[LabeledTestComment]
    println(jigsawTestLabelsDF.show(2))

    val jigsawTestDF = jigsawTestCommentDF.join(jigsawTestLabelsDF, "id")
      .filter($"toxic" =!= -1)

    //sanity checks
    jigsawDF.filter($"comment_text".isNull).show()
    jigsawTestDF.filter($"comment_text".isNull).show()

    jigsawTestDF.show(2)

    jigsawDF.describe("id", "toxic", "severe_toxic", "obscene").show()
    jigsawDF.describe("id", "threat", "insult", "identity_hate").show()

    val classes = Array("toxic", "severe_toxic", "obscene", "threat", "insult", "identity_hate")

    var dist:Map[String, Long] = Map()
    for (cl <- classes) {
      dist += (cl -> jigsawDF.filter(col(cl) === 1).count())
    }
    dist.foreach(println)
    dist.keySet.toArray.apply(0)

    var testDist:Map[String, Long] = Map()
    for (cl <- classes) {
      testDist += (cl -> jigsawTestDF.filter(col(cl) === 1).count())
    }
    testDist.foreach(println)

    jigsawDF.filter($"comment_text" === "").show()
    jigsawDF.filter($"id" === "00078f8ce7eb276d").describe()


    val rulesdataset = jigsawDF.select("toxic", "severe_toxic", "obscene", "threat", "insult", "identity_hate")
      .map(row => Utils.getClasses(Utils.getCat(row.getAs[Int](0), row.getAs[Int](1), row.getAs[Int](2),
                                      row.getAs[Int](3), row.getAs[Int](4), row.getAs[Int](5)))).toDF("items")
      .filter($"items" =!= Array[Double](0))

    val fpgrowth = new FPGrowth().setItemsCol("items").setMinSupport(0).setMinConfidence(0.8)
    val model = fpgrowth.fit(rulesdataset)

    // Display frequent itemsets.
    model.freqItemsets.show(1000,truncate = false)

    // Display generated association rules.
    model.associationRules.show(100,truncate = false)

    val tokenizer = new Tokenizer()
      .setInputCol("comment_text")
      .setOutputCol("words")
    //https://github.com/stanfordnlp/CoreNLP/blob/master/data/edu/stanford/nlp/patterns/surface/stopwords.txt
    val remover = new StopWordsRemover()
      .setInputCol("words")
      .setOutputCol("filtered")
//      .setStopWords(read.lines! wd/"")
    val hashingTF = new HashingTF().setInputCol("filtered").setOutputCol("rawFeatures")
    val idf = new IDF().setInputCol("rawFeatures").setOutputCol("features")
    val lp = new LabelPowerset()
      .setInputCol("features")
      .setLabelInputCols("toxic,severe_toxic,obscene,threat,insult,identity_hate")
      .setLabelOutputCol("label")
    val nb = new NaiveBayes()



    val pipeline = new Pipeline().setStages(Array(tokenizer, remover, hashingTF, idf, lp, nb))

    val paramGrid = new ParamGridBuilder().build()

    val cvmodel = new CrossValidator()
      .setEstimator(pipeline)
      .setEvaluator(new MulticlassClassificationEvaluator)
      .setEstimatorParamMaps(paramGrid)
      .setNumFolds(5)
      .setParallelism(8)

    val bestcvmodel = cvmodel.fit(jigsawDF)
    val predictions = bestcvmodel.transform(jigsawTestDF)

    val predictionAndLabels: RDD[(Double, Double)] = predictions.select("label", "prediction")
      .withColumn("label", $"label".cast(DoubleType)).withColumn("prediction", $"prediction".cast(DoubleType))
      .rdd.map(row => (row.getAs[Double]("prediction"), row.getAs[Double]("label")))

    val scoreAndLabels: RDD[(Array[Double], Array[Double])] = predictionAndLabels.map(
      row => getMultiLabel(row._1, row._2)
    )

    Utils.runMLMetrics(scoreAndLabels)
//    Utils.runMCMetrics(predictionAndLabels)

    bestcvmodel.write.overwrite.save("jigsaw.spark")

    None
  }

  def getMultiLabel(prediction: Double, label: Double): (Array[Double], Array[Double]) = {
    (Utils.getClasses(prediction), Utils.getClasses(label))
  }

}
