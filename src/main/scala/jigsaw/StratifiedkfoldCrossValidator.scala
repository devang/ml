/* Copyright 2019 Devang Mistry. All rights reserved.
 * You may not remove this notice, but you may add
 * your name if you've made software modifications.
 * Software provided as-is. Available under the
 * terms of the AGPL v3 license available here:
 * https://www.gnu.org/licenses/agpl-3.0.en.html
 */

package jigsaw

import java.util.concurrent.Executors

import org.apache.spark.internal.Logging
import org.apache.spark.ml.evaluation.Evaluator
import org.apache.spark.ml.param._
import org.apache.spark.ml.param.shared.{HasCollectSubModels, HasSeed}
import org.apache.spark.ml.util.{Identifiable, MLWritable, MLWriter}
import org.apache.spark.ml.{Estimator, Model}
import org.apache.spark.mllib.util.MLUtils
import org.apache.spark.sql.types.StructType
import org.apache.spark.sql.{DataFrame, Dataset}

import scala.concurrent.duration.Duration
import scala.concurrent.{Await, ExecutionContext, Future}

class StratifiedkfoldCrossValidator
  extends Estimator[StratifiedkfoldCrossValidatorModel]
  with HasCollectSubModels with MLWritable with HasSeed with Params with Logging {

  val numFolds: IntParam = new IntParam(this, "numFolds",
    "number of folds for cross validation (>= 2)", ParamValidators.gtEq(2))

  val estimator: Param[Estimator[_]] = new Param(this, "estimator", "estimator for selection")

  val evaluator: Param[Evaluator] = new Param(this, "evaluator",
    "evaluator used to select hyper-parameters that maximize the validated metric")

  val estimatorParamMaps: Param[Array[ParamMap]] =
    new Param(this, "estimatorParamMaps", "param maps for the estimator")

  val parallelism = new IntParam(this, "parallelism",
    "the number of threads to use when running parallel algorithms", ParamValidators.gtEq(1))

  setDefault(parallelism -> 1)
  setDefault(numFolds -> 3)

  def getParallelism: Int = $(parallelism)
  def getEstimatorParamMaps: Array[ParamMap] = $(estimatorParamMaps)
  def getEstimator: Estimator[_] = $(estimator)
  def getNumFolds: Int = $(numFolds)
  def getEvaluator: Evaluator = $(evaluator)


  override def fit(dataset: Dataset[_]): StratifiedkfoldCrossValidatorModel = {
    val schema = dataset.schema
    transformSchema(schema)
    val sparkSession = dataset.sparkSession
    val est = $(estimator)
    val eval = $(evaluator)
    val epm = $(estimatorParamMaps)

    logInfo(this.toString())
    logInfo(dataset.toString())
    logInfo(numFolds.toString())
    logInfo(seed.toString())
    logInfo(parallelism.toString())
    logInfo(estimator.toString())
    logInfo(evaluator.toString())
    logInfo(estimatorParamMaps.toString())

    val collectSubModelsParam = $(collectSubModels)

    var subModels: Option[Array[Array[Model[_]]]] = if (collectSubModelsParam) {
      Some(Array.fill($(numFolds))(Array.fill[Model[_]](epm.length)(null)))
    } else None

    // Compute metrics for each model over each split
    val splits = MLUtils.kFold(dataset.toDF.rdd, $(numFolds), $(seed))
    val metrics = splits.zipWithIndex.map { case ((training, validation), splitIndex) =>
      val trainingDataset = sparkSession.createDataFrame(training, schema).cache()
      val validationDataset = sparkSession.createDataFrame(validation, schema).cache()
      logDebug(s"Train split $splitIndex with multiple sets of parameters.")

      // Fit models in a Future for training in parallel
      val foldMetricFutures = epm.zipWithIndex.map { case (paramMap, paramIndex) =>
        Future[Double] {
          val model = est.fit(trainingDataset, paramMap).asInstanceOf[Model[_]]
          if (collectSubModelsParam) {
            subModels.get(splitIndex)(paramIndex) = model
          }
          // TODO: duplicate evaluator to take extra params from input
          val metric = eval.evaluate(model.transform(validationDataset, paramMap))
          logDebug(s"Got metric $metric for model trained with $paramMap.")
          metric
        }(ExecutionContext.fromExecutorService(Executors.newCachedThreadPool()))
      }

      // Wait for metrics to be calculated
      val foldMetrics = foldMetricFutures.map(Await.result(_, Duration.Inf))

      // Unpersist training & validation set once all metrics have been produced
      trainingDataset.unpersist()
      validationDataset.unpersist()
      foldMetrics
    }.transpose.map(_.sum / $(numFolds)) // Calculate average metric over all splits

    logInfo(s"Average cross-validation metrics: ${metrics.toSeq}")
    val (bestMetric, bestIndex) =
      if (eval.isLargerBetter) metrics.zipWithIndex.maxBy(_._1)
      else metrics.zipWithIndex.minBy(_._1)
    logInfo(s"Best set of parameters:\n${epm(bestIndex)}")
    logInfo(s"Best cross-validation metric: $bestMetric.")
    val bestModel = est.fit(dataset, epm(bestIndex)).asInstanceOf[Model[_]]
    copyValues(new StratifiedkfoldCrossValidatorModel(uid, bestModel, metrics)
      .setSubModels(subModels))
  }

  override def copy(extra: ParamMap): Estimator[StratifiedkfoldCrossValidatorModel] = {
    this
  }

  override def write: MLWriter = {
    new MLWriter {
      override protected def saveImpl(path: String): Unit = ()
    }
  }

  override val uid: String = Identifiable.randomUID("skcv")

  override def transformSchema(schema: StructType): StructType = transformSchemaImpl(schema)

  protected def transformSchemaImpl(schema: StructType): StructType = {
    require($(estimatorParamMaps).nonEmpty, s"Validator requires non-empty estimatorParamMaps")
    val firstEstimatorParamMap = $(estimatorParamMaps).head
    val est = $(estimator)
    for (paramMap <- $(estimatorParamMaps).tail) {
      est.copy(paramMap).transformSchema(schema)
    }
    est.copy(firstEstimatorParamMap).transformSchema(schema)
  }

}

class StratifiedkfoldCrossValidatorModel(
  val uid: String,
  val bestModel: Model[_],
  val avgMetrics: Array[Double]
) extends Model[StratifiedkfoldCrossValidatorModel]
{
  var _subModels: Option[Array[Array[Model[_]]]] = None

  def setSubModels(subModels: Option[Array[Array[Model[_]]]])
  : StratifiedkfoldCrossValidatorModel = {
    _subModels = subModels
    this
  }

  override def copy(extra: ParamMap): StratifiedkfoldCrossValidatorModel = {
    val copied = new StratifiedkfoldCrossValidatorModel(
      uid,
      bestModel.copy(extra).asInstanceOf[Model[_]],
      avgMetrics.clone()
    ).setSubModels(copySubModels(_subModels))
    copyValues(copied, extra).setParent(parent)
  }

  def copySubModels(subModels: Option[Array[Array[Model[_]]]])
  : Option[Array[Array[Model[_]]]] = {
    subModels.map(_.map(_.map(_.copy(ParamMap.empty).asInstanceOf[Model[_]])))
  }

  override def transform(dataset: Dataset[_]): DataFrame = {
    transformSchema(dataset.schema, logging = true)
    bestModel.transform(dataset)
  }

  override def transformSchema(schema: StructType): StructType = {
    bestModel.transformSchema(schema)
  }
}