/* Copyright 2019 Devang Mistry. All rights reserved.
 * You may not remove this notice, but you may add
 * your name if you've made software modifications.
 * Software provided as-is. Available under the
 * terms of the AGPL v3 license available here:
 * https://www.gnu.org/licenses/agpl-3.0.en.html
 */

package jigsaw

import org.apache.spark.ml.Transformer
import org.apache.spark.ml.linalg.SQLDataTypes.VectorType
import org.apache.spark.ml.param.{Param, ParamMap}
import org.apache.spark.ml.util.{DefaultParamsWritable, Identifiable}
import org.apache.spark.sql.expressions.UserDefinedFunction
import org.apache.spark.sql.functions.{col, udf}
import org.apache.spark.sql.types.{IntegerType, StructField, StructType}
import org.apache.spark.sql.{DataFrame, Dataset}

//transforms given binary labels into a single categorical label

class LabelPowerset(override val uid: String) extends Transformer with DefaultParamsWritable {
  final val labelInputCols= new Param[String](this, "labelInputCols", "The label input column")
  final val labelOutputCol = new Param[String](this, "labelOutputCol", "The label output column")
  final val inputCol= new Param[String](this, "inputCol", "The input column")
  final val outputCol = new Param[String](this, "outputCol", "The output column")

  def setLabelInputCols(value: String): this.type = set(labelInputCols, value)
  def setInputCol(value: String): this.type = set(inputCol, value)

  def setLabelOutputCol(value: String): this.type = set(labelOutputCol, value)
  def setOutputCol(value: String): this.type = set(outputCol, value)

  def this() = this(Identifiable.randomUID("labeledpowerset"))

  def copy(extra: ParamMap): LabelPowerset = {
    defaultCopy(extra)
  }

  override def transformSchema(schema: StructType): StructType = {
    // Check that the input type is a string
    val field = schema.fields(schema.fieldIndex($(inputCol)))
    if (field.dataType != VectorType) {
      throw new Exception(s"Input type ${field.dataType} did not match input type VectorType")
    }
    val fields = $(labelInputCols).split(",").map(x => schema.fields(schema.fieldIndex(x)))
    fields.foreach ( f => {
      if (f.dataType != IntegerType)
        throw new Exception(s"Input type ${field.dataType} did not match input type IntegerType")
    })
    // Add the return output label field
    schema.add(StructField($(labelOutputCol), IntegerType, nullable = false))
  }

  def transform(df: Dataset[_]): DataFrame = {
    df.select(col("features"))
    df.withColumn($(labelOutputCol), Udf(df("toxic"), df("severe_toxic"), df("obscene"), df("threat"), df("insult"), df("identity_hate")))
  }


  def Udf: UserDefinedFunction = udf((val1: Int, val2: Int, val3: Int, val4: Int, val5: Int, val6: Int) =>
    Utils.getCat(val1, val2, val3, val4, val5, val6)
  )



}

