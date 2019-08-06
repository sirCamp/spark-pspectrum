package org.apache.spark.ml.pspectrum


import org.apache.hadoop.fs.Path
import org.apache.spark.SparkException
import org.apache.spark.ml.attribute.NominalAttribute
import org.apache.spark.ml.linalg.{DenseVector, VectorUDT}
import org.apache.spark.ml.param.shared.{HasInputCol, HasOutputCol}
import org.apache.spark.ml.param.{IntParam, ParamMap, ParamValidators, Params}
import org.apache.spark.ml.util._
import org.apache.spark.ml.{Estimator, Model}
import org.apache.spark.rdd.RDD
import org.apache.spark.sql.catalyst.encoders.RowEncoder
import org.apache.spark.sql.functions.monotonicallyIncreasingId
import org.apache.spark.sql.types.{DataTypes, StringType, StructField, StructType}
import org.apache.spark.sql.{DataFrame, Dataset, Row, RowFactory}
import org.apache.spark.storage.StorageLevel
import org.apache.spark.util.VersionUtils.majorMinorVersion
import org.apache.spark.util.collection.CompactBuffer

import scala.collection.mutable

private[pspectrum] trait PSpectrumParams extends Params with HasInputCol with HasOutputCol {
  /**
   * The P spectrum degree
   * @group param
   */
  final val p: IntParam =
    new IntParam(this, "p", "The degree of the pspectrum" +
      "Must be > 0", ParamValidators.gt(0))

  /** @group getParam */
  def getP: Int = $(p)

    /** Validates and transforms the input schema. */
  protected def validateAndTransformSchema(schema: StructType): StructType = {

    require(!schema.fields.exists(field => field.name == outputCol.name),
      s"Output columns should not be duplicate.")

    val outputFields = Array($(inputCol)).zip(Array($(outputCol))).flatMap {
      case (inputColName, outputColName) =>
        schema.fieldNames.contains(inputColName) match {
          case true => Some(validateAndTransformField(schema, inputColName, outputColName))
          case _ => throw new SparkException(s"Input column $inputColName does not exist.")
        }
    }

    StructType(schema.fields ++ outputFields)
  }


  private def validateAndTransformField(
                                         schema: StructType,
                                         inputColName: String,
                                         outputColName: String): StructField = {
    val inputDataType = schema(inputColName).dataType
    require(inputDataType == StringType, s"The input column $inputColName must string type, but got $inputDataType.")

    require(schema.fields.forall(_.name != outputColName), s"Output column $outputColName already exists.")

    NominalAttribute.defaultAttr.withName($(outputCol)).toStructField()

  }

  setDefault(p -> 3)
}

private[pspectrum] object PSpectrum {


  def getSpectrumEmbedding(data: RDD[Row], p: Int): RDD[mutable.HashMap[Long, mutable.HashMap[String, Long]]] = data.map((row: Row) => {

    val document = row.getString(1)
    val id = row.getLong(0)
    val end = document.length - p
    val start = 0
    val step = 1
    val dictionary = new mutable.HashMap[String, Long]

    for(i <- start to end){
      val key = document.substring(i, i + p)
      dictionary.put(key, 1L + dictionary.getOrElse(key, 0L))
    }

    val res = new mutable.HashMap[Long,mutable.HashMap[String, Long]]
    res.put(id, dictionary)
    res

  })


  def computeKernelFromEmbedding(X: RDD[mutable.HashMap[Long,mutable.HashMap[String, Long]]], T: Option[RDD[mutable.HashMap[Long,mutable.HashMap[String, Long]]]]): RDD[Row] = {

    val cov = T.getOrElse(default = X)

    /**
     * row => Tuple2[Tuple2[mutable.HashMap[String,Long], Long] , mutable.HashMap[String,Long]
     **/
    X.zipWithIndex()
      .cartesian(cov)
      .groupBy(row => {
        row._1._2
      })
      .sortByKey()
      .map(row  => {
        //rowContent => Tuple2[Long, Iterable[Tuple2[Tuple2[mutable.HashMap[String, Long], Long], mutable.HashMap[String, Long]]]]
        var kernelRow = mutable.ListBuffer[Double]()
        row._2.asInstanceOf[CompactBuffer[  Tuple2[Tuple2[mutable.HashMap[Long,mutable.HashMap[String, Long]], Long],mutable.HashMap[ Long,mutable.HashMap[String, Long]]]]].foreach(
          rowContent => {

            val word1 = rowContent._1.asInstanceOf[(mutable.HashMap[Long,mutable.HashMap[String, Long]], Long)]._1.asInstanceOf[mutable.HashMap[Long,mutable.HashMap[String, Long]]]
            val word2 = rowContent._2.asInstanceOf[mutable.HashMap[Long,mutable.HashMap[String, Long]]]//[Long, mutable.HashMap[String, Long]]]

            var iThValues = 0L//new ListBuffer[Long]()
            word1.values.last.keySet.foreach( key => {

              iThValues += (word1.values.last.getOrElse(key, default = 0L)*word2.last._2.getOrElse(key, 0L))

            })

            kernelRow += iThValues.doubleValue()

          })

        RowFactory.create(row._1.asInstanceOf[Object], new DenseVector(kernelRow.toArray).toSparse)

      })
  }

}

class PSpectrum private[pspectrum](override val uid: String) extends Estimator[PSpectrumModel]
  with PSpectrumParams with DefaultParamsWritable {

  def this() = this(Identifiable.randomUID("pSpectrum"))

  /** @group setParam */
  def setInputCol(value: String): this.type = set(inputCol, value)

  /** @group setParam */
  def setOutputCol(value: String): this.type = set(outputCol, value)

  /** @group setParam */
  def setP(value: Int): this.type = set(p, value)

  override def transformSchema(schema: StructType): StructType = {
    val inputType = schema($(inputCol)).dataType
    require(inputType.isInstanceOf[StringType],
      s"The input column must be ${StringType}, but got ${schema($(inputCol)).dataType}.")

    SchemaUtils.appendColumn(schema, $(outputCol), new VectorUDT)
  }

  override def fit(dataset: Dataset[_]): PSpectrumModel = {


    // If input dataset is not originally cached, we need to unpersist it
    // once we persist it later.
    val needUnpersist = dataset.storageLevel == StorageLevel.NONE

    var uniqueId = Identifiable.randomUID("id")
    val indexedDataset = dataset.withColumn(uniqueId, monotonicallyIncreasingId)

    val rddSpectrumEmbedding = PSpectrum.getSpectrumEmbedding(indexedDataset.select(uniqueId, $(inputCol)).rdd, getP)


    if (needUnpersist) {
      dataset.unpersist()
    }

    copyValues(new PSpectrumModel(uid, rddSpectrumEmbedding).setParent(this))

  }


  override def copy(extra: ParamMap): PSpectrum = defaultCopy(extra)
}


class PSpectrumModel private[pspectrum] (
                                        override val uid: String,
                                        val trainRddSpectrumEmbedding: RDD[mutable.HashMap[Long,mutable.HashMap[String, Long]]]
                                      ) extends Model[PSpectrumModel] with PSpectrumParams with MLWritable {


  import PSpectrumModel._

  def setInputCol(value: String): this.type = set(inputCol, value)


  def setOutputCol(value: String): this.type = set(outputCol, value)


  override def transform(dataset: Dataset[_]): DataFrame = {
    transformSchema(dataset.schema, logging = true)

    val uniqueId = Identifiable.randomUID("id")
    val indexedDataset = dataset.withColumn(uniqueId, monotonicallyIncreasingId)
    val newSpectrumEmbedding = PSpectrum.getSpectrumEmbedding(indexedDataset.select(uniqueId,$(inputCol)).rdd, getP)

    val transformedKernelRDD = PSpectrum.computeKernelFromEmbedding(newSpectrumEmbedding, Some(trainRddSpectrumEmbedding))


    val schema = StructType(
      StructField(uniqueId, DataTypes.LongType, nullable = false, org.apache.spark.sql.types.Metadata.empty)::
        StructField($(outputCol), new VectorUDT(), nullable = false, org.apache.spark.sql.types.Metadata.empty)::Nil

    )
    val transformedSpectrumDT = dataset.sparkSession.createDataset(transformedKernelRDD)(RowEncoder(schema))

    val df = indexedDataset.join(transformedSpectrumDT, usingColumn = uniqueId).drop(uniqueId)

    df
  }



  override def transformSchema(schema: StructType): StructType = {
    validateAndTransformSchema(schema)
  }


  override def copy(extra: ParamMap): PSpectrumModel = {
    val copied = new PSpectrumModel(uid, trainRddSpectrumEmbedding)
    copyValues(copied, extra).setParent(parent)
  }

  override def write: PSpectrumModelWriter = new PSpectrumModelWriter(this)

}


private[pspectrum] object PSpectrumModel extends MLReadable[PSpectrumModel] {

  override def read: MLReader[PSpectrumModel] = new PSpectrumModelReader

  override def load(path: String): PSpectrumModel = super.load(path)

  private[PSpectrumModel]
  class PSpectrumModelWriter(instance: PSpectrumModel) extends MLWriter {

    override protected def saveImpl(path: String): Unit = {
      DefaultParamsWriter.saveMetadata(instance, path, sc)
      val data = Data(instance.trainRddSpectrumEmbedding)
      val dataPath = new Path(path, "data").toString
      sparkSession.createDataFrame(Seq(data)).repartition(1).write.parquet(dataPath)
    }

    private case class Data(trainRddSpectrumEmbedding: RDD[mutable.HashMap[Long,mutable.HashMap[String, Long]]])
  }

  private class PSpectrumModelReader extends MLReader[PSpectrumModel] {

    private val className = classOf[PSpectrumModel].getName

    override def load(path: String): PSpectrumModel = {
      val metadata = DefaultParamsReader.loadMetadata(path, sc, className)
      val dataPath = new Path(path, "data").toString

      // We support loading old `PSpectrumModel` saved by previous Spark versions.
      // Previous model has `labels`, but new model has `labelsArray`.
      val (majorVersion, minorVersion) = majorMinorVersion(metadata.sparkVersion)
      val trainRddSpectrumEmbedding = if (majorVersion < 3) {
        // Spark 2.4 and before.
        val data = sparkSession.read.parquet(dataPath)
          .select("train")
        //.head()
        data.rdd.map(r => r.getAs[mutable.HashMap[Long,mutable.HashMap[String, Long]]](0))
      } else {
        // After Spark 3.0.
        val data = sparkSession.read.parquet(dataPath)
          .select("trainRddSpectrumEmbedding")
        data.rdd.map(r => r.getAs[mutable.HashMap[Long,mutable.HashMap[String, Long]]](0))
      }
      val model = new PSpectrumModel(metadata.uid, trainRddSpectrumEmbedding)
      metadata.getAndSetParams(model)
      model
    }
  }
}