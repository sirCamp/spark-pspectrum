package org.apache.spark.ml.relaxation

import org.apache.spark.ml.Transformer
import org.apache.spark.ml.param.shared.{HasHandleInvalid, HasInputCol, HasOutputCol}
import org.apache.spark.ml.param.{HashMapParam, ParamMap, Params}
import org.apache.spark.ml.util.{DefaultParamsReadable, DefaultParamsWritable, Identifiable}
import org.apache.spark.sql.functions.udf
import org.apache.spark.sql.types._
import org.apache.spark.sql.{DataFrame, Dataset}

import scala.collection.mutable


private[relaxation] trait SequenceRelaxationParams extends Params with HasInputCol with HasOutputCol with HasHandleInvalid{
  /**
   * The P spectrum degree
   * @group param
   */
  final val characterEncoding: HashMapParam =
    new HashMapParam(this, "characterEncoding", "Map that contains the mapping for the string sequence relaxation")

  /** @group getParam */
  def getCharacterEncoding: mutable.HashMap[String,String] = $(characterEncoding)


  val defaultCharacterEncoding = new mutable.HashMap[String, String]()
  val upper_chars = Array("A", "B", "C", "D", "E", "F", "G", "H", "I", "J", "K", "L", "M", "N", "O", "P", "Q", "R", "S", "T", "U", "V", "W", "X", "Y", "Z")
  val lower_chars = Array("a", "b", "c", "d", "e", "f", "g", "h", "i", "j", "k", "l", "m", "n", "o", "p", "q", "r", "s", "t", "u", "v", "w", "x", "y", "z")
  val number_chars = Array("1", "2", "3", "4", "5", "6", "7", "8", "9", "0")
  val symbol_chars = Array("!", "\"", "£", "$", "%", "&", "/", "(", ")", "=", "?", "^", "§", "<", ">", ".", ":", ",", ";", "-", "_", "+", "*", "[", "]", "#")

  upper_chars.foreach(char => {
    defaultCharacterEncoding.put(char, "A")
  })
  lower_chars.foreach(char => {
    defaultCharacterEncoding.put(char, "a")
  })
  number_chars.foreach(char => {
    defaultCharacterEncoding.put(char, "0")
  })
  symbol_chars.foreach(char => {
    defaultCharacterEncoding.put(char, "+")
  })

  setDefault(characterEncoding, defaultCharacterEncoding)

  setDefault(handleInvalid, SequenceRelaxation.ERROR_INVALID)

}


class SequenceRelaxation (override val uid: String)
  extends Transformer with SequenceRelaxationParams with DefaultParamsWritable {


  def this() = this(Identifiable.randomUID("sequenceRelaxation"))

  /** @group setParam */
  def setInputCol(value: String): this.type = set(inputCol, value)

  /** @group setParam */
  def setOutputCol(value: String): this.type = set(outputCol, value)

  /** @group setParam */
  def setCharacterEncoding(value: mutable.HashMap[String,String]): this.type = set(characterEncoding, value)

  /** @group setParam */
  def setHandleInvalid(value: String): this.type = set(handleInvalid, value)


  override def transform(dataset: Dataset[_]): DataFrame = {

    val outputSchema = transformSchema(dataset.schema)

    val relaxSequence = udf { document: String =>
      val characterEncoding = getCharacterEncoding

      var newDocument = mutable.ListBuffer[String]()
      document.foreach { character =>

        getHandleInvalid match {
          case SequenceRelaxation.ERROR_INVALID => {
            if(characterEncoding.contains(character.toString)) {
              newDocument += characterEncoding.get(character.toString).get.toString
            }
            else {
              throw new IllegalArgumentException(s"An unmapped ${character} char has been faund!")
            }
          }
          case SequenceRelaxation.KEEP_INVALID =>  newDocument += characterEncoding.getOrElse(character.toString, character.toString)
          case SequenceRelaxation.SKIP_INVALID =>  newDocument += characterEncoding.getOrElse(character.toString, "_")
        }


      }
      newDocument.toList.mkString("")
    }

    val temporaryDataset = dataset.withColumn($(outputCol), relaxSequence(dataset.col($(inputCol))).as($(outputCol), Metadata.empty))


    temporaryDataset.toDF
  }

  override def transformSchema(schema: StructType): StructType = {
    val inputColName = $(inputCol)
    val outputColName = $(outputCol)
    val incorrectColumns =   schema(inputColName).dataType match {
        case _: StringType => None
        case other => Some(s"Data type ${other.catalogString} of column $inputColName is not supported.")
      }

    if (incorrectColumns.nonEmpty) {
      throw new IllegalArgumentException(incorrectColumns.mkString("\n"))
    }
    if (schema.fieldNames.contains(outputColName)) {
      throw new IllegalArgumentException(s"Output column $outputColName already exists.")
    }
    StructType(schema.fields :+ new StructField(outputColName, DataTypes.StringType, nullable = false))
  }

  override def copy(extra: ParamMap): SequenceRelaxation = defaultCopy(extra)
}


object SequenceRelaxation extends DefaultParamsReadable[SequenceRelaxation] {

  private[relaxation] val SKIP_INVALID: String = "skip"
  private[relaxation] val ERROR_INVALID: String = "error"
  private[relaxation] val KEEP_INVALID: String = "keep"
  private[relaxation] val supportedHandleInvalids: Array[String] =
    Array(SKIP_INVALID, ERROR_INVALID, KEEP_INVALID)


  override def load(path: String): SequenceRelaxation = super.load(path)


}