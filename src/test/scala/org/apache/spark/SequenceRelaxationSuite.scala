package org.apache.spark

import org.apache.spark.ml.relaxation.SequenceRelaxation
import org.apache.spark.sql.catalyst.encoders.RowEncoder
import org.apache.spark.sql.types.{DataTypes, StructField, StructType}
import org.apache.spark.sql.{Row, RowFactory, SparkSession}
import org.junit.runner.RunWith
import org.scalatest.junit.JUnitRunner
import org.scalatest.{BeforeAndAfterAll, FunSuite, Matchers}

import scala.collection.mutable
import scala.collection.mutable.ListBuffer

@RunWith(classOf[JUnitRunner])
class SequenceRelaxationSuite extends FunSuite with BeforeAndAfterAll with Matchers {


  private var spark: SparkSession = _

  override protected def beforeAll(): Unit = {
    super.beforeAll()
    spark = SparkSession.builder()
      .master("local[*]")
      .appName("SequenceRelaxationSuite")
      .getOrCreate()
  }

  override protected def afterAll(): Unit = {
    try {
      spark.sparkContext.stop()
    } finally {
      super.afterAll()
    }
  }


  test("Test character Embedding map") {
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

    var sr:SequenceRelaxation = new SequenceRelaxation()

    assert(sr.getCharacterEncoding === defaultCharacterEncoding)


  }

  test("Test set inputColumn") {

    var inputCol: String = "test"
    var sr: SequenceRelaxation = new SequenceRelaxation().setInputCol(inputCol)


    assert(inputCol === sr.getInputCol)

  }


  test("Test set outputColumn") {

    var outCol: String = "test"
    var sr: SequenceRelaxation = new SequenceRelaxation().setOutputCol(outCol)


    assert(outCol === sr.getOutputCol)

  }


  test("Test default handleInvalid") {

    var handleInvalid: String = "error"
    var sr: SequenceRelaxation = new SequenceRelaxation()


    assert(handleInvalid === sr.getHandleInvalid)
  }


  test("Test set character encoder Map") {

    val characterEncoding = new mutable.HashMap[String, String]()
    val upper_consonant_chars = Array("B", "C", "D", "F", "G", "H", "J", "K", "L", "M", "N", "P", "Q", "R", "S", "T", "V", "W", "X", "Y", "Z")
    val lower_consonant_chars = Array("b", "c", "d", "f", "g", "h", "j", "k", "l", "m", "n", "p", "q", "r", "s", "t", "v", "w", "x", "y", "z")
    val upper_vowel_chars = Array("A", "E", "I", "O", "U")
    val lower_vowel_chars = Array("a", "e", "i", "o", "u")
    val number_chars = Array("1", "2", "3", "4", "5", "6", "7", "8", "9", "0")
    val symbol_chars = Array("!", "\"", "£", "$", "%", "&", "/", "(", ")", "=", "?", "^", "§", "<", ">", ".", ":", ",", ";", "-", "_", "+", "*", "[", "]", "#")

    upper_consonant_chars.foreach(char => {
      characterEncoding.put(char, "C")
    })
    lower_consonant_chars.foreach(char => {
      characterEncoding.put(char, "c")
    })
    upper_vowel_chars.foreach(char => {
      characterEncoding.put(char, "V")
    })
    lower_vowel_chars.foreach(char => {
      characterEncoding.put(char, "v")
    })
    number_chars.foreach(char => {
      characterEncoding.put(char, "0")
    })
    symbol_chars.foreach(char => {
      characterEncoding.put(char, "+")
    })

    var sr: SequenceRelaxation = new SequenceRelaxation().setCharacterEncoding(characterEncoding)

    assert(characterEncoding === sr.getCharacterEncoding)
  }


  test("Test data transformation") {


    val characterEncoding = new mutable.HashMap[String, String]()
    val upper_consonant_chars = Array("B", "C", "D", "F", "G", "H", "J", "K", "L", "M", "N", "P", "Q", "R", "S", "T", "V", "W", "X", "Y", "Z")
    val lower_consonant_chars = Array("b", "c", "d", "f", "g", "h", "j", "k", "l", "m", "n", "p", "q", "r", "s", "t", "v", "w", "x", "y", "z")
    val upper_vowel_chars = Array("A", "E", "I", "O", "U")
    val lower_vowel_chars = Array("a", "e", "i", "o", "u")
    val number_chars = Array("1", "2", "3", "4", "5", "6", "7", "8", "9", "0")
    val symbol_chars = Array("!", "\"", "£", "$", "%", "&", "/", "(", ")", "=", "?", "^", "§", "<", ">", ".", ":", ",", ";", "-", "_", "+", "*", "[", "]", "#")

    upper_consonant_chars.foreach(char => {
      characterEncoding.put(char, "C")
    })
    lower_consonant_chars.foreach(char => {
      characterEncoding.put(char, "c")
    })
    upper_vowel_chars.foreach(char => {
      characterEncoding.put(char, "V")
    })
    lower_vowel_chars.foreach(char => {
      characterEncoding.put(char, "v")
    })
    number_chars.foreach(char => {
      characterEncoding.put(char, "0")
    })
    symbol_chars.foreach(char => {
      characterEncoding.put(char, "+")
    })

    var sr: SequenceRelaxation = new SequenceRelaxation()

    val rowList = new ListBuffer[Row]()
    rowList += RowFactory.create("ABCDEFGHIJKLMNOPQRSTUVWXYZ")
    rowList += RowFactory.create("abcdefghijklmnopqrstuvwxyz")
    rowList += RowFactory.create("1234567890")
    rowList += RowFactory.create("!\"£$%&/()=?^§<>.:,;-_+*[]#")
    rowList += RowFactory.create("AbaB1%")

    val schema = StructType(
      StructField("test_data", DataTypes.StringType, nullable = false, org.apache.spark.sql.types.Metadata.empty) :: Nil

    )

    val testDataset = spark.createDataset(rowList.toList)(RowEncoder.apply(schema))

    val relaxation = new SequenceRelaxation().setCharacterEncoding(characterEncoding).setInputCol("test_data").setOutputCol("test_data_output")

    val transformedTestDataset = relaxation.transform(testDataset)


    List("VCCCVCCCVCCCCCVCCCCCVCCCCC", "vcccvcccvcccccvcccccvccccc", "0000000000", "++++++++++++++++++++++++++", "VcvC0+") should contain theSameElementsAs transformedTestDataset.select("test_data_output").rdd.collect().map(row => {
      row.getString(0)
    }).toList
  }

}