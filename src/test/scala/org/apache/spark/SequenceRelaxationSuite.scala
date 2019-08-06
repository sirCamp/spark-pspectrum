package org.apache.spark

import org.apache.spark.ml.relaxation.SequenceRelaxation
import org.apache.spark.sql.SparkSession
import org.junit.runner.RunWith
import org.scalatest.junit.JUnitRunner
import org.scalatest.{BeforeAndAfterAll, FunSuite}

import scala.collection.mutable

@RunWith(classOf[JUnitRunner])
class SequenceRelaxationSuite extends FunSuite with BeforeAndAfterAll{


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
    val upper_chars = Array("A", "B", "C", "D", "E", "F", "G", "H", "I", "J", "K", "L", "M", "N", "O", "P", "Q", "R", "S", "T", "U", "V", "W", "X", "YZ")
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

}