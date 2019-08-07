package org.apache.spark

import org.apache.spark.ml.linalg.SparseVector
import org.apache.spark.ml.pspectrum.{PSpectrum, PSpectrumModel}
import org.apache.spark.sql.types.{DataTypes, StructField}
import org.apache.spark.sql.{SparkSession, types}
import org.junit.runner.RunWith
import org.scalatest.junit.JUnitRunner
import org.scalatest._

import scala.collection.mutable.ListBuffer

@RunWith(classOf[JUnitRunner])
class PSpectrumRelaxationSuite extends FunSuite with BeforeAndAfterAll with Matchers {

  private var spark: SparkSession = _

  override protected def beforeAll(): Unit = {
    super.beforeAll()
    spark = SparkSession.builder()
      .master("local[*]")
      .appName("PSpectrumRelaxationSuite")
      .getOrCreate()
  }

  override protected def afterAll(): Unit = {
    try {
      spark.sparkContext.stop()
    } finally {
      super.afterAll()
    }
  }

  test("Test P spectrum degree set"){

    val p:Int = 6
    val ps:PSpectrum = new PSpectrum()
    ps.setP(p)

    assert(p === ps.getP)

  }

  test("Test inputCol set"){

    val inputCol:String = "inputCol"
    val ps:PSpectrum = new PSpectrum()
    ps.setInputCol(inputCol)

    assert(inputCol === ps.getInputCol)

  }


  test("Test outputCol set"){

    val outputCol:String = "outputCol"
    val ps:PSpectrum = new PSpectrum()
    ps.setOutputCol(outputCol)

    assert(outputCol === ps.getOutputCol)
  }


  test("lol"){

    val training_data = spark.read
      .format("csv")
      .option("inferSchema", "true")
      .option("header", "true")
      .load("src/test/resources/data/test_pspectrum_training.csv")


    val test_data = spark.read
      .format("csv")
      .option("inferSchema", "true")
      .option("header", "true")
      .load("src/test/resources/data/test_pspectrum_test.csv")



    val schema = types.StructType(Array(
      StructField("val_0", DataTypes.DoubleType, nullable=false),
      StructField("val_1", DataTypes.DoubleType, nullable=false),
      StructField("val_2", DataTypes.DoubleType, nullable=false),
      StructField("val_3", DataTypes.DoubleType, nullable=false),
      StructField("val_4", DataTypes.DoubleType, nullable=false),
      StructField("val_5", DataTypes.DoubleType, nullable=false),
      StructField("val_6", DataTypes.DoubleType, nullable=false),
      StructField("val_7", DataTypes.DoubleType, nullable=false),
      StructField("val_8", DataTypes.DoubleType, nullable=false),
      StructField("val_9", DataTypes.DoubleType, nullable=false)
    ))

    val result_data = spark.read
      .format("csv")
      .schema(schema)
      .option("header", "true")
      .load("src/test/resources/data/test_pspectrum_result.csv")


    val ps:PSpectrum = new PSpectrum()

    ps.setInputCol("data").setOutputCol("data_transformed").setP(3)


    val psModel: PSpectrumModel = ps.fit(training_data)

    val result_data_generated = psModel.transform(test_data)

    var generatedContent:List[Array[Double]] = result_data_generated.select("data_transformed").collect().map(row => {
      row.getAs[SparseVector](0).toArray
    }).toList


    var trueGroundValues:List[Array[Double]] = result_data.rdd.collect().map(row => {

      var buffer = new ListBuffer[Double]()
      for (i <- row.schema.indices){
        buffer += row.getDouble(i)
      }

      buffer.result().toArray
    }).toList


    var assertionList = ListBuffer[Assertion]()
    for (j <- generatedContent.indices){
      assertionList += (generatedContent(j) should contain theSameElementsAs trueGroundValues(j))
    }

    assertionList.result().forall(_ == Succeeded)






  }

}
