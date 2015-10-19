package org.apache.spark.ml.nlp

import org.apache.spark.SparkFunSuite
import org.apache.spark.mllib.util.MLlibTestSparkContext
import org.scalatest.Matchers

class CRFTests extends SparkFunSuite with MLlibTestSparkContext with Matchers {
  private val CRFobj: CRF = null
  test("test CRF"){
    CRFobj.run("/home/hujiayin/Downloads/template_file", "/home/hujiayin/Downloads/train_file")
  }
}
