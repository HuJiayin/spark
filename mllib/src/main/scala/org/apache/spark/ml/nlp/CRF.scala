package org.apache.spark.ml.nlp

import org.apache.spark.util.ThreadUtils

private[spark] class CRF {
  private val freq: Integer = 1
  private val maxiter: Integer = 100000
  private val cost: Double = 1.0
  private val eta: Double = 0.0001
  private val threadID: Integer = 1
  private val modelFile: String = ""
  private val threadNum: Integer = Runtime.getRuntime.availableProcessors()
  private var C: Integer = 0 //Convert
  private val thrinkingSize = 20
  private val threadPool = ThreadUtils.newDaemonFixedThreadPool(threadNum, "CRF")
  private var start_i: Int = 0


  def run(template: String, train: String)={
    //readParameters(template, train)
    learn(template, train)
  }

  def learn(template: String, train: String): Unit ={
    val tagger:Tagger = new Tagger()
    val featureIndex: FeatureIndex = new FeatureIndex()
    tagger.open(featureIndex)
    tagger.read(train)
    tagger.shrink()
    featureIndex.shrink(freq)
    featureIndex.initAlpha(featureIndex.maxid)
    runCRF(tagger, featureIndex, featureIndex.alpha)
  }

  def runCRF(tagger: Tagger, featureIndex: FeatureIndex, alpha: Vector[Double]): Unit = {
    (0 until threadNum) foreach { _ =>
      start_i += 1
      threadPool.execute(new CRFProcess)
    }
  }

  private class CRFProcess extends Runnable {
    override def run: Unit = {
      var obj: Double = 0.0
      var err: Int = 0
      var zeroOne: Int = 0
      val expected: Vector[Double] = null
      val x: Vector[Tagger] = _
      val size: Int = 0
      var idx: Int = 0
      while(idx>= start_i && idx < size){
        obj += x(idx).gradient(expected)
        err += x(idx).eval()
        if(err!=0){
          zeroOne += 1
        }
        idx = idx + threadNum
      }
    }
  }

  //def readParameters(template: String, train: String): Unit ={}

}
