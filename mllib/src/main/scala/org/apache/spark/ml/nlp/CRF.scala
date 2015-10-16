package org.apache.spark.ml.nlp

private[spark] class CRF {
  private val freq: Integer = 1
  private val maxiter: Integer = 100000
  private val cost: Double = 1.0
  private val eta: Double = 0.0001
  private var threadID: Integer = 1
  private var modelFile: String = ""
  private var threadNum: Integer = 0
  private var C: Integer = 0 //Convert
  private var thrinkingSize = 20
  private var threadPool:Option[Thread] = _


  def run(template: String, train: String)={
    //readParameters(template, train)
    learn(template, train)
  }

  def learn(template: String, train: String): Unit ={
    val tagger:Tagger = new Tagger()
    val featureIndex: FeatureIndex = new FeatureIndex()
    getThreadSize()
    tagger.open(featureIndex)
    tagger.read(train)
    tagger.shrink()
    featureIndex.shrink(freq)
    featureIndex.initAlpha(featureIndex.maxid)
    runCRF(tagger, featureIndex, featureIndex.alpha)


  }

  def runCRF(tagger: Tagger, featureIndex: FeatureIndex, alpha: Vector[Double]): Boolean = {
    true
  }

  def getThreadSize(): Integer = {
    threadNum = 1 //get cpu numbers
    threadNum
  }

  def onStart() {
    val thread = new Thread() {
      override def run() {
        var obj: Double = 0.0
        var err: Int = 0
        var zeroOne: Int = 0
        val expected: Vector[Double] = null
        val x: Vector[Tagger] = _
        val start_i: Int = 0
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
    threadPool = Some(thread)
    thread.start()
  }

  def readParameters(template: String, train: String): Unit ={

  }

}
