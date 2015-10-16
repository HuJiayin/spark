package org.apache.spark.ml.nlp

class FeatureCache {
  private var feature_freelist: Vector[Int]  = _
  def shrink(old2new: Map[Integer, Integer]): Unit = {

  }
  def add(f: Vector[Int]): Unit = {
    for(i<-0 until f.length - 1){
      feature_freelist :+ f(i)
    }
    feature_freelist :+ -1 //...until *f!=-1 end mark sentinel
  }
}
