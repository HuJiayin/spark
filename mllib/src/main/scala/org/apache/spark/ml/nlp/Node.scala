package org.apache.spark.ml.nlp

private[ml] class Node extends Serializable {
  var x: Integer = _
  var y: Integer= _
  var alpha: Double= _
  var beta: Double= _
  var cost: Double= _
  var bestCost: Double= _
  var prev: Node= _
  var fvector: Vector[Integer]= _
  var fIdx: Integer= _
  var lpath: Vector[Path]= _
  var rpath: Vector[Path]= _
  val MINUS_LOG_EPSILON = 50

/*  object Node{
    val node = new Node
    def getInstance: Node = {node}
  }
*/

  def logsumexp(x:Double,y:Double, flg:Boolean): Double={
    if(flg) return y
    val vMin: Double = math.min(x,y)
    val vMax: Double = math.max(x,y)
    if(vMax > vMin + MINUS_LOG_EPSILON)
    {
      vMax
    } else {
      vMax + math.log(math.exp(vMin-vMax) + 1.0)
    }
  }

  def calcAlpha()={
    alpha = 0
    for (i <- 0 until lpath.length - 1)
    {
      alpha = logsumexp(alpha, lpath(i).cost+lpath(i).lnode.cost,i==0)
    }
    alpha+=cost

  }
  def calcBeta()={
    beta = 0
    for (i <- 0 until rpath.length - 1)
    {
      beta = logsumexp(beta, rpath(i).cost+rpath(i).rnode.cost, i==0)
    }
  }

  def calExpectation(expected : Vector[Double], Z: Double, size: Integer) = {
    var c: Double = math.exp(alpha + cost + beta - Z)
    val pathObj: Path = new Path()
    while(fvector(fIdx)!= -1) {
      expected(fvector(0) + y) += c
      fIdx += 1
    }
    for(i<-0 until lpath.length - 1)
    {
      pathObj.calExpectation(expected, Z, size)
    }
  }

  def clear() = {
    x = 0
    y = 0
    alpha = 0
    beta = 0
    cost = 0
  }
}