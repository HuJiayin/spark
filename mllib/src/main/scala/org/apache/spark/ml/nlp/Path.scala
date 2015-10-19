package org.apache.spark.ml.nlp

private[ml] class Path extends Serializable{
  var rnode: Node = _
  var lnode: Node = _
  var cost: Double = _
  var fvector: Vector[Integer] = _
  var fIdx: Integer = _

  object Path{
    val path = new Path
    def getInstance: Path = {path}
  }
  def calExpectation(expected : Array[Double], Z: Double, size: Integer) = {
    var c: Double = math.exp(lnode.alpha + cost + rnode.beta - Z)
    while(fvector(fIdx)!= -1) {
      expected(fvector(0) + lnode.y*size + rnode.y) += c
      fIdx += 1
    }
  }

  def add(_lnode : Node, _rnode: Node) = {
    lnode = _lnode
    rnode = _rnode
    lnode.rpath :+ lnode
    rnode.lpath :+ rnode
  }

  def clear() = {
    cost = 0
  }

}
