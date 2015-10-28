/*
 * Licensed to the Apache Software Foundation (ASF) under one or more
 * contributor license agreements.  See the NOTICE file distributed with
 * this work for additional information regarding copyright ownership.
 * The ASF licenses this file to You under the Apache License, Version 2.0
 * (the "License"); you may not use this file except in compliance with
 * the License.  You may obtain a copy of the License at
 *
 *    http://www.apache.org/licenses/LICENSE-2.0
 *
 * Unless required by applicable law or agreed to in writing, software
 * distributed under the License is distributed on an "AS IS" BASIS,
 * WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
 * See the License for the specific language governing permissions and
 * limitations under the License.
 */
package org.apache.spark.ml.nlp

import scala.collection.mutable.ArrayBuffer
import scala.io.Source._

private[ml] class Tagger extends Serializable {
  var mode: Integer = 2
  // LEARN
  var vlevel: Integer = 0
  var nbest: Integer = 0
  var ysize: Integer = 0
  var cost: Double = 0.0
  var Z: Double = 0.0
  var feature_id: Integer = 0
  var thread_id: Integer = 0
  var feature_idx: FeatureIndex = new FeatureIndex
  var x: ArrayBuffer[Array[String]] = new ArrayBuffer[Array[String]]()
  var node: ArrayBuffer[ArrayBuffer[Node]] = new ArrayBuffer[ArrayBuffer[Node]]()
  var penalty: ArrayBuffer[ArrayBuffer[Double]] = new ArrayBuffer[ArrayBuffer[Double]]()
  var answer: ArrayBuffer[Integer] = new ArrayBuffer[Integer]()
  var result: ArrayBuffer[Integer] = new ArrayBuffer[Integer]()
  val MINUS_LOG_EPSILON = 50

  def open(featureIndex: FeatureIndex): Unit = {
    feature_idx = featureIndex
    ysize = feature_idx.y.size
  }

  def read(filename: String): Tagger = {
    val lineIter: Iterator[String] = fromFile(filename).getLines()
    val line: Array[String] = lineIter.toArray
    var i: Int = 0
    var columns: Array[String] = null
    val s: Integer = x.size
    var r: Integer = ysize
    while (i < line.length) {
      if (line(i).charAt(0) != '\0'
        && line(i).charAt(0) != ' '
        && line(i).charAt(0) != '\t') {
        columns = line(i).split('|')
        x.append(columns)
        if (mode == 2) {
          // LEARN
          for (i <- 0 until ysize) {
            if (feature_idx.y(columns(feature_idx.xsize))) {
              r = i
            }
          }
          answer.insert(s, r)
        }
      }
      i += 1
    }
    this
  }

  def setFeatureId(id: Integer): Unit = {
    feature_id = id
  }

  def shrink(): Unit = {
    feature_idx.buildFeatures(this)
  }

  def buildLattice(): Unit = {
    if (x.isEmpty) {
      // rebuildFeatures
      for (i <- 0 until x.length - 1) {
        for (j <- 0 until ysize) {
          feature_idx.calcCost(node(i)(j))
          for (k <- 0 until node(i)(j).lpath.length - 1) {
            feature_idx.calcCost(node(i)(j).lpath(k))
          }
        }
      }
    }
    if (penalty.nonEmpty) {
      for (i <- 0 until x.length - 1) {
        for (j <- 0 until ysize) {
          node(i)(j).cost += penalty(i)(j)
        }
      }
    }
  }

  def forwardBackward(): Unit = {
    var idx: Int = x.length - 1
    if (x.nonEmpty) {
      for (i <- 0 until x.length - 1) {
        for (j <- 0 until ysize) {
          node(i)(j).calcAlpha()
        }
      }
      while (idx >= 0) {
        for (j <- 0 until ysize) {
          node(idx)(j).calcBeta()
          idx -= 1
        }
      }
      Z = 0.0
      for (i <- 0 until ysize) {
        Z = logsumexp(Z, node(0)(i).beta, i == 0)
      }
    }
  }

  def viterbi(): Unit = {
    var bestc: Double = -1e37
    var best: Node = null
    var cost: Double = 0.0
    var nd: Node = null
    for (i <- 0 until x.length - 1) {
      for (j <- 0 until ysize) {
        for (k <- 0 until node(i)(j).lpath.length - 1) {
          cost = node(i)(j).lpath(k).lnode.bestCost
          +node(i)(j).lpath(k).lnode.cost + node(i)(j).cost
          if (cost > bestc) {
            bestc = cost
            best = node(i)(j).lpath(k).lnode
          }
          node(i)(j).prev = best
          if (best != null) {
            node(i)(j).cost = bestc
          } else {
            node(i)(j).cost = node(i)(j).cost
          }
        }
      }
    }
    bestc = -1e37
    var j:Int = 0
    while (j < ysize) {
      if (node(x.length - 1)(j).bestCost > bestc) {
        best = node(x.length - 1)(j)
        bestc = node(x.length - 1)(j).bestCost
      }
      j += 1
    }
    nd = best
    while (nd != null) {
      result(nd.x) = nd.y
      nd = nd.prev
    }
    cost = -node(x.length - 1)(result(x.length - 1)).bestCost
  }

  def gradient(expected: Array[Double]): Double = {
    var s: Double = 0.0
    var lNode: Node = null
    var rNode: Node = null
    var lPath: Path = null
    var idx: Int = 0

    if (x.isEmpty) {
      0.0
    }
    buildLattice()
    forwardBackward()

    for (i <- 0 until x.length - 1) {
      for (j <- 0 until ysize) {
        node(i)(j).calExpectation(expected, Z, ysize)
      }
    }
    for (row <- 0 until x.length - 1) {
      while (node(row)(answer(row)).fvector(idx) != -1) {
        expected(node(row)(answer(row)).fvector(0) + answer(row)) -= 1
        idx += 1
      }
      s += node(row)(answer(row)).cost
      for (i <- 0 until node(row)(answer(row)).lpath.length - 1) {
        lNode = node(row)(answer(row)).lpath(i).lnode
        rNode = node(row)(answer(row)).lpath(i).rnode
        lPath = node(row)(answer(row)).lpath(i)
        if (lNode.y == answer(lNode.x)) {
          while (lPath.fvector(idx) != -1) {
            expected(lPath.fvector(0) + lNode.y * ysize + rNode.y) -= 1
            idx += 1
          }
          s += lPath.cost
        }
      }
    }
    viterbi()
    Z - s
  }

  def eval(): Int = {
    var err: Int = 0
    for (i <- 0 until x.length - 1) {
      if (answer(i) != result(i)) {
        err += 1
      }
    }
    err
  }

  def logsumexp(x: Double, y: Double, flg: Boolean): Double = {
    if (flg) return y
    val vMin: Double = math.min(x, y)
    val vMax: Double = math.max(x, y)
    if (vMax > vMin + MINUS_LOG_EPSILON) {
      vMax
    } else {
      vMax + math.log(math.exp(vMin - vMax) + 1.0)
    }
  }

  def getFeatureIdx(): FeatureIndex = {
    if(feature_idx!=null){
      return feature_idx
    }
    null
  }
}
