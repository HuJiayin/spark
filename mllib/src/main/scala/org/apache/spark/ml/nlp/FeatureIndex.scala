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

import scala.StringBuilder
import scala.collection.mutable.Map
import scala.io.Source
import scala.io.Source._
import scala.collection.mutable
import scala.collection.mutable.ArrayBuffer

private[ml] class FeatureIndex extends Serializable {
  var maxid: Integer = 0
  var alpha: Array[Double] = Array[Double]()
  var alpha_float: Array[Float] = Array[Float]()
  var cost_factor: Double = 0.0
  var xsize: Integer = 0
  var check_max_xsize: Boolean = false
  var max_xsize: Integer = _
  var unigram_templs: ArrayBuffer[String] = new ArrayBuffer[String]()
  var bigram_templs: ArrayBuffer[String] = new ArrayBuffer[String]()
  var y: Set[String] = Set[String]()
  var templs: String = new String
  var dic: Map[String, Map[Integer, Integer]] = Map[String, Map[Integer, Integer]]()
  val kMaxContextSize: Integer = 8
  val BOS = Vector[String]("B-1", "_B-2", "_B-3", "_B-4",
    "_B-5", "_B-6", "_B-7", "_B-8")
  val EOS = Vector[String]("_B+1", "_B+2", "_B+3", "_B+4",
    "_B+5", "_B+6", "_B+7", "_B+8")

  def openTemplate(filename: String): Unit = {
    val lineIter: Iterator[String] = fromFile(filename).getLines()
    val line: Array[String] = lineIter.toArray
    var i: Int = 0
    while (i < line.length) {
      if (line(i).charAt(0) == 'U') {
        unigram_templs += line(i)
      } else if (line(i).charAt(0) == 'B') {
        bigram_templs += line(i)
      }
      i += 1
    }
    make_templs()
  }

  def openTagSet(filename: String): FeatureIndex = {
    val lineIter: Iterator[String] = fromFile(filename).getLines()
    val line: Array[String] = lineIter.toArray
    var lineHead = line(0).charAt(0)
    var tag: Array[String] = null
    var i: Int = 0
    var max: Int = 0
    while ( i < line.length) {
      lineHead = line(i).charAt(0)
      if (lineHead != '\0' && lineHead != ' ' && lineHead != '\t') {
        tag = line(i).split('\t')
        if(tag.size > max){
          max = tag.size
        }
        y += tag(tag.length - 1)
      }
      i += 1
    }
    xsize = max - 1
    this
  }

  def make_templs(): Unit = {
    var i: Int = 0
    while (i < unigram_templs.length) {
      templs += unigram_templs(i)
      i += 1
    }
    while (i < bigram_templs.length) {
      templs += bigram_templs(i)
      i += 1
    }
  }

  def shrink(freq: Integer): Unit = {
    var newMaxId: Integer = 0
    val key: String = null

    if (freq > 1) {
      while (dic.iterator.next() != null) {
        dic.foreach { case (_, con) =>
          con.foreach(pair => if (pair._2 > freq) {
            con.update(newMaxId, pair._1)
          } else {
            dic - key
          })
        }
        if (key.toString.charAt(0) == 'U') {
          newMaxId += y.size
        } else {
          newMaxId += y.size * y.size
        }
      }
      maxid = newMaxId
    }
  }

  def buildFeatures(tagger: Tagger): Unit = {
    var os: String = null
    val feature: ArrayBuffer[Integer] = null
    var id: Integer = 0
    for (cur <- 0 until tagger.x.size - 1) {
      for (it <- 0 until unigram_templs.length - 1) {
        os = applyRule(unigram_templs(it), cur, tagger)
        id = getId(os)
        if (id != -1) {
          feature :+ id
        }
      }

      for (it <- 0 until bigram_templs.length - 1) {
        os = applyRule(bigram_templs(it), cur, tagger)
        id = getId(os)
        if (id != -1) {
          feature :+ id
        }
      }
    }
  }

  def rebuildFeatures(tagger: Tagger): Unit = {

  }

  def getId(src: String): Integer = {
    var n: Integer = maxid
    var idx: Integer = 0
    if (dic.get(src) == null) {
      dic.foreach { case (_, con) =>
        con.foreach(pair =>
          con.update(maxid, 1)
        )
      }
      n = maxid
      if (src.charAt(0) == 'U') {
        // Unigram
        maxid += y.size
      }
      else {
        // Bigram
        maxid += y.size * y.size
      }
      return n
    }
    else {
      idx = dic.get(src).get(maxid)
      idx += 1
      dic.foreach { case (_, con) =>
        con.foreach(pair =>
          con.update(maxid, idx)
        )
      }
    }
    -1
  }

  def applyRule(src: String, idx: Integer, tagger: Tagger): String = {
    var dest: String = null
    var r: String = null
    for (i <- 0 until src.length) {
      if (src.charAt(i) == '%') {
        if (src.charAt(i + 1) == 'X') {
          r = getIndex(src.substring(i + 2), idx, tagger)
          dest += r
        }
      } else {
        dest += src.charAt(i)
      }
    }
    dest
  }

  def getIndex(src: String, pos: Integer, tagger: Tagger): String = {
    var neg: Integer = 0
    var col: Integer = 0
    var row: Integer = 0
    var idx: Integer = 0
    var rtn: String = null
    var encol: Boolean = false
    if (src.charAt(0) != '[') {
      null
    }
    if (src.charAt(1) == '-') {
      neg = -1
    }
    for (i <- 1 - neg until src.length - 1) {
      if (src.charAt(i) - '0' <= 9 && src.charAt(i) - '0' >= 0) {
        if (encol == false) {
          row = 10 * row + (src.charAt(i) - '0')
        } else {
          col = 10 * col + (src.charAt(i) - '0')
        }
      } else if (src.charAt(i) == ',') {
        encol = true
      } else {
        0
      }
    }
    row *= neg
    if (row < -kMaxContextSize || row > kMaxContextSize ||
      col < 0 || col >= xsize) {
      0
    }

    max_xsize = math.max(max_xsize, col + 1)

    idx = pos + row
    if (idx < 0) {
      BOS(-idx - 1)
    }
    if (idx >= tagger.x.size) {
      EOS(idx - tagger.x.size)
    }
    tagger.x(idx)(col)
  }

  def setAlpha(_alpha: Array[Double]): Unit = {
    alpha = _alpha
  }

  def initAlpha(size: Integer): Unit = {
    for (i <- 0 until size - 1) {
      alpha :+ 0.0
    }
  }

  def calcCost(n: Node): Unit = {
    var c: Float = 0
    var cd: Double = 0.0
    var idx: Int = 0
    n.cost = 0.0
    if (alpha_float.nonEmpty) {
      while (n.fvector(idx) != -1) {
        c += alpha_float(n.fvector(0) + n.y)
        n.cost = c
        idx += 1
      }
    } else if (alpha.nonEmpty) {
      while (n.fvector(idx) != -1) {
        cd += alpha(n.fvector(0) + n.y)
        n.cost = cd
        idx += 1
      }
    }
  }

  def calcCost(p: Path): Unit = {
    var c: Float = 0
    var cd: Double = 0.0
    var idx: Int = 0
    p.cost = 0.0
    if (alpha_float.nonEmpty) {
      while (p.fvector(idx) != -1) {
        c += alpha_float(p.fvector(0) + p.lnode.y * y.size + p.rnode.y)
        p.cost = c
        idx += 1
      }
    } else if (alpha.nonEmpty) {
      while (p.fvector(idx) != -1) {
        cd += alpha(p.fvector(0) + p.lnode.y * y.size + p.rnode.y)
        p.cost = cd
        idx += 1
      }
    }
  }
}

private[ml] class Allocate extends Serializable {
  var thread_num: Integer = _
  var feature_cache: FeatureCache = _
}
