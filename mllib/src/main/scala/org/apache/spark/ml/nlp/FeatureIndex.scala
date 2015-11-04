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

import scala.io.Source._
import scala.collection.mutable.ArrayBuffer

private[ml] class FeatureIndex extends Serializable {
  var maxid: Int = 0
  var alpha: ArrayBuffer[Double] = ArrayBuffer[Double]()
  var alpha_float: ArrayBuffer[Float] = ArrayBuffer[Float]()
  var cost_factor: Double = 0.0
  var xsize: Integer = 0
  var check_max_xsize: Boolean = false
  var max_xsize: Int = 0
  var unigram_templs: ArrayBuffer[String] = new ArrayBuffer[String]()
  var bigram_templs: ArrayBuffer[String] = new ArrayBuffer[String]()
  var y: Set[String] = Set[String]()
  var templs: String = new String
  //var dic: Map[String, Map[Int, Int]] = Map[String, Map[Int, Int]]()
  var dic: scala.collection.mutable.Map[String, (Int,Int)] =
    scala.collection.mutable.Map[String, (Int,Int)]()
  val kMaxContextSize: Int = 8
  val BOS = Vector[String]("_B-1", "_B-2", "_B-3", "_B-4",
    "_B-5", "_B-6", "_B-7", "_B-8")
  val EOS = Vector[String]("_B+1", "_B+2", "_B+3", "_B+4",
    "_B+5", "_B+6", "_B+7", "_B+8")
  val featureCache: ArrayBuffer[Int] = new ArrayBuffer[Int]()
  val featureCacheH: ArrayBuffer[Int] = new ArrayBuffer[Int]()

  def getFeatureCacheIdx(fVal: Int): Int = {
    var i: Int = 0
    while(i < featureCache.size){
      if(featureCache(i) == fVal){
        return i
      }
      i += 1
    }
    0
  }

  def getFeatureCache(): ArrayBuffer[Int] = {
    featureCache
  }

  def getFeatureCacheH(): ArrayBuffer[Int] = {
    featureCacheH
  }

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
        tag = line(i).split('|')
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
    var newMaxId: Int = 0
    val key: String = null
    val count: Int = 0
    val currId: Int = 0

    if (freq > 1) {
      while (dic.iterator.next() != null) {
         /* dic.foreach { case (_, con) =>
          con.foreach(pair => if (pair._2 > freq) {
            con.update(newMaxId, pair._1)
          } else {
            dic - key
          })
        } */
        dic.getOrElse(key,(currId,count))
        if(count > freq) {
          dic.getOrElseUpdate(key, (newMaxId, count))
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

  def rebuildFeatures(tagger: Tagger): Unit = {
    var cur: Int = 0
    var i: Int = 0
    var j: Int = 0
    var fid = tagger.feature_id
    // var thead_id = tagger.thread_id
    // tagger.node = new ArrayBuffer[ArrayBuffer[Node]](tagger.x.size)
    val nodeList: ArrayBuffer[Node] = new ArrayBuffer[Node]()
    var nd = new Node
    var path = new Path

    while(cur < tagger.x.size){
      while(i < tagger.ysize){
        // tagger.node(_) = new ArrayBuffer[Node](i + 1)
        nd = new Node
        nd.x = cur
        nd.y = i
        nd.fvector = featureCacheH(fid)
        nodeList.append(nd)
        //tagger.node(cur)(i) = nd
        i += 1
      }
      fid += 1
      tagger.node.append(nodeList)
      cur += 1
    }

    cur = 1
    i = 0
    while(cur < tagger.x.size){
      while(j < tagger.ysize){
        while(i< tagger.ysize){
          path = new Path
          path.add(tagger.node(cur - 1)(j),tagger.node(cur)(i))
          path.fvector = featureCacheH(fid)
          i += 1
        }
        j += 1
      }
      cur += 1
    }
  }

  def buildFeatures(tagger: Tagger): Unit = {
    var os: String = null
    var id: Integer = 0
    var cur: Int = 0
    var it: Int = 0
    featureCacheH.append(0)
    while (cur < tagger.x.size) {
      while (it < unigram_templs.length) {
        os = applyRule(unigram_templs(it), cur, tagger)
        id = getId(os)
        featureCache.append(id)
        it += 1
      }
      featureCache.append(-1)
      featureCacheH.append(maxid)
      cur += 1
      it = 0
    }
    it = 0
    cur = 1
    while (cur < tagger.x.size) {
      while (it < bigram_templs.length) {
        os = applyRule(bigram_templs(it), cur, tagger)
        id = getId(os)
        featureCache.append(id)
        it += 1
      }
      featureCache.append(-1)
      featureCacheH.append(maxid)
      cur += 1
      it = 0
    }
  }

  def getId(src: String): Integer = {
    var n: Integer = maxid
    var idx: Integer = 0
    if (dic.get(src).isEmpty) {
      /* dic.foreach { case (src, con) =>
        con.foreach(pair =>
          con.update(maxid, 1)
        )
      } */
      dic.update(src,(maxid,1))
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
      // idx = dic.get(src).get(maxid)
      idx = dic.get(src).get._2
      idx += 1
      dic.update(src, (maxid, idx))
      /* dic.foreach { case (_, con) =>
        con.foreach(pair =>
          con.update(maxid, idx)
        )
      } */
      return maxid
    }
    -1
  }

  def applyRule(src: String, idx: Integer, tagger: Tagger): String = {
    var dest: String = ""
    var r: String = ""
    var i: Int = 0
    while (i < src.length) {
      if (src.charAt(i) == '%') {
        if (src.charAt(i + 1) == 'x') {
          r = getIndex(src.substring(i + 2), idx, tagger)
          if(r==null){
            return null
          }
          dest += r
        }
      } else {
        dest += src.charAt(i)
      }
      i += 1
    }
    dest
  }

  def getIndex(src: String, pos: Integer, tagger: Tagger): String = {
    var neg: Integer = 1
    var col: Integer = 0
    var row: Integer = 0
    var idx: Integer = 0
    var rtn: String = null
    var encol: Boolean = false
    var i: Int = 0
    if (src.charAt(0) != '[') {
      return null
    }
    i += 1
    if (src.charAt(1) == '-') {
      neg = -1
      i += 1
    }
    while (i < src.length) {
      if (src.charAt(i) - '0' <= 9 && src.charAt(i) - '0' >= 0) {
        if (!encol) {
          row = 10 * row + (src.charAt(i) - '0')
        } else {
          col = 10 * col + (src.charAt(i) - '0')
        }
      } else if (src.charAt(i) == ',') {
        encol = true
      } else if (src.charAt(i) == ']'){
        i = src.length // break
      }
      i += 1
    }
    row *= neg
    if (row < -kMaxContextSize || row > kMaxContextSize ||
      col < 0 || col >= xsize) {
      return null
    }

    max_xsize = math.max(max_xsize, col + 1)

    idx = pos + row
    if (idx < 0) {
      return BOS(-idx - 1)
    }
    if (idx >= tagger.x.size) {
      return EOS(idx - tagger.x.size)
    }
    tagger.x(idx)(col)
  }

  def setAlpha(_alpha: ArrayBuffer[Double]): Unit = {
    alpha = _alpha
  }

  def initAlpha(size: Integer): Unit = {
    var i: Int = 0
    // alpha = new ArrayBuffer[Double](size)
    while(i < size){
      alpha.append(0.0)
      i += 1
    }
    // alpha_float = new Array[Float](size)
  }

  def calcCost(n: Node): Unit = {
    var c: Float = 0
    var cd: Double = 0.0
    var idx: Int = n.fvector

    n.cost = 0.0
    if (alpha_float.nonEmpty) {
      while (featureCache(idx) != -1) {
        c += alpha_float(featureCache(idx) + n.y)
        n.cost = c
        idx += 1
      }
    } else if (alpha.nonEmpty) {
      while (featureCache(idx) != -1) {
        cd += alpha(featureCache(idx) + n.y)
        n.cost = cd
        idx += 1
      }
    }
  }

  def calcCost(p: Path): Unit = {
    var c: Float = 0
    var cd: Double = 0.0
    var idx: Int = p.fvector
    var pivot: Int = 0
    p.cost = 0.0
    while(pivot < featureCache.size){
      if(featureCache(pivot)==idx){
        idx = pivot
        pivot = featureCache.size
      }
      pivot += 1
    }
    if (alpha_float.nonEmpty) {
      while (featureCache(idx) != -1) {
        c += alpha_float(featureCache(idx) +
          p.lnode.y * y.size + p.rnode.y)
        p.cost = c
        idx += 1
      }
    } else if (alpha.nonEmpty) {
      while (featureCache(idx) != -1) {
        cd += alpha(featureCache(idx) +
          p.lnode.y * y.size + p.rnode.y)
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
