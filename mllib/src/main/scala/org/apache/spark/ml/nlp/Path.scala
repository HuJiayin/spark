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
  def calExpectation(expected : Array[Double], Z: Double, size: Integer): Unit = {
    var c: Double = math.exp(lnode.alpha + cost + rnode.beta - Z)
    while(fvector(fIdx)!= -1) {
      expected(fvector(0) + lnode.y*size + rnode.y) += c
      fIdx += 1
    }
  }

  def add(_lnode : Node, _rnode: Node): Unit = {
    lnode = _lnode
    rnode = _rnode
    lnode.rpath :+ lnode
    rnode.lpath :+ rnode
  }

  def clear(): Unit = {
    cost = 0
  }

}
