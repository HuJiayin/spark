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
import org.apache.spark.annotation.DeveloperApi

private[spark] class CRF {
  private val freq: Integer = 1
  private val maxiter: Integer = 100000
  private val cost: Double = 1.0
  private val eta: Double = 0.0001
  // private val threadID: Integer = 1
  // private val modelFile: String = ""
  private val C: Float = 1
  // Convert
  // private val thrinkingSize = 20
  private val threadNum: Integer = Runtime.getRuntime.availableProcessors()
  private val threadPool: Array[CRFThread] = null


  def run(template: String, train: String): Unit = {
    learn(template, train)
  }

  def learn(template: String, train: String): Unit = {
    var tagger: Tagger = new Tagger()
    val taggerList: ArrayBuffer[Tagger] = new ArrayBuffer[Tagger]()
    val featureIndex: FeatureIndex = new FeatureIndex()
    featureIndex.openTemplate(template)
    featureIndex.openTagSet(train)
    tagger.open(featureIndex)
    tagger.read(train)
    taggerList += tagger
    tagger = null
    featureIndex.shrink(freq)
    featureIndex.initAlpha(featureIndex.maxid)
    runCRF(taggerList, featureIndex, featureIndex.alpha)
  }

  def runCRF(tagger: ArrayBuffer[Tagger], featureIndex: FeatureIndex, alpha: Array[Double]): Unit = {
    var diff: Double = 0.0
    var old_obj: Double = 1e37
    var converge: Int = 0
    var itr: Int = 0
    var all: Int = 0
    val lbfgs = new Lbfgs()

    for (i <- 0 until tagger.length - 1) {
      all += tagger(i).x.size
    }

    while (itr <= maxiter) {
      for (i <- 0 until threadNum) {
        threadPool(i).start_i = i
        threadPool(i).size = tagger.size
        threadPool(i).x = tagger
        threadPool(i).start()
        threadPool(i).join()
        threadPool(0).obj += threadPool(i).obj
        threadPool(0).err += threadPool(i).err
        threadPool(0).zeroOne += threadPool(i).zeroOne
        for (k <- 0 until featureIndex.maxid) {
          threadPool(0).expected(k) += threadPool(i).expected(k)
          threadPool(0).obj += alpha(k) * alpha(k) / 2.0 * C
          threadPool(0).expected(k) += alpha(k) / C
        }
      }
      if (itr == 0) {
        diff = 1.0
      } else {
        diff = math.abs(old_obj - threadPool(0).obj / old_obj)
      }
      old_obj = threadPool(0).obj
      printf("iter=%d, terr=%e, serr=%e, act=%d, obj=%e,diff=%e",
        itr, 1.0 * threadPool(0).err / all,
        1.0 * threadPool(0).zeroOne / tagger.size, featureIndex.maxid,
        threadPool(0).obj, diff)
      if (diff < eta) {
        converge += 1
      } else {
        converge = 0
      }
      if (converge == 3) {
        itr = maxiter + 1 // break
      }
      lbfgs.lbfgs(featureIndex.maxid, alpha, threadPool(0).obj, threadPool(0).expected, C)
      itr += 1
    }
  }


  private[ml] class CRFThread extends Thread {
    var x: ArrayBuffer[Tagger] = new ArrayBuffer[Tagger]()
    var start_i: Int = 0
    var err: Int = 0
    var zeroOne: Int = 0
    var size: Int = 0
    var obj: Double = 0.0
    val expected: Array[Double] = null

    override def run: Unit = {
      var idx: Int = 0
      while (idx >= start_i && idx < size) {
        obj += x(idx).gradient(expected)
        err += x(idx).eval()
        if (err != 0) {
          zeroOne += 1
        }
        idx = idx + threadNum
      }
    }
  }

  // def readParameters(template: String, train: String): Unit ={}

}

@DeveloperApi
object CRF {
  def runCRF(template: String, train: String): Unit = {
    val crf = new CRF()
    crf.run(template, train)
  }
}
