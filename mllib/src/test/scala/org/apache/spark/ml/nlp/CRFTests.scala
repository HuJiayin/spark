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

import org.apache.spark.{Logging, SparkFunSuite}
import org.scalatest.{Matchers, BeforeAndAfterAll}

class CRFTests extends SparkFunSuite with BeforeAndAfterAll with Matchers with Logging {
<<<<<<< HEAD
  CRF.verifyCRF("/home/hujiayin/git/CRFConfig/model_file",
    "/home/hujiayin/git/CRFConfig/test_file","/home/hujiayin/git/CRFConfig/test_result")
  CRF.runCRF("/home/hujiayin/git/CRFConfig/template_file",
    "/home/hujiayin/git/CRFConfig/train_file")
}
=======
  CRF.runCRF("/home/hujiayin/Downloads/template_file", "/home/hujiayin/Downloads/train_file")
}
>>>>>>> 575fb04776f6334c691eb3d5e7d00c319ae03548
