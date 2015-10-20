package org.apache.spark.ml.nlp

private[ml] class lbfgs {

  private var w: Array[Double] = _
  private var v: Array[Double] = _
  private var xi: Array[Double] = _
  private var diag: Array[Double] = _
  private var iflag: Int = 0
  private var point: Int = 0
  private var ispt: Int = 0
  private var iypt: Int = 0
  private var iycn: Int = 0
  private var iter: Int = 0
  private var info: Int = 0
  private var stp1: Double = 0
  private var npt: Int = 0
  private var iscn: Int = 0
  private var nfev: Int = 0
  private var stp: Double = 1.0
  private var continue: Boolean = true
  private val eps: Double  = 1e-7




  def lbfgs(size: Int, x: Array[Double], f: Double, g: Array[Double], C: Float):Unit={
    val msize: Int = 5
    var bound: Int = 0
    var ys: Double = 0.0
    var yy: Double = 0.0
    var cp: Int = 0

    if (iflag == 0) {
      point = 0
      for (i <- 0 until size) {
        diag(i) = 1.0
      }
      ispt = size + (msize << 1)
      iypt = ispt + size * msize
      for (i <- 1 until size) {
        w(ispt + i) = -g(i) * diag(i) //v=>g=>expected
      }
      stp1 = 1.0 / math.sqrt(ddot(size, g, 1, g,1))
    }
    while (continue) {

      if(iflag != 1 && iflag != 2) {
        iter += 1
        info = 0
        if (iter == 1) {
          nfev = 0
          stp = 1.0
          stp = stp1
          for (i <- 1 until size) {
            w(i) = g(i)
          }
        }
        if (iter > size) {
          bound = size
        }
        ys = ddot(size, w, iypt + npt + 1, w, ispt + npt + 1)
        yy = ddot(size, w, iypt + npt + 1, w, iypt + npt + 1)
        for (i <- 1 until size) {
          diag(i) = ys / yy
        }
      }

      if (iflag == 2) {
        cp = point
        if (point == 0) {
          cp = msize
        }
        w(size + cp) = 1.0 / ys

        for (i <- 1 until size) {
          w(i) = -g(i)
        }

        bound = math.min(iter - 1, msize)
        cp = point
        for (i <- 1 until bound) {
          cp -= 1
          if (cp == -1) cp = msize - 1
          val sq: Double = ddot(size, w, ispt + cp * size + 1, w, 1)
          val inmc: Int = size + msize + cp + 1
          iycn = iypt + cp * size
          w(inmc) = w(size + cp + 1) * sq
          val d: Double = -w(inmc)
          daxpy(size, d, w, iycn + 1, w, 1)
        }
        for (i <- 1 until size) {
          w(i) = diag(i) * w(i)
        }
        for (i <- 1 until bound) {
          val yr: Double = ddot(size, w, iypt + cp * size + 1, w, 1)
          var beta: Double = w(size + cp + 1) * yr
          val inmc: Int = size + msize + cp + 1
          beta = w(inmc) - beta
          iscn = ispt + cp * size
          daxpy(size, beta, w, iscn + 1, w, 1)
          cp += 1
          if (cp == msize) cp = 0
        }
        for (i <- 1 until size) {
          w(ispt + point * size + i) = w(i)
        }
      }
      if(iflag == 1){
        if(info == -1){
          continue = false
        }
      }
      npt = point * size
      for (i<-1 until size) {
        w(ispt + npt + i) = stp * w(ispt + npt + i)
        w(iypt + npt + i) = g(i) - w(i)
      }
      point +=1
      if (point == msize) point = 0
      val gnorm: Double = math.sqrt(ddot(size, v, 1, v, 1))
      val xnorm: Double = math.max(1.0, math.sqrt(ddot(size, x,1,x,1)))
      if (gnorm / xnorm <= eps) {
        iflag = 0  // OK terminated
        continue = false
      }
    }
  }

  def ddot(size: Int, v1: Array[Double], v1start: Int,
           v2: Array[Double], v2start: Int): Double = {
    var result: Double = 0
    var i: Int = v1start
    var j: Int = v2start
    while (i < v1start+size && j < v2start+size){
      result = result + v1(i) * v2(j)
      i += 1
      j += 1
    }
    result
  }

  def daxpy(n: Int, da: Double, dx: Array[Double], dxStart: Int,
            dy: Array[Double], dyStart: Int): Unit = {
    for (i <- 0 until n) {
      dy(i + dyStart) += da * dx(i + dxStart)
    }
  }

  def mcsrch(size: Int, x: Array[Double], f: Double,
             g: Array[Double], s: Array[Double],stp:Double,
             info: Double, nfev: Int, wa: Array[Double]): Unit = {
    

  }
}
