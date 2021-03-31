package scalation.analytics

import MatrixTransform._
import RegTechnique._
import TranRegression._
import scalation.util.banner
import scala.math._
import scalation.linalgebra._
import scalation.math._
import scalation.plot._
import scalation.analytics.ExampleAutoMPG._
import scalation.analytics.TranRegression.{box_cox, cox_box}
import Perceptron._
import ActivationFun._
import Initializer._
import Optimizer._                                  // Optimizer - configuration
//import Optimizer_SGD._                            // Stochastic Gradient Descent
import Optimizer_ADAM._                             // Stochastic Gradient Descent with Momentum
import PredictorMat2._
import StoppingRule._
import NeuralNet_3L._

object TranRegressionsel extends App{
  def recip (y: Double): Double = 1.0 / y
//    val f = (recip _ , recip _, "recip")
//  val f = (log _ , exp _, "log")
       val f = (sqrt _ , sq _, "sqrt")
  //     val f = (sq _ , sqrt _, "sq")
  //     val f = (exp _ , log _, "exp")
  //  TranRegression.setLambda (-1.0); val f = (box_cox _ , cox_box _, "box_cox")
  println (s"TranRegression with ${f._3} transform")
  TranRegression.rescaleOff ()
  val trg = TranRegression (ox, y, null, null, f._1, f._2, QR, null)
  println (trg.analyze ().report)
  println (trg.summary)

  banner ("TranRegression forwardSel")
  val (cols, rSq) = trg.forwardSelAll () // R^2, R^2 bar, R^2 cv
  val k = cols.size
  println (s"k = $k, n = ${x.dim2}")
  val t = VectorD.range (1, k) // instance index
  new PlotM (t, rSq.t, Array ("R^2", "R^2 bar", "R^2 cv"),
    "R^2 vs n for TranRegression forwardSel", lines = true).saveImage("TranRegression forwardSel")
  println (s"rSq = $rSq")

  banner ("TranRegression backwardElim")
  val (colsb, rSqb) = trg.backwardElimAll ()   // R^2, R^2 bar, R^2 cv
  val kb = colsb.size
  println (s"k = $kb, n = ${x.dim2}")
  val tb = VectorD.range (1, rSqb.dim1) // instance index
  new PlotM (tb, rSqb.t, Array ("R^2", "R^2 bar", "R^2 cv"),
    "R^2 vs n for TranRegression backwardElim", lines = true).saveImage("TranRegression backwardElim")
  println (s"rSqb = $rSqb")

  banner ("TranRegression stepwise")
  val (colss, rSqs) = trg.stepwiseSelAll ()   // R^2, R^2 bar, R^2 cv
  val ks = colss.size
  println (s"k = $ks, n = ${x.dim2}")
  val ts = VectorD.range (1, rSqs.dim1) // instance index
  new PlotM (ts, rSqs.t, Array ("R^2", "R^2 bar", "R^2 cv"),
    "R^2 vs n for TranRegression stepwise", lines = true).saveImage("TranRegression stepwise")
  println (s"rSqs = $rSqs")
}

object Perceptronsel extends App{
  val f_ = f_sigmoid                                              // try different activation function
//  val f_ = f_id                                                   // try different activation function
  val nn = Perceptron (oxy, f0 = f_)                              // factory function automatically rescales
  //  val nn = new Perceptron (ox, y, f0 = f_)                        // constructor does not automatically rescale
  nn.reset (eta_ = 0.001)                                         // try several values - for train0
  nn.reset (eta_ = 0.03)                                          // try several values - for train1, 2
  nn.trainSwitch (2).eval ()                                      // fit the parameters using the dataset
  println (nn.report)
  println (nn.summary)

  banner ("Perceptron forwardSel")
  val (cols, rSq) = nn.forwardSelAll () // R^2, R^2 bar, R^2 cv
  val k = cols.size
  println (s"k = $k, n = ${x.dim2}")
  val t = VectorD.range (1, k) // instance index
  new PlotM (t, rSq.t, Array ("R^2", "R^2 bar", "R^2 cv"),
    "R^2 vs n for Perceptron forwardSel", lines = true).saveImage("Perceptron forwardSel")
  println (s"rSq = $rSq")

  banner ("Perceptron backwardElim")
  val (colsb, rSqb) = nn.backwardElimAll ()   // R^2, R^2 bar, R^2 cv
  val kb = colsb.size
  println (s"k = $kb, n = ${x.dim2}")
  val tb = VectorD.range (1, rSqb.dim1) // instance index
  new PlotM (tb, rSqb.t, Array ("R^2", "R^2 bar", "R^2 cv"),
    "R^2 vs n for Perceptron backwardElim", lines = true).saveImage("Perceptron backwardElim")
  println (s"rSqb = $rSqb")

  banner ("Perceptron stepwise")
  val (colss, rSqs) = nn.stepwiseSelAll ()   // R^2, R^2 bar, R^2 cv
  val ks = colss.size
  println (s"k = $ks, n = ${x.dim2}")
  val ts = VectorD.range (1, rSqs.dim1) // instance index
  new PlotM (ts, rSqs.t, Array ("R^2", "R^2 bar", "R^2 cv"),
    "R^2 vs n for Perceptron stepwise", lines = true).saveImage("Perceptron stepwise")
  println (s"rSqs = $rSqs")

}

object NeuralNet_3Lsel extends App{
  banner ("NeuralNet_3L with scaled y values")
  //  hp("eta") = 0.0014                                              // try several values - train0
  hp("eta") = 0.1                                                 // try several values - train

  val nn = NeuralNet_3L (xy)                                      // factory function automatically rescales
  //  val nn = new NeuralNet_3L (x, MatrixD (Seq (y)))                // constructor does not automatically rescale

  nn.trainSwitch (2).eval ()                                      // fit the weights using training data (0, 1, 2)
  println (nn.report)

  banner ("NeuralNet_3L forwardSel")
  val (cols, rSq) = nn.forwardSelAll ()                          // R^2, R^2 bar, R^2 cv
  println (s"rSq = $rSq")
  val k = cols.size
//  println (s"k = $k, n = $n")
  val t = VectorD.range (1, k)                                   // instance index
  new PlotM (t, rSq.t, Array ("R^2", "R^2 bar", "R^2 cv"),
    "R^2 vs n for NeuralNet_3L forwardSel", lines = true).saveImage("NeuralNet_3L forwardSel")

  banner ("NeuralNet_3L backwardElim")
  val (colsb, rSqb) = nn.backwardElimAll ()   // R^2, R^2 bar, R^2 cv
  val kb = colsb.size
//  println (s"k = $kb, n = ${x.dim2}")
  val tb = VectorD.range (1, rSqb.dim1) // instance index
  new PlotM (tb, rSqb.t, Array ("R^2", "R^2 bar", "R^2 cv"),
    "R^2 vs n for NeuralNet_3L backwardElim", lines = true).saveImage("NeuralNet_3L backwardElim")
  println (s"rSqb = $rSqb")

  banner ("NeuralNet_3L stepwise")
  val (colss, rSqs) = nn.stepwiseSelAll ()   // R^2, R^2 bar, R^2 cv
  val ks = colss.size
  val ts = VectorD.range (1, rSqs.dim1) // instance index
  new PlotM (ts, rSqs.t, Array ("R^2", "R^2 bar", "R^2 cv"),
    "R^2 vs n for NeuralNet_3L stepwise", lines = true).saveImage("NeuralNet_3L stepwise")
  println (s"rSqs = $rSqs")

}

object NeuralNet_XLsel extends App{
  val n = ox.dim2                                                // number of parameters/variables
  val af_ = Array (f_sigmoid, f_sigmoid, f_id)                   // try different activation functions
  //  val af_ = Array (f_tanh, f_tanh, f_id)                         // try different activation functions

  banner ("NeuralNet_XL with scaled y values")
  hp("eta") = 0.02                                               // learning rate hyoer-parameter (see Optimizer)
  val nn  = NeuralNet_XL (oxy, af = af_)                         // factory function automatically rescales
  //  val nn  = new NeuralNet_XL (ox, y, af = af_)                   // constructor does not automatically rescale

  nn.train ().eval ()                                            // fit the weights using training data
  println (nn.report)                                            // report parameters and fit
  val ft  = nn.fitA(0)                                           // fit for first output variable

  banner ("NeuralNet_XL forwardSel")
  val (cols, rSq) = nn.forwardSelAll ()                          // R^2, R^2 bar, R^2 cv
  println (s"rSq = $rSq")
  val k = cols.size
  println (s"k = $k, n = $n")
  val t = VectorD.range (1, k)                                   // instance index
  new PlotM (t, rSq.t, Array ("R^2", "R^2 bar", "R^2 cv"),
    "R^2 vs n for NeuralNet_XL", lines = true).saveImage("NeuralNet_XL forwardSel")
  banner ("NeuralNet_XL backwardElim")
  val (colsb, rSqb) = nn.backwardElimAll ()   // R^2, R^2 bar, R^2 cv
  val kb = colsb.size
  println (s"k = $kb, n = ${x.dim2}")
  val tb = VectorD.range (1, rSqb.dim1) // instance index
  new PlotM (tb, rSqb.t, Array ("R^2", "R^2 bar", "R^2 cv"),
    "R^2 vs n for NeuralNet_XL backwardElim", lines = true).saveImage("NeuralNet_XL backwardElim")
  println (s"rSqb = $rSqb")
  banner ("NeuralNet_XL stepwise")
  val (colss, rSqs) = nn.stepwiseSelAll ()   // R^2, R^2 bar, R^2 cv
  val ks = colss.size
  println (s"k = $ks, n = ${x.dim2}")
  val ts = VectorD.range (1, rSqs.dim1) // instance index
  new PlotM (ts, rSqs.t, Array ("R^2", "R^2 bar", "R^2 cv"),
    "R^2 vs n for NeuralNet_XL stepwise", lines = true).saveImage("NeuralNet_XL stepwise")
  println (s"rSqs = $rSqs")
}