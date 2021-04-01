
//::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::
/** @author  John Miller, Hao Peng
 *  @version 1.6
 *  @date    Fri Mar 16 15:13:38 EDT 2018
 *  @see     LICENSE (MIT style license file).
 *
 *  @title   Model: Neural Network with 4+ Layers (input, multiple hidden, output layers)
 *
 *  @see     hebb.mit.edu/courses/9.641/2002/lectures/lecture03.pdf
 *  @see     http://neuralnetworksanddeeplearning.com/
 */


package scalation.analytics

import scala.collection.mutable.Set
import scala.math.{max => MAX}

import scalation.linalgebra.{FunctionV_2V, MatriD, MatrixD, VectoD, VectorD}
import scalation.math.noDouble
import scalation.plot.PlotM
import scalation.stat.Statistic
import scalation.util.banner

import ActivationFun._
import Fit._
import Initializer._
import MatrixTransform._
import Optimizer._                                  // Optimizer - configuration
//import Optimizer_SGD._                            // Stochastic Gradient Descent
import Optimizer_SGDM._                             // Stochastic Gradient Descent with Momentum

import PredictorMat2._

//::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::
/** The `NeuralNet_XL` class supports multi-output, multi-layer (input, multiple hidden and output)
 *  Neural-Networks.  It can be used for both classification and prediction,
 *  depending on the activation functions used.  Given several input vectors and output
 *  vectors (training data), fit the weight and bias parameters connecting the layers,
 *  so that for a new input vector 'v', the net can predict the output value.
 *  Defaults to two hidden layers.
 *  This implementation is partially adapted from Michael Nielsen's Python implementation found in
 *  @see  github.com/mnielsen/neural-networks-and-deep-learning/blob/master/src/network2.py
 *  @see  github.com/MichalDanielDobrzanski/DeepLearningPython35/blob/master/network2.py
 *------------------------------------------------------------------------------
 *  @param x       the m-by-nx data/input matrix (training data having m input vectors)
 *  @param y       the m-by-ny response/output matrix (training data having m output vectors)
 *  @param nz      the number of nodes in each hidden layer, e.g., Array (9, 8) => 2 hidden of sizes 9 and 8 
 *  @param fname_  the feature/variable names (if null, use x_j's)
 *  @param hparam  the hyper-parameters for the model/network
 *  @param f       the array of activation function families between every pair of layers
 *  @param itran   the inverse transformation function returns responses to original scale
 */
class NeuralNet_XL (x: MatriD, y: MatriD,
                    private var nz: Array [Int] = null,
                    fname_ : Strings = null, hparam: HyperParameter = hp,
                    f:  Array [AFF] = Array (f_tanh, f_tanh, f_id),
                    val itran: FunctionV_2V = null)
      extends PredictorMat2 (x, y, fname_, hparam)                               // sets eta in parent class
{
    private val DEBUG     = false                                                // debug flag
    private val bSize     = hp ("bSize").toInt                                   // mini-batch size
    private val maxEpochs = hp ("maxEpochs").toInt                               // maximum number of training epochs/iterations
    private val lambda    = hp ("lambda")                                        // regularization hyper-parameter

    if (nz == null) nz = compute_nz (nx)                                         // [1] default number of nodes for hidden layers
//  if (nz == null) nz = compute_nz (nx, ny)                                     // [2] default number of nodes for hidden layers
    val df_m = compute_df_m (nz)                                                 // degrees of freedom for model (first output only)
    resetDF (df_m, x.dim1 - df_m)                                                // degrees of freedom for (model, error)

    if (f.length != nz.length + 1) {
        flaw ("NeuralNet_XL Constructor", "dimension mismatch among number of layers or activation functions")
    } // if

    protected val sizes  = nx +: nz :+ ny                                        // sizes of all layers
    protected val nl     = sizes.length - 1                                      // number of active layers
    protected val layers = 0 until nl

    protected var b = for (l <- layers) yield
                       new NetParam (weightMat (sizes(l), sizes(l+1)),           // parameters (weights & 
                                     weightVec (sizes(l+1)))                     // biases) per active layer

    println (s"Create a NeuralNet_XL with $nx input, ${nz.deep} hidden and $ny output nodes: df_m = $df_m")

    //::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::
    /** Compute default values for the number nodes in each hidden layer, based on
     *  the number of nodes in the input layer using the drop one/two rule.
     *  Rule [1] nx, nx - 2, ...
     *  @param nx  the number of nodes in the input layer
     */
    def compute_nz (nx: Int): Array [Int] =
    {
        (for (l <- 1 until f.length) yield MAX (1, nx + 2 - 2*l)).toArray
    } // compute_nz

    //::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::
    /** Compute default values for the number nodes in each hidden layer, based on
     *  the number of nodes in the input and output layers using average of prior layer
     *  and output layer rule.
     *  Rule [2] (nx + ny) / 2, (nx + 3ny) / 4, ...
     *  @param nx  the number of nodes in the input layer
     *  @param ny  the number of nodes in the output layer
     */
    def compute_nz (nx: Int, ny: Int): Array [Int] =
    {
        val n = Array.ofDim [Int] (f.length - 1)
        for (l <- 0 until f.length - 1) n(l) = if (l == 0) (nx + ny) / 2 else (n(l-1) + ny) / 2
        n
    } // compute_nz

    //::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::
    /** Compute the degrees of freedom for the model (based on nx, n's, ny = 1).
     *  Rough extimate based on total number of parameters - 1.
     *  FIX: use better estimate
     *  @param n  the number of nodes in each hidden layer
     */
    def compute_df_m (n: Array [Int]): Int =
    {
        var sum = n.last
        for (l <- n.indices) {
            if (l == 0) sum += nx * n(0) + n(0)
            else        sum += n(l-1) * n(l) + n(l)
        } // for
        sum
    } // compute_df_m

    //::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::
    /** Return the parameters (weight matrices and bias vectors).
     */
    def parameters: NetParams = b

    //::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::
    /** Given training data 'x_r' and 'y_r', fit the parameters 'b' (weight matrices and
     *  bias vectors).  Iterate over several epochs (no batching).
     *  b.w(l) *= 1.0 - eta * (lambda / m)     // regularization factor, weight decay
     *  @param x_r  the training/full data/input matrix
     *  @param y_r  the training/full response/output matrix
     */
    def train0 (x_r: MatriD = x, y_r: MatriD = y): NeuralNet_XL =
    {
        println (s"train0: eta = $eta")
        var sse0 = Double.MaxValue                                               // hold prior value of sse
        val z    = Array.ofDim [MatriD] (nl+1); z(0) = x_r                       // storage: activations f(b(l), z(l))
        val d    = Array.ofDim [MatriD] (nl)                                     // storage: deltas

        for (epoch <- 1 to maxEpochs) {                                          // iterate over each epoch
            for (l <- layers) z(l+1) = f(l).fM (b(l) * z(l))                     // feedforward and store activations
            ee      = z.last - y_r                                               // negative of error matrix
            d(nl-1) = f.last.dM (z.last) ** ee                                   // delta for last layer before output
            for (l <- nl-2 to 0 by -1)
                d(l) = f(l).dM (z(l+1)) ** (d(l+1) * b(l+1).w.t)                 // deltas for previous hidden layers
            for (l <- layers) b(l) -= (z(l).t * d(l) * eta, d(l).mean * eta)     // update parameters (weights, biases)

            val sse = ee.normFSq
            if (DEBUG) println (s"train0: parameters for $epoch th epoch: b = $b, sse = $sse")
            if (sse > sse0) return this                                          // return early if moving up
            sse0 = sse                                                           // save prior sse
        } // for
        this
    } // train0

    //::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::
    /** Given training data 'x_r' and 'y_r', fit the parameters 'b' (weight matrices and
     *  bias vectors).  Iterate over several epochs, where each epoch divides the
     *  training set into 'nbat' batches.  Each batch is used to update the weights.
     *  @param x_r  the training/full data/input matrix
     *  @param y_r  the training/full response/output matrix
     */
    def train (x_r: MatriD = x, y_r: MatriD = y): NeuralNet_XL =
    {
        val epochs = optimizeX (x_r, y_r, b, eta, bSize, maxEpochs, lambda, f)    // optimize parameters (weights & biases)
        println (s"ending epoch = $epochs")
        estat.tally (epochs._2)
        this
    } // train

    //::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::
    /** Given training data 'x_r' and 'y_r', fit the parameters 'b' (weight matrices and
     *  bias vectors).  Iterate over several epochs, where each epoch divides the
     *  training set into 'nbat' batches.  Each batch is used to update the weights.
     *  This version preforms an interval search for the best 'eta' value.
     *  @param x_r  the training/full data/input matrix
     *  @param y_r  the training/full response/output matrix
     */
    override def train2 (x_r: MatriD = x, y_r: MatriD = y): NeuralNet_XL =
    {
        val etaI = (0.25 * eta, 4.0 * eta)                                       // quarter to four times eta
        val epochs = optimizeXI (x_r, y_r, b, etaI, bSize, maxEpochs, lambda, f)  // optimize parameters (weights & biases)
        println (s"ending epoch = $epochs")
        estat.tally (epochs._2)
        this
    } // train2

    //::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::
    /** Build a sub-model that is restricted to the given columns of the data matrix.
     *  @param x_cols  the columns that the new model is restricted to
     */
    def buildModel (x_cols: MatriD): NeuralNet_XL =
    {
        new NeuralNet_XL (x_cols, y, null, null, hparam, f, itran)
    } // buildModel

    //::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::
    /** Return the network parameters (weights and biases) for the given 'layer'.
     *  @param layer  the layer to get the parameters from
     */
    def getNetParam (layer: Int = 1) = b(layer)

    //::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::
    /** Given a new input vector 'v', predict the output/response vector 'f(v)'.
     *  @param v  the new input vector
     */
    def predictV (v: VectoD): VectoD =
    {
        var u = v
        for (l <- layers) u = f(l).fV (b(l) dot u)
        u
    } // predictV

    //::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::
    /** Given an input matrix 'x', predict the output/response matrix 'f(x)'.
     *  @param v  the input matrix
     */
    def predictV (v: MatriD = x): MatriD =
    {
        var u = v
        for (l <- layers) u = f(l).fM (b(l) * u)
        u
    } // predictV

} // NeuralNet_XL class


//::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::
/** The `NeuralNet_XL` companion object provides factory functions for buidling multi-layer
 *  neural nets (defaults to two hidden layers).
 *  Note, 'rescale' is defined in `ModelFactory` in Model.scala.
 */
object NeuralNet_XL extends ModelFactory
{
    private val DEBUG = false                                          // debug flag

    //::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::
    /** Create a `NeuralNet_XL` for a combined data matrix.
     *  @param xy      the combined input and output matrix
     *  @param nz      the number of nodes in each hidden layer, e.g., Array (5, 10) means 2 hidden with sizes 5 and 10
     *  @param fname   the feature/variable names
     *  @param hparam  the hyper-parameters
     *  @param af      the array of activation function families over all layers
     */
    def apply (xy: MatriD, nz: Array [Int] = null, 
               fname: Strings = null, hparam: HyperParameter = Optimizer.hp,
               af: Array [AFF] = Array (f_tanh, f_tanh, f_id)): NeuralNet_XL =
    {
        var itran: FunctionV_2V = null                                 // inverse transform -> original scale
        val (x, y) = pullResponse (xy)                                 // assumes the last column is the response

        val x_s = if (rescale) rescaleX (x, af(0))
                  else x
        val y_s = if (af.last.bounds != null) { val y_i = rescaleY (y, af.last); itran = y_i._2; y_i._1 }
                  else y

        if (DEBUG) println (s" scaled: x = $x_s \n scaled y = $y_s")
        new NeuralNet_XL (x_s, MatrixD (Seq (y_s)), nz, fname, hparam, af, itran)
    } // apply

    //::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::
    /** Create a `NeuralNet_XL` for a data matrix and response vector.
     *  @param x       the input/data matrix
     *  @param y       the output/response vector
     *  @param nz      the number of nodes in each hidden layer, e.g., Array (5, 10) means 2 hidden with sizes 5 and 10
     *  @param fname   the feature/variable names
     *  @param hparam  the hyper-parameters
     *  @param af      the array of activation function families over all layers
     */
    def apply (x: MatriD, y: VectoD, nz: Array [Int],
               fname: Strings, hparam: HyperParameter,
               af: Array [AFF]): NeuralNet_XL =
    {
        val hp2 = if (hparam == null) Optimizer.hp else hparam
        var itran: FunctionV_2V = null                                 // inverse transform -> original scale

        val x_s = if (rescale) rescaleX (x, af(0))
                  else x
        val y_s = if (af.last.bounds != null) { val y_i = rescaleY (y, af.last); itran = y_i._2; y_i._1 }
                  else y

        if (DEBUG) println (s" scaled: x = $x_s \n scaled y = $y_s")
        new NeuralNet_XL (x_s, MatrixD (Seq (y_s)), nz, fname, hp2, af, itran)
    } // apply

} // NeuralNet_XL object


//::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::
/** The `NeuralNet_XLTest` object is used to test the `NeuralNet_XL` class.
 *  @see www4.rgu.ac.uk/files/chapter3%20-%20bp.pdf
 *  > runMain scalation.analytics.NeuralNet_XLTest
 */
object NeuralNet_XLTest extends App
{
    val s = 0                                                           // random number stream to use
    val x = new MatrixD ((3, 3), 1.0, 0.35, 0.9,                        // training data - input matrix (m vectors)
                                 1.0, 0.20, 0.7,
                                 1.0, 0.40, 0.95)
    val y = new MatrixD ((3, 2), 0.5, 0.4,                              // training data - output matrix (m vectors)
                                 0.3, 0.3,
                                 0.6, 0.5)

    println ("input  matrix x = " + x)
    println ("output matrix y = " + y)

    val hp2 = hp.updateReturn ("bSize", 1)
    val nn  = new NeuralNet_XL (x, y, Array (3, 2), hparam = hp2)       // create a NeuralNet_XL

    for (i <- 1 to 20) {
        val eta = i * 0.5
        banner (s"NeuralNet_XLTest: Fit the parameter b using optimization with learning rate $eta")

        nn.reset (eta_ = eta)
        nn.train ().eval ()                                             // fit the weights using training data
        println (nn.report)

//      yp = nn.predict (x)                                             // predicted output values
//      println ("target output:    y   = " + y)
//      println ("predicted output: yp  = " + yp)
        println ("yp0 = " + nn.predict (x(0)))                          // predicted output values for row 0
    } // for

    banner ("NeuralNet_XLTest: Compare with Linear Regression - first column of y")

    val y0  = y.col(0)                                                  // use first column of matrix y
    val rg0 = new Regression (x, y0)                                    // create a Regression model
    println (rg0.analyze ().report)

    val y0p = rg0.predict (x)                                           // predicted output value
    println ("target output:    y0  = " + y0)
    println ("predicted output: y0p = " + y0p)

    banner ("NeuralNet_XLTest: Compare with Linear Regression - second column of y")

    val y1 = y.col(1)                                                   // use second column of matrix y
    val rg1 = new Regression (x, y1)                                    // create a Regression model
    println (rg1.analyze ().report)

    val y1p = rg1.predict (x)                                           // predicted output value
    println ("target output:    y1  = " + y1)
    println ("predicted output: y1p = " + y1p)

} // NeuralNet_XLTest object


//::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::
/** The `NeuralNet_XLTest2` object trains a neural netowrk on the `ExampleBasketBall` dataset.
 *  > runMain scalation.analytics.NeuralNet_XLTest2
 */
object NeuralNet_XLTest2 extends App
{
    import ExampleBasketBall._
    banner ("NeuralNet_XL vs. Regession - ExampleBasketBall")

    println ("ox = " + ox)
    println ("y  = " + y)

    banner ("Regression")
    val rg = Regression (oxy)
    println (rg.analyze ().report)

    banner ("prediction")                                               // not currently rescaling
    val yq = rg.predict ()                                              // scaled predicted output values for all x
    println ("target output:    y  = " + y)
    println ("predicted output: yq = " + yq)
    println ("error:            e  = " + (y - yq))

    banner ("NeuralNet_XL with scaled y values")
//  hp("eta") = 0.016                                                   // try several values - train0
    hp("eta") = 0.1                                                     // try several values - train

    val nn = NeuralNet_XL (xy)                                          // factory function automatically rescales
//  val nn = new NeuralNet_XL (x, MatrixD (Seq (y)))                    // constructor does not automatically rescale

    nn.trainSwitch (2).eval ()                                          // fit the weights using training data
    println (nn.report)

    banner ("scaled prediction")
    val yp = nn.predictV ().col (0)                                     // scaled predicted output values for all x
    println ("target output:    y  = " + y)
    println ("predicted output: yp = " + yp)
    println ("error:            e  = " + (y - yp))

/*
    banner ("unscaled prediction")
//  val (ymu, ysig) = (y.mean, sqrt (y.variance))                       // should obtain from apply - see below
//  val ypu = denormalizeV ((ymu, ysig))(yp)                            // denormalize predicted output values for all x
    val ypu = nn.itran (yp)                                             // denormalize predicted output values for all x
    println ("target output:   y   = " + y)
    println ("unscaled output: ypu = " + ypu)
*/

} // NeuralNet_XLTest2 object


//::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::
/** The `NeuralNet_XLTest3` object trains a neural netowrk on the `ExampleAutoMPG` dataset.
 *  > runMain scalation.analytics.NeuralNet_XLTest3
 */
object NeuralNet_XLTest3 extends App
{
    import ExampleAutoMPG._
    banner ("NeuralNet_XL vs. Regession - ExampleAutoMPG")

    banner ("Regression")
    val rg = Regression (oxy)
    println (rg.analyze ().report)

    banner ("prediction")                                               // not currently rescaling
    val yq = rg.predict ()                                              // scaled predicted output values for all x
    println ("target output:    y  = " + y)
    println ("predicted output: yq = " + yq)
    println ("error:            e  = " + (y - yq))

    banner ("NeuralNet_XL with scaled y values")
//  hp("eta") = 0.0014                                                  // try several values - train0
    hp("eta") = 0.01                                                    // try several values - train

    val nn = NeuralNet_XL (xy)                                          // factory function automatically rescales
//  val nn = new NeuralNet_XL (x, MatrixD (Seq (y)))                    // constructor does not automatically rescale

    nn.trainSwitch (2).eval ()                                          // fit the weights using training data (0, 1, 2)
    println (nn.report)

/*
    banner ("scaled prediction")
    val yp = nn.predict ().col (0)                                      // scaled predicted output values for all x
    println ("target output:    y  = " + y)
    println ("predicted output: yp = " + yp)
    println ("error:            e  = " + (y - yp))

    banner ("unscaled prediction")
//  val (ymu, ysig) = (y.mean, sqrt (y.variance))                       // should obtain from apply - see below
//  val ypu = denormalizeV ((ymu, ysig))(yp)                            // denormalize predicted output values for all x
    val ypu = nn.itran (yp)                                             // denormalize predicted output values for all x
    println ("target output:   y   = " + y)
    println ("unscaled output: ypu = " + ypu)
*/

} // NeuralNet_XLTest3 object


//::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::
/** The `NeuralNet_XLTest4` object trains a neural netowrk on the `ExampleAutoMPG` dataset.
 *  It test cross-validation.
 *  > runMain scalation.analytics.NeuralNet_XLTest4
 */
object NeuralNet_XLTest4 extends App
{
    import ExampleAutoMPG._
    banner ("NeuralNet_XL cross-validation - ExampleAutoMPG")

    banner ("NeuralNet_XL with scaled y values")
//  hp("eta") = 0.0014                                              // try several values - train0
    hp("eta") = 0.02                                                // try several values - train

    val nn = NeuralNet_XL (xy)                                      // factory function automatically rescales
//  val nn = new NeuralNet_XL (x, MatrixD (Seq (y)))                // constructor does not automatically rescale

    nn.trainSwitch (1).eval ()                                      // fit the weights using training data (0, 1, 2)
    println (nn.report)

    banner ("cross-validation")
    nn.crossValidate ()

} // NeuralNet_XLTest4 object


//::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::
/** The `NeuralNet_XLTest5` object trains a neural network on the `ExampleAutoMPG` dataset.
 *  This tests forward feature/variable selection.
 *  > runMain scalation.analytics.NeuralNet_XLTest5
 */
object NeuralNet_XLTest5 extends App
{
    import ExampleAutoMPG._
    banner ("NeuralNet_XL feature selection - ExampleAutoMPG")

    banner ("NeuralNet_XL with scaled y values")
    hp("eta") = 0.02
    val nn = NeuralNet_XL (xy, Array (5, 6))                        // factory function automatically rescales
//  val nn = new NeuralNet_XL (x, y)                                // constructor does not automatically rescale
    val ft = nn.fitA(0)

    nn.train ().eval ()                                             // fit the weights using training data
    val n = x.dim2                                                  // number of parameters/variables
    println (nn.report)

    banner ("Forward Selection Test")
    nn.forwardSelAll ()
} // NeuralNet_XLTest5 object


//::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::
/** The `NeuralNet_XLTest6` object trains a neural network on the `ExampleAutoMPG` dataset.
 *  This tests forward feature/variable selection with plotting of R^2,
 *  > runMain scalation.analytics.NN4L_forSel_ConcreteTest
 */
object NN4L_forSel_ConcreteTest extends App
{
    import ExampleConcrete._
    // import ExampleAutoMPG._

    val n = ox.dim2                                                // number of parameters/variables
    banner ("NeuralNet_XL feature selection - ExampleAutoMPG")

    val af_ = Array (f_sigmoid, f_sigmoid, f_id)                   // try different activation functions
//  val af_ = Array (f_tanh, f_tanh, f_id)                         // try different activation functions

    banner ("NeuralNet_XL with scaled y values")
    hp("eta") = 0.02                                               // learning rate hyoer-parameter (see Optimizer)
    val nn  = NeuralNet_XL (oxy, Array (5,7), af = af_)            // factory function automatically rescales
//  val nn  = new NeuralNet_XL (ox, y, af = af_)                   // constructor does not automatically rescale

    nn.train ().eval ()                                            // fit the weights using training data
    println (nn.report)                                            // report parameters and fit
    val ft  = nn.fitA(0)                                           // fit for first output variable

    banner ("Forward Selection Test")
    val (cols, rSq) = nn.forwardSelAll ()                          // R^2, R^2 bar, R^2 cv
    println (s"rSq = $rSq")
    val k = cols.size
    println (s"k = $k, n = $n")
    val t = VectorD.range (1, k)                                   // instance index
    new PlotM (t, rSq.t, Array ("R^2", "R^2 bar", "R^2 cv", "AIC"),
               "R^2 vs n for NeuralNet_4L with forwardSel", lines = true)

} // NN4L_forSel_ConcreteTest object

object NN4L_backElim_ConcreteTest extends App
{
    import ExampleConcrete._
    // import ExampleAutoMPG._

    val n = ox.dim2                                                // number of parameters/variables
    banner ("NeuralNet_XL feature selection - ExampleAutoMPG")

    val af_ = Array (f_sigmoid, f_sigmoid, f_id)                   // try different activation functions
    //  val af_ = Array (f_tanh, f_tanh, f_id)                     // try different activation functions

    banner ("NeuralNet_XL with scaled y values")
    hp("eta") = 0.02                                               // learning rate hyoer-parameter (see Optimizer)
    val nn  = NeuralNet_XL (oxy, Array (5,7), af = af_)            // factory function automatically rescales
    //  val nn  = new NeuralNet_XL (ox, y, af = af_)               // constructor does not automatically rescale

    nn.train ().eval ()                                            // fit the weights using training data
    println (nn.report)                                            // report parameters and fit
    val ft  = nn.fitA(0)                                           // fit for first output variable

    banner ("Backward Elimination Test")
    val (cols, rSq) = nn.backwardElimAll ()                        // R^2, R^2 bar, R^2 cv
    println (s"rSq = $rSq")
    val k = rSq.dim1
    println (s"k = $k, n = $n")
    val t = VectorD.range (1, k)                                   // instance index
    new PlotM (t, rSq.t, Array ("R^2", "R^2 bar", "R^2 cv", "AIC"),
        "R^2 vs n for NeuralNet_4L with backwardElim", lines = true)

} // NN4L_backElim_ConcreteTest object

object NN4L_stepReg_ConcreteTest extends App
{
    import ExampleConcrete._
    // import ExampleAutoMPG._

    val n = ox.dim2                                                // number of parameters/variables
    banner ("NeuralNet_XL feature selection - ExampleAutoMPG")

    val af_ = Array (f_sigmoid, f_sigmoid, f_id)                   // try different activation functions
    //  val af_ = Array (f_tanh, f_tanh, f_id)                     // try different activation functions

    banner ("NeuralNet_XL with scaled y values")
    hp("eta") = 0.02                                               // learning rate hyoer-parameter (see Optimizer)
    val nn  = NeuralNet_XL (oxy, af = af_)                         // factory function automatically rescales
    //  val nn  = new NeuralNet_XL (ox, y, af = af_)               // constructor does not automatically rescale

    nn.train ().eval ()                                            // fit the weights using training data
    println (nn.report)                                            // report parameters and fit
    val ft  = nn.fitA(0)                                           // fit for first output variable

    banner ("Stepwise Regression Test")
    val (cols, rSq) = nn.stepRegressionAll ()                      // R^2, R^2 bar, R^2 cv
    println (s"rSq = $rSq")
    val k = cols.size
    println (s"k = $k, n = $n")
    val t = VectorD.range (1, k)                                   // instance index
    new PlotM (t, rSq.t, Array ("R^2", "R^2 bar", "R^2 cv", "AIC"),
        "R^2 vs n for NeuralNet_4L with stepReg", lines = true)

} // NN4L_stepReg_ConcreteTest object


//::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::
/** The `NN4L_forSel_AutoMPGTest` object trains a neural network on the `ExampleAutoMPG` dataset.
 *  This tests forward feature/variable selection with plotting of R^2,
 *  > runMain scalation.analytics.NN4L_forSel_AutoMPGTest
 */
object NN4L_forSel_AutoMPGTest extends App
{
    import ExampleAutoMPG._

    val n = ox.dim2                                                // number of parameters/variables
    banner ("NeuralNet_XL feature selection - ExampleAutoMPG")

    val af_ = Array (f_sigmoid, f_sigmoid, f_id)                   // try different activation functions
    //  val af_ = Array (f_tanh, f_tanh, f_id)                         // try different activation functions

    banner ("NeuralNet_XL with scaled y values")
    hp("eta") = 0.02                                               // learning rate hyoer-parameter (see Optimizer)
    val nn  = NeuralNet_XL (oxy, Array (5,7), af = af_)            // factory function automatically rescales
    //  val nn  = new NeuralNet_XL (ox, y, af = af_)                   // constructor does not automatically rescale

    nn.train ().eval ()                                            // fit the weights using training data
    println (nn.report)                                            // report parameters and fit
    val ft  = nn.fitA(0)                                           // fit for first output variable

    banner ("Forward Selection Test")
    val (cols, rSq) = nn.forwardSelAll ()                          // R^2, R^2 bar, R^2 cv
    println (s"rSq = $rSq")
    val k = cols.size
    println (s"k = $k, n = $n")
    val t = VectorD.range (1, k)                                   // instance index
    new PlotM (t, rSq.t, Array ("R^2", "R^2 bar", "R^2 cv", "AIC"),
        "R^2 vs n for NeuralNet_4L with forwardSel", lines = true).saveImage("NeuralNet_4L with forward selection for AutoMPG")

} // NN4L_forSel_AutoMPGTest object

//::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::
/** The `NN4L_backElim_AutoMPGTest` object trains a neural network on the `ExampleAutoMPG` dataset.
 *  This tests backward elimination feature/variable selection with plotting of R^2,
 *  > runMain scalation.analytics.NN4L_backElim_AutoMPGTest
 */
object NN4L_backElim_AutoMPGTest extends App
{
    import ExampleAutoMPG._

    val n = ox.dim2                                                // number of parameters/variables
    banner ("NeuralNet_XL feature selection - ExampleAutoMPG")

    val af_ = Array (f_sigmoid, f_sigmoid, f_id)                   // try different activation functions
    //  val af_ = Array (f_tanh, f_tanh, f_id)                     // try different activation functions

    banner ("NeuralNet_XL with scaled y values")
    hp("eta") = 0.01                                               // learning rate hyoer-parameter (see Optimizer)
    val nn  = NeuralNet_XL (oxy, Array (4, 6), af = af_)            // factory function automatically rescales
    //  val nn  = new NeuralNet_XL (ox, y, af = af_)               // constructor does not automatically rescale

    nn.train ().eval ()                                            // fit the weights using training data
    println (nn.report)                                            // report parameters and fit
    val ft  = nn.fitA(0)                                           // fit for first output variable

    banner ("Backward Elimination Test")
    val (cols, rSq) = nn.backwardElimAll ()                        // R^2, R^2 bar, R^2 cv
    println (s"rSq = $rSq")
    val k = rSq.dim1
    println (s"k = $k, n = $n")
    val t = VectorD.range (1, k)                                   // instance index
    new PlotM (t, rSq.t, Array ("R^2", "R^2 bar", "R^2 cv", "AIC"),
        "R^2 vs n for NeuralNet_4L with backwardElim", lines = true).saveImage("NeuralNet_4L with backwardElim for AutoMPG")

} // NN4L_backElim_AutoMPGTest object

//::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::
/** The `NN4L_stepReg_AutoMPGTest` object trains a neural network on the `ExampleAutoMPG` dataset.
 *  This tests stepwise Regression feature/variable selection with plotting of R^2,
 *  > runMain scalation.analytics.NN4L_stepReg_AutoMPGTest
 */
object NN4L_stepReg_AutoMPGTest extends App
{
    import ExampleAutoMPG._

    val n = ox.dim2                                                // number of parameters/variables
    banner ("NeuralNet_XL feature selection - ExampleAutoMPG")

    val af_ = Array (f_sigmoid, f_sigmoid, f_id)                   // try different activation functions
    //  val af_ = Array (f_tanh, f_tanh, f_id)                     // try different activation functions

    banner ("NeuralNet_XL with scaled y values")
    hp("eta") = 0.01                                               // learning rate hyoer-parameter (see Optimizer)
    val nn  = NeuralNet_XL (oxy, af = af_)                         // factory function automatically rescales
    //  val nn  = new NeuralNet_XL (ox, y, af = af_)               // constructor does not automatically rescale

    nn.train ().eval ()                                            // fit the weights using training data
    println (nn.report)                                            // report parameters and fit
    val ft  = nn.fitA(0)                                           // fit for first output variable

    banner ("Stepwise Regression Test")
    val (cols, rSq) = nn.stepRegressionAll ()                      // R^2, R^2 bar, R^2 cv
    println (s"rSq = $rSq")
    val k = cols.size
    println (s"k = $k, n = $n")
    val t = VectorD.range (1, k)                                   // instance index
    new PlotM (t, rSq.t, Array ("R^2", "R^2 bar", "R^2 cv", "AIC"),
        "R^2 vs n for NeuralNet_4L with stepReg", lines = true).saveImage("NeuralNet_4L with stepwise regression for AutoMPG")

} // NN4L_stepReg_AutoMPGTest object
