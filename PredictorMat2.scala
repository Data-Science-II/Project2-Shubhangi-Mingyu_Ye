
//::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::
/** @author  John Miller
 *  @version 1.6
 *  @date    Fri Mar 16 15:13:38 EDT 2018
 *  @see     LICENSE (MIT style license file).
 *
 *  @title   Model Framework: Predictor for Matrix Input and Matrix Output
 *
 *  @see     hebb.mit.edu/courses/9.641/2002/lectures/lecture03.pdf
 */

package scalation.analytics

import scala.collection.mutable.{ArrayBuffer, Map, Set}
import scala.util.control.Breaks.{break, breakable}

import scalation.linalgebra.{FunctionV_2V, MatriD, MatrixD, VectoD, VectorD, VectorI}
import scalation.math.{double_exp, noDouble}
import scalation.stat.Statistic
import scalation.random.PermutedVecI
import scalation.util.banner

import Fit._
import MatrixTransform._

//::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::
/** The `PredictorMat2` abstract class provides the basic structure and API for
 *  a variety of modeling techniques with multiple outputs/responses, e.g., Neural Networks.
 *  @param x       the m-by-nx data/input matrix (data consisting of m input vectors)
 *  @param y       the m-by-ny response/output matrix (data consisting of m output vectors)
 *  @param fname   the feature/variable names (if null, use x_j's)
 *  @param hparam  the hyper-parameters for the model/network
 */
abstract class PredictorMat2 (x: MatriD, y: MatriD,
                              protected var fname: Strings, hparam: HyperParameter)
         extends Predictor
{
    if (x.dim1 != y.dim1) flaw ("constructor", "row dimensions of x and y are incompatible")

    private   val DEBUG   = true                                         // debug flag
    private   val DEBUG2  = false                                        // verbose debug flag
    protected val m       = x.dim1                                       // number of data points (input vectors)
    protected val nx      = x.dim2                                       // dimensionality of the input
    protected val ny      = y.dim2                                       // dimensionality of the output
    protected val _1      = VectorD.one (m)                              // vector of all ones
    protected var eta     = if (hparam == null) 0.0 else hparam ("eta")  // the learning/convergence rate (adjustable)

    private   val stream  = 0                                            // random number stream to use
    private   val permGen = PermutedVecI (VectorI.range (0, m), stream)  // permutation generator

    protected var ee: MatriD = null                                      // residual/error matrix

    if (fname == null) fname = x.range2.map ("x" + _).toArray            // default feature/variable names

    val fitA = Array.ofDim [Fit] (ny)
    for (k <- fitA.indices) fitA(k) = new Fit (y.col(k), nx, (nx-1, m - nx))

    //::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::
    /** Return the data matrix 'x'.  Mainly for derived classes where 'x' is expanded
     *  from the given columns in 'x_', e.g., `QuadRegression` add squared columns.
     */
    def getX: MatriD = x

    //::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::
    /** Return the first response vector 'y.col(0)'.  Mainly for derived classes where 'y'
     *  is transformed.
     */
    def getY: VectoD = y.col(0)

    //::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::
    /** Return the response matrix 'y'.  Mainly for derived classes where 'y' is
     *  transformed.
     */
    def getYY: MatriD = y

    //::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::
    /** Reset the learning rate 'eta'.  Since this hyper-parameter needs frequent tuning,
     *  this method is provided to facilitate that.
     *  @param eta_  the learning rate
     */
    def reset (eta_ : Double) { eta = eta_ }

    //::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::
    /** Given data matrix 'x_r' and response matrix 'y_r', fit the parameters 'b'
     *  (weights and biases) using a simple, easy to follow algorithm.
     *  @param x_r  the training/full data/input matrix
     *  @param y_r  the training/full response/output matrix
     */
    def train0 (x_r: MatriD = x, y_r: MatriD = y): PredictorMat2

    //::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::
    /** Given data/input matrix 'x_r' and response matrix 'y_r', fit the parameters 'b'
     *  (weights and  biases).
     *  @param x_r  the training/full data/input matrix
     *  @param y_r  the training/full response/output matrix
     */
    def train (x_r: MatriD = x, y_r: MatriD = y): PredictorMat2

    //::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::
    /** Given data matrix 'x_r' and response vector 'y_r', fit the parameter 'b'
     *  (weights and biases).
     *  @param x_r  the training/full data/input matrix
     *  @param y_r  the training/full response/output vector, e.g., for the first variable/column
     */
    def train (x_r: MatriD, y_r: VectoD): PredictorMat2 = train (x_r, MatrixD (Seq (y_r)))

    //::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::
    /** Given data matrix 'x_r' and response matrix 'y_r', fit the parameters 'b'
     *  (weights and biases).  Overriding implementations (if needed) of this method
     *  should optimize hyper-parameters (e.g., the learning rate 'eta').
     *  @param x_r  the training/full data/input matrix
     *  @param y_r  the training/full response/output matrix
     */
    def train2 (x_r: MatriD = x, y_r: MatriD = y): PredictorMat2 = train (x_r, y_r)

    //::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::
    /** Switch between 'train' methods: simple (0), regular (1) and hyper-parameter
     *  optimizing (2).
     *  @param which  the kind of 'train' method to use
     *  @param x_r   the training/full data/input matrix
     *  @param y_r   the training/full response/output matrix
     */
    def trainSwitch (which: Int, x_r: MatriD = x, y_r: MatriD = y): PredictorMat2 =
    {
        which match {
        case 0 => train0 (x_r, y_r)
        case 1 => train (x_r, y_r)
        case 2 => train2 (x_r, y_r)
        case _ => flaw ("trainSwitch", s"which = $which not in (0, 1, 2)"); null
        } // match
    } // trainSwitch

    //::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::
    /** Reset the degrees of freedom to the new updated values.  For some models,
     *  the degrees of freedom is not known until after the model is built.
     *  Caveat:  only applies to the first response/output variable.
     *  @param df_update  the updated degrees of freedom (model, error)
     */
    def resetDF (df_update: PairD)
    {
        fitA(0).resetDF (df_update)
        if (DEBUG) println (s"resetDF: df = $df_update")
    } // resetDF

    //::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::
    /** Evaluate the quality of the fit for the parameter/weight matrices on the
     *  entire dataset or the test dataset.  Only considers the first response/output
     *  variable/column.
     *  @param x_e  the test/full data/input matrix
     *  @param y_e  the test/full response/output vector (first column only)
     */
    def eval (x_e: MatriD = x, y_e: VectoD = y.col(0)): PredictorMat2 =
    {
        val yp0 = predictV (x_e).col(0)                                 // predict output/responses, first column
        val e   = y_e - yp0                                             // error vector, first column
        fitA(0).diagnose (e, y_e, yp0)                                  // compute diagonostics, first column
        this
    } // eval

    //::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::
    /** Evaluate the quality of the fit for the parameter/weight matrices on the
     *  entire dataset or the test dataset.  Considers all the response/output
     *  variables/columns.
     *  @param x_e  the test/full data/input data matrix
     *  @param y_e  the test/full response/output response matrix
     */
    def eval (x_e: MatriD, y_e: MatriD): PredictorMat2 =
    {
        val yp = predictV (x_e)                                         // predict output/responses
        val e  = y_e - yp                                               // error matrix
        for (k <- e.range2) fitA(k).diagnose (e.col(k), y_e.col(k), yp.col(k))   // compute diagonostics, per column
        this
    } // eval

    //::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::
    /** Compute the error (difference between actual and predicted) and useful
     *  diagnostics for the test dataset.  Requires predicted responses to be
     *  passed in.
     *  @param ym   the training/full mean actual response/output vector
     *  @param y_e  the test/full actual response/output vector
     *  @param yp   the test/full predicted response/output vector
     */
    def eval (ym: Double, y_e: VectoD, yp: VectoD): PredictorMat2 = ???

    //:::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::
    /** Analyze a dataset using this model using ordinary training with the
     *  'train' method.
     *  Only uses the first output variable's value.
     *  @param x_r  the data/input matrix (training/full)
     *  @param y_r  the response/output vector (training/full)
     *  @param x_e  the data/input matrix (testing/full)
     *  @param y_e  the response/output vector (testing/full)
     */
    def analyze (x_r: MatriD = x, y_r: VectoD = y(0),
                 x_e: MatriD = x, y_e: VectoD = y(0)): PredictorMat2 =
    {
        train (x_r, y_r)                                                // train the model on the training dataset
        val ym = y_r.mean                                               // compute mean of training response - FIX use ym
        eval (x_e, y_e)                                                 // evaluate using the testing dataset
        this
    } // analyze

    //::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::
    /** Return the vector of residuals/errors for first response/output variable/column.
     */
    def residual: VectoD = ee.col(0)

    //::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::
    /** Return the matrix of residuals/errors.
     */
    def residuals: MatriD = ee

    //::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::
    /** Return the labels for the quality of fit measures.
     */
    def fitLabel: Seq [String] = fitA(0).fitLabel

    //::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::
    /** Return 'fitMap' results for each y-column and print the overall 'rSq' average
     *  over all y-columns.
     */
    def fitMap: IndexedSeq [Map [String, String]] =
    {
        val fits = Array.ofDim [Map [String, String]] (fitA.length)
        var sst, sse = 0.0
        for (k <- fitA.indices) {
            fits(k) = fitA(k).fitMap
            sst += fits(k)("sst").toDouble
            sse += fits(k)("sse").toDouble
        } // for
        val rSq = (sst - sse) / sst
        println (s"fitMap: overall: rSq = $rSq, sst = $sst, sse = $sse")
        fits
    } // fitMap

    //::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::
    /** Return the hyper-parameters.
     */
    def hparameter: HyperParameter = hparam

    //::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::
    /** Return the parameter/weight vector (first layer, first output).
     */
    def parameter: VectoD = parameters (0).w(0)

    //::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::
    /** Return the all parameters (weights and biases).
     */
    def parameters: NetParams

    //::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::
    /** Return a basic report on the trained model.
     *  @see 'summary' method for more details
     */
    def report: String =
    {
        s"""
REPORT
    hparameter hp  = $hparameter
    parameters b   = $parameters
    fitMap     qof = ${fitMap.map ("\n" + _)}
        """
    } // report

    //::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::
    /** Build a sub-model that is restricted to the given columns of the data matrix.
     *  @param x_cols  the columns that the new model is restricted to
     */
    def buildModel (x_cols: MatriD): PredictorMat2

    //::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::
    /** Perform forward selection to find the most predictive variable to add the
     *  existing model, returning the variable to add and the new model.
     *  May be called repeatedly.
     *  @see `Fit` for index of QoF measures.
     *  @param cols     the columns of matrix x currently included in the existing model
     *  @param index_q  index of Quality of Fit (QoF) to use for comparing quality
     */
    def forwardSel (cols: Set [Int], index_q: Int = index_rSqBar): (Int, PredictorMat2) =
    {
        var j_mx   = -1                                                  // best column, so far
        var mod_mx = null.asInstanceOf [PredictorMat2]                   // best model, so far
        var fit_mx = noDouble                                            // best fit, so far

        for (j <- x.range2 if ! (cols contains j)) {
            val cols_j = cols + j                                        // try adding variable/column x_j
            val x_cols = x.selectCols (cols_j.toArray)                   // x projected onto cols_j columns
            val mod_j  = buildModel (x_cols)                             // regress with x_j added
            mod_j.train ().eval ()                                       // train model, evaluate QoF
            val fit_j = mod_j.fitA(0).fit(index_q)                       // new fit for first response
            if (fit_j > fit_mx) { j_mx = j; mod_mx = mod_j; fit_mx = fit_j }
        } // for
        if (j_mx == -1) {
            flaw ("forwardSel", "could not find a variable x_j to add: j = -1")
        } // if
        (j_mx, mod_mx)                                                    // return best column and model
    } // forwardSel

    //::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::
    /** Perform forward selection to find the most predictive variables to have
     *  in the model, returning the variables added and the new Quality of Fit (QoF)
     *  measures for all steps.
     *  @see `Fit` for index of QoF measures.
     *  @param index_q  index of Quality of Fit (QoF) to use for comparing quality
     *  @param cross    whether to include the cross-validation QoF measure
     */
    def forwardSelAll (index_q: Int = index_rSqBar, cross: Boolean = true): (Set [Int], MatriD) =
    {
        val rSq  = new MatrixD (x.dim2 - 1, 3)                           // R^2, R^2 Bar, R^2 cv
        val cols = Set (0)                                               // start with x_0 in model

        breakable { for (l <- 0 until x.dim2 - 1) {
            val (j, mod_j) = forwardSel (cols)                           // add most predictive variable
            if (j == -1) break
            cols     += j                                                // add variable x_j
            val fit_j = mod_j.fitA(0).fit
            rSq(l)    = if (cross) Fit.qofVector (fit_j, mod_j.crossValidate ())   // use new model, mod_j, with cross
                        else       Fit.qofVector (fit_j, null)                     // use new model, mod_j, no cross
            if (DEBUG) {
                val k = cols.size + 1
                println (s"forwardSel: add (#$k) variable $j, qof = ${fit_j(index_q)}")
            } // if
        }} // breakable for

        (cols, rSq.slice (0, cols.size-1))
    } // forwardSelAll

    //::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::
    /** Perform backward elimination to find the least predictive variable to remove
     *  from the existing model, returning the variable to eliminate, the new parameter
     *  vector and the new Quality of Fit (QoF).  May be called repeatedly.
     *  @see `Fit` for index of QoF measures.
     *  @param cols     the columns of matrix x currently included in the existing model
     *  @param index_q  index of Quality of Fit (QoF) to use for comparing quality
     *  @param first    first variable to consider for elimination
     *                        (default (1) assume intercept x_0 will be in any model)
     */
    def backwardElim (cols: Set [Int], index_q: Int = index_rSqBar, first: Int = 1): (Int, PredictorMat2) =
    {
        var j_mx   = -1                                                  // best column, so far
        var mod_mx = null.asInstanceOf [PredictorMat2]                   // best model, so far
        var fit_mx = noDouble                                            // best fit, so far

        for (j <- first until x.dim2 if cols contains j) {
            val cols_j = cols - j                                        // try removing variable/column x_j
            val x_cols = x.selectCols (cols_j.toArray)                   // x projected onto cols_j columns
            val mod_j  = buildModel (x_cols)                             // regress with x_j added
            mod_j.train ().eval ()                                       // train model, evaluate QoF
            val fit_j = mod_j.fitA(0).fit(index_q)                       // new fit for first response
            if (fit_j > fit_mx) { j_mx = j; mod_mx = mod_j; fit_mx = fit_j }
        } // for
        if (j_mx == -1) {
            flaw ("backwardElim", "could not find a variable x_j to eliminate: j = -1")
        } // if
        (j_mx, mod_mx)                                                   // return best column and model
    } // backwardElim

    //::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::
    /** Perform forward selection to find the most predictive variables to have
     *  in the model, returning the variables added and the new Quality of Fit (QoF)
     *  measures for all steps.
     *  @see `Fit` for index of QoF measures.
     *  @param index_q  index of Quality of Fit (QoF) to use for comparing quality
     *  @param first    first variable to consider for elimination
     *  @param cross    whether to include the cross-validation QoF measure
     */
    def backwardElimAll (index_q: Int = index_rSqBar, first: Int = 1, cross: Boolean = true): (Set [Int], MatriD) =
    {   
        val rSq  = new MatrixD (x.dim2 - 1, 3)                           // R^2, R^2 Bar, R^2 cv
        val cols = Set (Array.range (0, x.dim2) :_*)                     // start with all x_j in model
        
        breakable { for (l <- 1 until x.dim2 - 1) {
            val (j, mod_j) = backwardElim (cols, first)                  // remove most predictive variable
            if (j == -1) break                                           
            cols     -= j                                                // remove variable x_j
            val fit_j = mod_j.fitA(0).fit
            rSq(l)    = if (cross) Fit.qofVector (fit_j, mod_j.crossValidate ())   // use new model, mod_j, with cross
                        else       Fit.qofVector (fit_j, null)                     // use new model, mod_j, no cross
            if (DEBUG) {
                println (s"<== backwardElimAll: remove (#$l) variable $j, qof = ${fit_j(index_q)}")
            } // if
        }} // breakable for
        
//        (cols, rSq.slice (0, cols.size-1))
        (cols, reverse (rSq.slice (1, rSq.dim1)))

    } // backwardElimAll
    /** Return a matrix that is in reverse row order of the given matrix 'a'.

     *  @param a  the given matrix

     */

    def reverse (a: MatriD): MatriD =

    {

        val b = new MatrixD (a.dim1, a.dim2)

        for (i <- a.range1) b(i) = a(a.dim1 - 1 - i)

        b

    } // reverse

    //::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::
    /** Compute the Variance Inflation Factor 'VIF' for each variable to test
     *  for multi-collinearity by regressing 'x_j' against the rest of the variables.
     *  A VIF over 10 indicates that over 90% of the variance of 'x_j' can be predicted
     *  from the other variables, so 'x_j' may be a candidate for removal from the model.
     *  Note:  override this method to use a superior regression technique.
     *  @param skip  the number of columns of x at the beginning to skip in computing VIF
     */
    def vif (skip: Int = 1): VectoD =
    {
        val vifV = new VectorD (x.dim2 - skip)                         // VIF vector for x columns except skip columns
        for (j <- skip until x.dim2) {
            val (x_noj, x_j) = pullResponse (x, j)                     // x matrix without column j, only column j
            val rg_j  = new Regression (x_noj, x_j)                    // regress with x_j removed
            rg_j.train ().eval ()                                      // train model, evaluate QoF
            val rSq_j = rg_j.fit(index_rSq)                            // R^2 for predicting x_j
            if (DEBUG2) println (s"vif: for variable x_$j, rSq_$j = $rSq_j")
            vifV(j-1) =  1.0 / (1.0 - rSq_j)                           // store vif for x_1 in vifV(0)
        } // for
        vifV
    } // vif
//    def stepwise (cols: Set [Int], index_q: Int = index_rSqBar): (Int, PredictorMat2) =
//    {
//        var j_mx   = -1                                                  // best column, so far
//        var mod_mx = null.asInstanceOf [PredictorMat2]                   // best model, so far
//        var fit_mx = noDouble                                            // best fit, so far
//
//
//        (f_col, f_reg) = forward_sel(reg, X, y, cols)
//        (b_col, b_reg) = backward_elim(reg, X, y, cols)
//
//    }
    def stepwiseSel (cols: Set [Int], index_q: Int = index_rSqBar): (Int, PredictorMat2) =
    {
        var j_mx   = -1                                                  // best column, so far
        var mod_mx = null.asInstanceOf [PredictorMat2]                    // best model, so far
        var fit_mx = noDouble                                            // best fit, so far

        for (j <- x.range2 if ! (cols contains j)) {
            val cols_j = cols + j                                        // try adding variable/column x_j
            val x_cols = x.selectCols (cols_j.toArray)                   // x projected onto cols_j columns
            val mod_j   = buildModel (x_cols)                            // regress with x_j added
            mod_j.train ().eval ()                                       // train model, evaluate QoF
            val fit_j = mod_j.fitA(0).fit(index_q)                              // new fit
            if (fit_j > fit_mx) {
                j_mx = j
                mod_mx = mod_j;
                fit_mx = fit_j
                for (m <- 0 until x.dim2 if cols contains m) {
                    val cols_m = cols - m                                        // try removing variable/column x_j
                    val m_cols = x.selectCols (cols_m.toArray)                   // x projected onto cols_j columns
                    val mod_m  = buildModel (m_cols)                             // regress with x_j added
                    mod_m.train ().eval ()                                       // train model, evaluate QoF
                    val fit_m = mod_m.fitA(0).fit(index_q)                       // new fit
                    if (fit_m > fit_mx) { j_mx = m; mod_mx = mod_m; fit_mx = fit_m }
                }}
        } // for
        (j_mx, mod_mx)                                                    // return best column and model
    } // stepwiseSel
    def stepwiseSelAll (index_q: Int = index_rSqBar, cross: Boolean = true): (Set [Int], MatriD) =
    {
        val rSq  = new MatrixD (x.dim2, 3)                           // R^2, R^2 Bar, R^2 cv
        val cols = Set (0)                                               // start with x_0 in model
        val (first, mod_first) = forwardSel (cols)
        cols     += first
        val fit_j = mod_first.fitA(0).fit
        rSq(first)    = if (cross) Fit.qofVector (fit_j, mod_first.crossValidate ())   // use new model, mod_j, with cross
        else        Fit.qofVector (fit_j, null)                      // use new model, mod_j, no cross
        println (s"rSq = $rSq")
        breakable { for (l <- 1 until x.dim2 - 1) {
            println (s"l = $l")
            val (j, mod_j) = stepwiseSel (cols)                           // add most predictive variable
            println (s"j = $j")
            println (s"mod_j.crossValidate () = $mod_j.crossValidate ()")
            if (j == -1) break
            cols     += j                                                // add variable x_j
            val fit_j = mod_j.fitA(0).fit
            rSq(l)    = if (cross) Fit.qofVector (fit_j, mod_j.crossValidate ())   // use new model, mod_j, with cross
            else       Fit.qofVector (fit_j, null)                     // use new model, mod_j, no cross
            println (s"rSq = $rSq")
            //            if (DEBUG) {
            //                val k = cols.size - 1
            //                println (s"==> stepwiseSel: add (#$k) variable $j, qof = ${fit_j(index_q)}")
            //            } // if
        }} // breakable for

        (cols, rSq.slice (0, cols.size-1))
    } // stepwiseSelAll


    //::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::
    /** Given a new input vector 'z', predict the output/response value 'f(z)'.
     *  Return only the first output variable's value.
     *  @param z  the new input vector
     */
    def predict (z: VectoD): Double = predictV (z)(0)

    //::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::
    /** Given a new input matrix 'z', predict the output/response matrix 'f(z)'.
     *  Return only the first output variable's value.
     *  @param z  the new input matrix
     */
    def predict (z: MatriD = x): VectoD = predictV (z).col(0)

    //::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::
    /** Given a new input vector 'z', predict the output/response vector 'f(z)'.
     *  @param z  the new input vector
     */
    def predictV (z: VectoD): VectoD

    //::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::
    /** Given a new input matrix 'z', predict the output/response matrix 'f(z)'.
     *  @param z  the new input matrix
     */
    def predictV (z: MatriD = x): MatriD

    //:::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::
    /*  Perform 'k'-fold cross-validation to compute test quality of fit measures
     *  by dividing the dataset into test datasets and training datasets.
     *  Each test dataset is defined by 'idx' and the rest of the data is the training dataset.
     *  @param k      the number of folds/crosses (defaults to 10x).
     *  @param rando  whether to use randomized cross-validation (defaults to true)
     */
    def crossValidate (k: Int = 10, rando: Boolean = true): Array [Statistic] =
    {
        if (k < MIN_FOLDS) flaw ("crossValidate", s"k = $k must be at least $MIN_FOLDS")
        val fLabel = fitA(0).fitLabel                                    // labels for qof measures
        val stats  = Array.ofDim [Statistic] (ny * fitLabel.length)
        for (i <- stats.indices) stats(i) = new Statistic (fLabel(i % fLabel.size))
        val indices = if (rando) permGen.igen.split (k)                  // groups of indices
                      else       VectorI (0 until m).split (k)

        for (idx <- indices) {
            val (x_e, x_r) = x.splitRows (idx)                           // test, training data/input matrices
            val (y_e, y_r) = y.splitRows (idx)                           // test, training response/output matrices

            train (x_r, y_r)                                             // train the model
            eval (x_e, y_e)                                              // evaluate model on test dataset
            for (j <- 0 until ny) {
                val fit_j = fitA(j)
                val qof   = fit_j.fit                                    // get quality of fit measures
                val qsz   = qof.size
                if (qof(index_sst) > 0.0) {                              // requires variation in test set
                    for (q <- qof.range) stats(j*qsz + q).tally (qof(q))    // tally these measures
                } // if
            } // for
        } // for

        stats
    } // crossValidate

} // PredictorMat2 abstract class


//::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::
/** The `PredictorMat2` object provides functions for rescaling data and performing
 *  analysis.
 */
object PredictorMat2
{
    //:::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::
    /** Rescale the input/data matrix 'x' to the bounds of the "first" activation
     *  function 'f'; otherwise normalize.  Return the rescaled matrix.
     *  @param x  the input/data matrix
     *  @param f  the activation function family (first)
     */
    def rescaleX (x: MatriD, f: AFF): MatriD =
    {
        if (f.bounds != null) {                                 // scale to bounds of f
            val (min_x, max_x) = (min (x), max (x))
            scale (x, (min_x, max_x), f.bounds)
        } else {                                                // normalize: Normal (0, 1)
            val (mu_x, sig_x) = (x.mean, stddev (x))
            normalize (x, (mu_x, sig_x))
        } // if
    } // rescaleX

    //:::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::
    /** Rescale the output/response vector 'y' to the bounds of the "last" activation
     *  function 'f'; otherwise normalize.  Return the rescaled vector and the
     *  rescaling inverse function.
     *  @param y  the output/response vector
     *  @param f  the activation function family (last)
     */
    def rescaleY (y: VectoD, f: AFF): (VectoD, FunctionV_2V) =
    {
        if (f.bounds != null) {                                 // scale to bounds of f
            val (min_y, max_y) = (y.min (), y.max ())
            (scaleV ((min_y, max_y), f.bounds)(y),
             unscaleV ((min_y, max_y), f.bounds) _)             // rescaling inverse 
        } else {                                                // normalize: Normal (0, 1)
            val (mu_y, sig_y) = (y.mean, stddev (y))
            (normalizeV ((mu_y, sig_y))(y),
             denormalizeV ((mu_y, sig_y)) _)                    // rescaling inverse
        } // if
    } // rescaleY

    //:::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::
    /** Analyze a dataset using the given model using ordinary training with the
     *  'train' method.
     *  @param model  the model to be used
     */
    def analyze (model: PredictorMat2)
    {
        println (model.train ().eval ().report)
    } // analyze

    //:::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::
    /** Analyze a dataset using the given model where training includes
     *  hyper-parameter optimization with the 'train2' method.
     *  @param model  the model to be used
     */
    def analyze2 (model: PredictorMat2)
    {
        println (model.train2 ().eval ().report)
    } // analyze2

} // PredictorMat2 object

