# Project2-Shubhangi-Mingyu_Ye

Group Members- Shubhangi Rai , Mingyu Sun and Ye Tian

Shubhangi Rai-   Forward , Backward , Stepwise Regression, L1, L2 Regularization for Perceptron Model, 3 Layers Neural Network and XL-Neural Network models done on Python - kerasANN.ipynb

Report done for the work done

Configuration to Setup: 
Install Python Version 3.7 
Libraries included and needed to be installed in Python 
1. Scikit learn 
2. Pandas 
3. NumPy  
4. Tensorflow. 
5. Keras

How to run?

1. Open terminal Run jupyter notebook Open “.ipynb” file from jupyter notebook
2. Download each dataset and run separately by providing path for each dataset in the code

Data Sets used- Downloaded data sets from UCI Machine Learning Repository

1.AutoMPG 2.Concrete 3.Forest-fires 4.SkillCraft1.Dataset 5. Bias_correction_ucl


Mingyu Sun- Forward, Backward, Stepwise Regression for TranRegression, Perceptron Model, 3 Layers Neural Network and XL-Neural Network models with scalation.
Download project2.scala, PredictorMat.scala, PredictorMat2.scala, NeuralNet_XL.scala replace the PredictorMat.scala, PredictorMat2.scala, NeuralNet_XL.scala and add project2.scala under scalation_1.6/scalation_modeling/src/main/scala/scalation/analytics. The whole project2 should be run inside scalation_1.6, with the following code

cd scalation_1.6 $ ./build_all.sh cd scalation_modeling sbt runMain scalation.analytics.the_object

TranRegressionsel includes Forward, Backward, Stepwise Regression, similar for Perceptronsel, NeuralNet_3Lsel, NeuralNet_XLsel

Ye Tian- Ridge and Lasso Regression for TranRegression in python, tried forward and backward regression with help from Shubhangi, but unsuccessful; 4 Layers Neural Network model in scalation for Forward, Backward, Stepwise feature selection. (I was in charge of working on NNXL on scalation, but Mingyu also did that part. I also did my work on PredictorMat2.scala and NeuralNetXL.scala within scalation 1.6, but in order to distinguish from her file, I change the filename to PredictorMat3.scala and NeuralNetXL1.scala, respectively. To run the two files, it would be better to replace PredictorMat2.scala and NeuralNetXL.scala if they are already in scalation 1.6)

To run the scalation code, simply cd to the scalation_modeling and enter: sbt compile runMain scalation.analytics.object

Here is a list of object names: 
NN4L_forSel_AutoMPGTest
NN4L_backElim_AutoMPGTest
NN4L_stepReg_AutoMPGTest
NN4L_forSel_ConcreteTest
NN4L_backElim_ConcreteTest
NN4L_stepReg_ConcreteTest

(the report "report_first_draft_Sun" is only the first draft, Shubhangi and Ye may upload the final version.)






