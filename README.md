# SciKit_Learn Credit Risk Analysis
Prepared by Sheri Karasick <br />
July 9, 2020 <br />

## Summary and Resources
Scikit_Learn (SKL) is Python's machine learning module that is built on top of SciPy and NumPy to examine datasets in a machine learning context.  In this challenge, we examine credit risk using supervised learning techniques: random oversampling, undersampling and synthetic minority oversampling technique (SMOTE).  After running each of these classification and regression models, we analysis three key outputs to evaluate the efficacy of each side by side on the same dataset: the balanced accuracy score, precision score, and recall score.  

*Objective: Evaluate three machine learning models and determine which is most effective for the given dataset with credit analysis data.*

[Jupyter Notebook File](add ipynb file info)

## Analysis
Four models utilizing a variety of over and undersampling techniques were utilized to evaluate the efficacy of the models with the dataset utilized.
Given that different models perform differentially better with one dataset over another, testing with multiple models and evaluating the efficacy of the models side by side are critical steps in the machine learning process.  With these models, one does not fit all circumstances.

#### Results Table

      +--------------------------+------------------+------------------+------------------+------------------+
      |Model:                    |  Model 1:        |   Model 2:       |   Model 3:       |   Model 4:       |
      |                          |  Oversampling    |   SMOTE          |   Undersampling  |   SMOTEENN       | 
      +--------------------------+------------------+------------------+------------------+------------------+
      |Balanced Accuracy Score   |             0.65 |             0.65 |             0.55 |             0.55 |
      |Precision Score           |             0.99 |             0.99 |             0.99 |             0.99 |
      |Recall Score              |             0.56 |             0.69 |             0.41 |             0.57 |
      |F1 Score                  |             0.71 |             0.81 |             0.58 |             0.72 |
      +--------------------------+------------------+------------------+------------------+------------------+

#### Written Analysis

      ######Interpreting based on the balanced accuracy score
      The accuracy score is equal to the proportion of predictions that the model being implemented has correctly classified.  If the accuracy score is too close to one it may be an indicator that the model has been overtrained to the specific data set being used and may not be applicable with other datasets.  This is not the case with any of the accuracy scores from our three models in light of values for 1, 2, 3 and 4 being 0.65, 0.65, 0.55, and 0.55 respectively.  However, according to this parameter, models 1 and 2 outperformed models 3 and 4.

      ######Intrepreting based on the precision score
      The precision score informs us as to how many of the actual positives were identified correctly.  It reflects the number of true positives divided by the sum of true and false positives.  The precision score approximated 1 at 0.99, indicating accurate performance in not picking up false positives as part of the model. All four models performed equivantly and this metric is not useful in distinguishing between the validity of the three models by itself.

      ######Intrepreting based on the recall score
      This model was mroe informative than the precision score.  The recall score informs on how many positive values were missed by the model and identified as not members of the desired class when they should have been positive.  The value is calculated by dividing the number of accurately identified positive values by the sum of those values added to the false negatives, or those positives that were mislabeled. The recall scores for Group 1, 2, 3 were 0.56, 0.69, 0.41, and 0.57 respectively.  This value confers more value to the SMOTE model of oversampling than the other two models being examined.  

      ######F1 Scores
      One more test is important to note from the confusion matrix.  The F1 score is a measure of a test's accuracy, incorporating the precision and recall variables, true positives, as well as true negatives and false negatives.  When evaluating the F1 scores for the four tests, the values are consisent with the findings of other tests, indicating that the most robust test for this dataset is the SMOTE test with a value that is 9% more accurate than the SMOTEENN test. 

## Recommended Algorhythm for Anlaysis
Based on the data analysis, the SMOTE test is the most robust test for this dataset.  The SMOTE test was indicated by the balanced accuracy score, and strongly favored by both the recall score and the F1 Score.
