# Bank_Marketing
Working on the Bank Marketing Dataset of a Portuguese banking institution. Dataset taken from UCI


# Contents

-   [Problem Statement](https://github.com/a-anurag1024/Bank_Marketing/blob/main/problem_statement.txt)
-   [Dataset Information](https://github.com/a-anurag1024/Bank_Marketing/blob/main/dataset_given_details.txt)
-   [Classifiers](#classifiers)
-   [Notebooks Index](#notebook-index)


# Classifiers
Notes on work done towards the Primary Objective of the Problem statement.

### A note on the evaluation Metrics

The primary metrics used in the [paper](https://www.semanticscholar.org/paper/A-data-driven-approach-to-predict-the-success-of-Moro-Cortez/cab86052882d126d43f72108c6cb41b295cc8a9e) which is being used for this project are AUC and ALIFT. But because the dataset has a highly skewed target feature (88% has negative as value), it is not wise to either use either AUC or accuracy as a metric because they would by default give pretty good scores. In our case, even though we always get good values for these two metrics, we can easily see the bad quality of the classifiers when see Precision or Recall. So, instead of using AUC as a metric, in this project, **AuPRC (Area under Precision Recall Curve)** is used. And the AUC metric is used only to compare the results with the paper.


## Approaches tried:- 
1. [Basic ML Classifiers](#basic-ml-classifiers)  
2. [Neural Net Classifier](#neural-net-classifier)
3. [With PCA](#with-pca)
4. [Comparision with previous result](#comparision-with-previous-results)

#### Basic ML classifiers
Among the classifiers tried (Logistic Regression, Decision Trees, Random Forest Classifier, XGBoost Classifier), the XGBoost classfier gives the best results with evaluation metrics (on dev set) as:- 
```
Precision: 0.6084905660377359
Recall: 0.5194630872483221
F1: 0.5604634322954382
Accuracy: 0.9078627808136005
AUC: 0.9351117094165334
AUPRC: 0.6195725679242132
```

#### Neural Net Classifier
Two approaches were tried. First with original unbalanced dataset and then with a shorter balanced dataset per epoch. The NN-net used for both of the approches were 4 layered ANN. Adam optimizer with Cross Entropy loss function was used for training. Since, with 4 dense layers, the number of parameters were already very high (12,078) given the limited number of examples we had, number of parameters were not increased. Also, since in the training it was observed that the validation error was similar to training error, no regularizations were implemented. The trainings are done in Batch Gradient Decent Method as the number of training samples is small enough for the GPU memory.
<br>

The First approach with unbalanced dataset had performance better to the XGBoost model.
```
Precision: 0.7631578947368421
Recall: 0.4027777777777778
F1: 0.5272727272727273
Accuracy: 0.921092564491654
AUC: 0.930200643573727
AUPRC: 0.632742475851821
```
<br>

In the second approach, a balanced dataset per epoch has been taken, since we are using categorical cross entropy as loss function. By giving equal representation of both positive and negative cases in each batch(epoch), the target loss function has equal representation from both the classes and has a higher room for better fitting. Since, the number of positive cases are less, so all the positive cases along with randomly selected same number of negative cases were used in each epoch. This random selection of negative cases ensured that most of the negative cases are represented. The results were way better than the earlier approach with the XGBoost. **They also outperformed the XGBoost method at AUPRC and Recall but with a lower Precision**. 
```
Precision: 0.46714285714285714
Recall: 0.8605263157894737
F1: 0.6055555555555555
Accuracy: 0.8706739526411658
AUC: 0.9397617671495141
AUPRC: 0.6800074381392581
```
As the Recall value is higher here, for the purpose of our use-case (where we can afford to have some False Positives when we are able to capture most of the True positives) this model seems to be more fitting even though we have lower accuracy and precision.


#### With PCA
With the help of PCA, the highly correlated Socio-Economic features were reduced to a lower dimensional input. And using these new transformed (and fewer) features, a new XGBoost classifier was trained. The classifier showed improved Recall, at similar precision, but overall the improvement is very minor.
```
Precision: 0.6034214618973561
Recall: 0.5315068493150685
F1: 0.5651857246904589
Accuracy: 0.9093806921675774
AUC: 0.9371181898539405
AUPRC: 0.6063654649372963
```


#### Comparision with previous results

Both the NN models outperformed in terms of AUC and ALIFT compared to the best model in [previous results](https://www.semanticscholar.org/paper/A-data-driven-approach-to-predict-the-success-of-Moro-Cortez/cab86052882d126d43f72108c6cb41b295cc8a9e) (`AUC = 0.8` and `ALIFT = 0.7`).

The NN model trained with the original composition of the dataset outperformed the the NN model with the artificially balanced train dataset as the test dataset composition is similar is more similar to the dataset used to train it compared to the other one. It gave `AUC = 0.86` and `ALIFT = 0.82`

The NN Model with unbalanced dataset also gave better results compared to the previous results with `AUC = 0.86` and `ALIFT = 0.72`. This model underperformed in ALIFT wrt to the basic NN model (due to train, test dataset composition mismatch) but it definitely is a more desirable model because it has way higher `recall = 0.68` compared to the very low `recall = 0.21` in the basic NN model

**NOTE:-** The LIFT analysis done here is using a cummalative lift score rather the usual lift score that is used in most cases. This is done so to replicate the metric used in the previous result.


# Notebook Index

-   [EDA notebook](/notebooks/explore_data.ipynb) :- Notes on null values, outliers, skewed features and correlations in the given dataset
-   [Data Cleaning](/notebooks/data_cleaning.ipynb) 
-   [Feature Engineering](/notebooks/feature_engineering.ipynb) :- Feature engineering Strategy and pipeline
-   [Basic ML Classifiers](/notebooks/ml_classifiers.ipynb) :- Modelling with ML classifiers and their evaluations
-   [Neural Net Classifiers](/notebooks/neural_net_classifier.ipynb) :- Approaches with the neural nets
-   [XGBoost with PCA](/notebooks/with_PCA.ipynb)
-   [LIFT analysis](/notebooks/alift.ipynb) :- Alift analysis and test evaluation