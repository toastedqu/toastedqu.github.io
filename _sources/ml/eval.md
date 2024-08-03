---
title : "Evaluation"
description: ""
lead: ""
date: 2020-10-06T08:48:45+00:00
lastmod: 2020-10-06T08:48:45+00:00
draft: false
images: []
weight: 6
---
# Metrics
## Classification
### Confusion Matrix

<center>
<img width=500 src="/images/ml/confusion_matrix.png"/>
<br>
<a href="https://www.datacamp.com/tutorial/what-is-a-confusion-matrix-in-machine-learning">source</a>
</center>

| Metric | Def | Formula |
|:-------|:----|:-------:|
| **Type I Error** | reject $ H_0$ when true (predict P when N) | $P(FP) $ |
| **Type II Error** | accept $ H_0$ when false (predict N when P) | $P(FN) $ |
| **Accuracy** | % of correct predictions | $ P(TP)+P(TN) $ |
| **Precision** | % of actual Ps among predicted Ps | $ \frac{TP}{TP+FP} $ |
| **Recall/Sensitivity** | % of predicted Ps among actual Ps | $ \frac{TP}{TP+FN} $ |
| **Specificity** | % of predicted Ns among actual Ns | $ \frac{TN}{TN+FP} $ |
| **F1-score** | Balance tradeoff between precision & recall | $ 2\frac{\text{prec}\*\text{rec}}{\text{prec}+\text{rec}} $ |
| **F$ \boldsymbol{\beta}$-score** | Place importance on recall $\beta$ times of precision | $(1+\beta^2)\frac{\text{prec}\*\text{rec}}{(\beta^2\*\text{prec})+\text{rec}} $ |

### ROC & AUC

<center>
<img width=500 src="/images/ml/ROC.png"/>
<br>
<a href="https://scikit-learn.org/stable/modules/generated/sklearn.metrics.RocCurveDisplay.html#sklearn.metrics.RocCurveDisplay">source</a>
</center>

- **TPR (True Positive Rate)**: = Recall, $ \frac{TP}{TP+FN} $
- **FPR (False Positive Rate)**: = 1 -- Specificity, $ \frac{FP}{TN+FP} $
- **Classification threshold**: anything above this defined value is classified as 1, $ P(y=1) $ (default 0.5)
- **ROC (Receiver Operating Characteristic)**: performance at different classification thresholds
- **AUC (Area Under the Curve)**: sum up performance across all classification thresholds
    - $ \text{AUC}=0.5 $: random guess (worst)
    - $ \text{AUC}=1 $: all correct (best)
    - $ \text{AUC}=0 $: all wrong (reverse)



## Regression
### Errors
Basically loss functions.

| Metric | Def | Formula |
|:-------|:----|:-------:|
| **MAE** | Mean of Residuals | $ \frac{1}{m}\sum_{i=1}^{m}\|y_i-\hat{y}_i\| $ |
| **MSE** | Variance of Residuals | $ \frac{1}{m}\sum_{i=1}^{m}(y_i-\hat{y}_i)^2 $ |
| **RMSE** | Standard deviation of Residuals | $ \sqrt{\frac{1}{m}\sum_{i=1}^{m}(y_i-\hat{y}_i)^2} $ |

Notes:
- MAE is robust against outliers, but MAE doesn't penalize large errors heavily and is not differentiable.
- MSE penalizes large errors heavily and is differentiable, but MSE is sensitive to outliers.
- RMSE has the same units as the dependent variables, so it is generally more preferred than both.

### Explained Variance

| Metric | Def | Formula |
|:-------|:----|:-------:|
| **Explained Variance** | % of $ \text{Var}[y]$ explained by the $x$s in the model | $1-\frac{\text{Var}[y-\hat{y}]}{\text{Var}[y]}=\frac{\text{Var}[\hat{y}]}{\text{Var}[y]} $ |
| $ \boldsymbol{R}^\textbf{2}$ | % of $\text{Var}[y]$ explained by the $x$s in the model,<br> but account for systematic offset in prediction | $1-\frac{\text{Var}[y-\hat{y}]}{\text{Var}[y-\bar{y}]} $ |
| **Adjusted** $ \boldsymbol{R}^\textbf{2}$ | $R^2$, but account for #$x$s | $1-\left[\frac{(1-R^2)(m-1)}{(m-n-1)}\right] $ |

Notes:
- Explained variance measures how well the independent variables explain the variance in dependent variables.
- $ R^2 $ is also called Coefficient of Determination, or Goodness of Fit, because it can capture how well unseen samples are likely to be predicted by the model.
- Adjusted $ R^2$ accounts for the issue that $R^2$ automatically increases when #$x $s increases, which is undesired due to redundancy.
- Adjusted $ R^2$ is always less than or equal to $R^2 $.


<!-- # Cross Validation
Problem: Model overfits on training data.

Solution: 

# Hyperparameter Tuning
 -->
