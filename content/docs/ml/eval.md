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
| **Type I Error** | reject {{< math >}}$ H_0$ when true (predict P when N) | $P(FP) ${{</ math>}} |
| **Type II Error** | accept {{< math >}}$ H_0$ when false (predict N when P) | $P(FN) ${{</ math>}} |
| **Accuracy** | % of correct predictions | {{< math >}}$ P(TP)+P(TN) ${{</ math>}} |
| **Precision** | % of actual Ps among predicted Ps | {{< math >}}$ \frac{TP}{TP+FP} ${{</ math>}} |
| **Recall/Sensitivity** | % of predicted Ps among actual Ps | {{< math >}}$ \frac{TP}{TP+FN} ${{</ math>}} |
| **Specificity** | % of predicted Ns among actual Ns | {{< math >}}$ \frac{TN}{TN+FP} ${{</ math>}} |
| **F1-score** | Balance tradeoff between precision & recall | {{< math >}}$ 2\frac{\text{prec}\*\text{rec}}{\text{prec}+\text{rec}} ${{</ math>}} |
| **F{{< math >}}$ \boldsymbol{\beta}$-score** | Place importance on recall $\beta$ times of precision | $(1+\beta^2)\frac{\text{prec}\*\text{rec}}{(\beta^2\*\text{prec})+\text{rec}} ${{</ math>}} |

### ROC & AUC

<center>
<img width=500 src="/images/ml/ROC.png"/>
<br>
<a href="https://scikit-learn.org/stable/modules/generated/sklearn.metrics.RocCurveDisplay.html#sklearn.metrics.RocCurveDisplay">source</a>
</center>

- **TPR (True Positive Rate)**: = Recall, {{< math >}}$ \frac{TP}{TP+FN} ${{</ math>}}
- **FPR (False Positive Rate)**: = 1 -- Specificity, {{< math >}}$ \frac{FP}{TN+FP} ${{</ math>}}
- **Classification threshold**: anything above this defined value is classified as 1, {{< math >}}$ P(y=1) ${{</ math>}} (default 0.5)
- **ROC (Receiver Operating Characteristic)**: performance at different classification thresholds
- **AUC (Area Under the Curve)**: sum up performance across all classification thresholds
    - {{< math >}}$ \text{AUC}=0.5 ${{</ math>}}: random guess (worst)
    - {{< math >}}$ \text{AUC}=1 ${{</ math>}}: all correct (best)
    - {{< math >}}$ \text{AUC}=0 ${{</ math>}}: all wrong (reverse)

&nbsp;

## Regression
### Errors
Basically loss functions.

| Metric | Def | Formula |
|:-------|:----|:-------:|
| **MAE** | Mean of Residuals | {{< math >}}$ \frac{1}{m}\sum_{i=1}^{m}\|y_i-\hat{y}_i\| ${{</ math>}} |
| **MSE** | Variance of Residuals | {{< math >}}$ \frac{1}{m}\sum_{i=1}^{m}(y_i-\hat{y}_i)^2 ${{</ math>}} |
| **RMSE** | Standard deviation of Residuals | {{< math >}}$ \sqrt{\frac{1}{m}\sum_{i=1}^{m}(y_i-\hat{y}_i)^2} ${{</ math>}} |

Notes:
- MAE is robust against outliers, but MAE doesn't penalize large errors heavily and is not differentiable.
- MSE penalizes large errors heavily and is differentiable, but MSE is sensitive to outliers.
- RMSE has the same units as the dependent variables, so it is generally more preferred than both.

### Explained Variance

| Metric | Def | Formula |
|:-------|:----|:-------:|
| **Explained Variance** | % of {{< math >}}$ \text{Var}[y]$ explained by the $x$s in the model | $1-\frac{\text{Var}[y-\hat{y}]}{\text{Var}[y]}=\frac{\text{Var}[\hat{y}]}{\text{Var}[y]} ${{</ math>}} |
| {{< math >}}$ \boldsymbol{R}^\textbf{2}$ | % of $\text{Var}[y]$ explained by the $x$s in the model,<br> but account for systematic offset in prediction | $1-\frac{\text{Var}[y-\hat{y}]}{\text{Var}[y-\bar{y}]} ${{</ math>}} |
| **Adjusted** {{< math >}}$ \boldsymbol{R}^\textbf{2}$ | $R^2$, but account for #$x$s | $1-\left[\frac{(1-R^2)(m-1)}{(m-n-1)}\right] ${{</ math>}} |

Notes:
- Explained variance measures how well the independent variables explain the variance in dependent variables.
- {{< math >}}$ R^2 ${{</ math>}} is also called Coefficient of Determination, or Goodness of Fit, because it can capture how well unseen samples are likely to be predicted by the model.
- Adjusted {{< math >}}$ R^2$ accounts for the issue that $R^2$ automatically increases when #$x ${{</ math>}}s increases, which is undesired due to redundancy.
- Adjusted {{< math >}}$ R^2$ is always less than or equal to $R^2 ${{</ math>}}.


<!-- # Cross Validation
Problem: Model overfits on training data.

Solution: 

# Hyperparameter Tuning
 -->
