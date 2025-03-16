# Permutation feature importance

## Concept
* *What is it:* A model inspection technique that measures the contribution of each feature to a fitted model’s statistical performance
* *When to use it:* Following conditions are satisfied
	- Tabular dataset
	- Non-linear or opaque estimators
## Algorithm
1. Given a fitted predictive model $m$, a tabular dataset (training or validation) $D$
2. Compute the reference score $s$ of the model $m$ on data $D$ *e.g.* the accuracy for a classifier or the R-squared for a regressor
3. For each feature $j$ (column of $D$):
	1. For each repetition $k$ in $1,...,K$:
		1. Randomly shuffle column $j$ of dataset $D$ to generate a corrupted version of the data named $\tilde{D}_{k,j}$ 
		2. Compute the score $s_{k,j}$ of model $m$ on corrupted data $\tilde{D}_{k,j}$
	2. Compute importance $i_{j}$ for feature $f_{j}$ defined as $$i_{j}=s-\frac{1}{K} \sum\limits_{k=1}^{K} s_{k,j}$$

## Caution
* It is always important to evaluate the predictive power of a model using a held-out set (or better with cross-validation) prior to computing importances. *Reason:* Permutation importance does not reflect to the intrinsic predictive value of a feature by itself but how important this feature is for a particular model
* Permutation importances can be computed either on the training set or a held-out set (*i.e.* validation set or testing set). To show which features the model relied on most to make predictions on data it has seen, use the training set. To reveal which features are most important for the model's performance on unseen data, use a held-out set. If there are features that are important on the training set but not on the held-out set, that means overfit happens
* In general, different scoring metrics tend to produce similar rankings of feature importance. While the rankings may be similar, the actual numerical values of importance can vary significantly between different metrics. Despite the general trend of consistency, it's not guaranteed that all metrics will always produce the same feature importance rankings. This is particularly true in certain scenarios, such as imbalanced classification problems.
* The results are misleading when the features are correlated. *Reason:* When one of the features is permuted, the model can still access to the permuted feature i.e. same information through its correlated features. *Result:* The correlated features have low importance value even though they might be important.  *How to discover correlation:* None of the features show significant importance but the model has high performance on the test set

## Pros
* The interpretation is easy because feature importance is the increase of model error when the feature's information is destroyed
* It can be applied to any fitted estimators
* Can be calculated multiple times with different permutations of the feature. This can provide a measure of the variance or standard error of the estimated feature importances for the specific trained model

## Cons
* PFI reflects how important a feature is for a specific model, not its intrinsic predictive value. If the model is poorly trained or overfits, the feature importance scores may not be reliable
* The results of PFI depend on the choice of scoring metric used to evaluate model performance. Different metrics may yield different rankings of feature importance
* When features are correlated, permuting one feature can disrupt the relationship between correlated features, leading to unrealistic data instances. This can bias the importance scores, as the model's performance may degrade more than it would in realistic scenarios


## Reference
* scikit-learn documentation - [Permutation feature importance](https://scikit-learn.org/stable/modules/permutation_importance.html) 
- scikit-learn documentation - [Permutation Importance with Multicollinear or Correlated Features](https://scikit-learn.org/stable/auto_examples/inspection/plot_permutation_importance_multicollinear.html)