# Predicting House Prices

[[Source](https://www.kaggle.com/competitions/house-prices-advanced-regression-techniques)] 

### [Project](https://www.kaggle.com/code/chongzhenjie/house-prices-kernel-methods-tree-models): Predict sales price for each house.

* Analyzed dataset containing 79 explanatory variables describing almost every aspect of residential homes in Ames, Iowa with the goal of predicting the sales price.
* Used univariate statistical tests (F-test statistics and mutual information) to help in feature selection and compared the performance of kernel methods and tree models in terms of RMSE.


### Model Validation Summary.
![validation_summary](https://user-images.githubusercontent.com/77932796/222948629-c9c9b410-6d9f-4001-a7ee-309ae373dc44.png)

The ensemble model, which is the average of gradient boosting, kernel SVR and kernel ridge, has the smallest median value of the validation RMSE as indicated by the red vertical dashed line.
