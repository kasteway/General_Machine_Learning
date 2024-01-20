# Regularization Techniques to solve Overfitting:

These techniques help in balancing the model's ability to fit the training data while maintaining its capacity to generalize well to new data.

![Screenshot 2024-01-20 at 7 55 37 AM](https://github.com/kasteway/General_Machine_Learning/assets/62068733/81a638f0-8ff3-486d-ba79-358a470b4feb)

## Choosing  Lamda:

- If Lamda = 0 --> The model will overfit, because we are not really applying regularization since it zeros out the regularization terms.
- If lamda = Large # ex. 1,000  ---> The model will underfit because all the features will be almost 0.

## Therefore: Choose Lamda:
- That appropriately balances minimizing the mean squared error & keeping the parameters small

### 1. L1 Regularization (Lasso Regression): Adds a penalty equal to the absolute value of the magnitude of coefficients. This can lead to some coefficients being zero, thereby performing feature selection.
 
  - Lasso regression is particularly useful when you have a high number of features and you suspect that only a subset of them are important, or when you want a model that is easy to interpret.
  
  - L1 Regularization, commonly known as Lasso Regression (Least Absolute Shrinkage and Selection Operator), is a type of regularization technique used to prevent overfitting in machine learning models, particularly in linear regression models.
  
  - Penalty on Absolute Size of Coefficients: L1 regularization adds a penalty to the loss function equal to the absolute value of the coefficients.
  
  - Feature Selection: One of the distinctive properties of L1 regularization is its ability to reduce some coefficients to exactly zero. This means it not only prevents overfitting but also performs feature selection, identifying and using only the most important features in the model.
  
  - Sparse Solutions: By reducing coefficients to zero, L1 regularization leads to sparse models, which can be beneficial when dealing with high-dimensional data, as it simplifies the model and makes it more interpretable.
  
  - Tuning of Lamda: The strength of the regularization is controlled by the parameter Lamda. A higher value of Lamda increases the penalty, leading to more coefficients being shrunk to zero. The optimal value of Lamda is usually found through cross-validation.

### 2. L2 Regularization (Ridge Regression): Implements a penalty equal to the square of the magnitude of coefficients. This doesn't set coefficients to zero but makes them smaller, diminishing the model's complexity.
  
  - Ridge regression is particularly useful when data suffer from multicollinearity (independent variables are highly correlated). By adding regularization, it helps to mitigate the problem of multicollinearity.
  
  - L2 regularization is a common choice for regularization in many machine learning models, particularly when the goal is to prevent overfitting without necessarily reducing the model to only a subset of the features.
  
  - L2 regularization adds a penalty to the loss function equivalent to the square of the magnitude of the coefficients.
  
  - Unlike L1 regularization which can set some coefficients to zero, L2 regularization shrinks the coefficients towards zero but never exactly to zero. This means all features are kept in the model but their impact on the outcome is reduced.
  
  - By penalizing the sum of the squares of the coefficients, it controls the complexity of the model, as large coefficients (which can lead to overfitting) are particularly penalized.
  
  - Tuning of Lamda: The strength of the regularization is controlled by the parameter Lamda. A higher value of Lamda increases the penalty, leading to more coefficients being shrunk to zero. The optimal value of Lamda is usually found through cross-validation.
  
  - Unlike L1 regularization, which can abruptly eliminate a variable by setting its coefficient to zero, L2 regularization smoothly adjusts the coefficients, often leading to better generalization.
    
### 3. Elastic Net: Combines L1 and L2 regularization. It adds both penalties to the model, effectively blending the feature selection of L1 and the smoothing of L2.
  
  - Elastic Net is a regularization technique used in linear regression models that combines the properties of both L1 and L2 regularization (Lasso and Ridge regression, respectively). It's designed to blend the simplicity of L1 regularization with the stability of L2. 
 
  - Elastic Net is particularly advantageous in situations with large numbers of correlated features, where Lasso might be too aggressive in feature elimination and Ridge might not enforce enough sparsity.
  
  - Combination of L1 and L2 Regularization: Elastic Net adds both L1 and L2 penalties to the loss function.
  
  - Feature Selection and Shrinkage: Like Lasso, Elastic Net can reduce some coefficients to zero, performing feature selection. At the same time, it shrinks other coefficients like Ridge regression, making it effective in scenarios where multiple correlated features predict similar outcomes.
  
  - Control of Regularization Mix: The parameter α controls the balance between L1 and L2 regularization. When α = 1, Elastic Net is equivalent to Lasso; when α = 0, it is equivalent to Ridge regression.
  
  - Useful in Highly Correlated Data: Elastic Net is particularly useful in situations where there are multiple features that are correlated with each other. Lasso might randomly select one of these features while Elastic Net is more likely to include all of them, but with smaller coefficients.
  
  - Robust to Overfitting and Multicollinearity: By combining the properties of both L1 and L2 regularization, Elastic Net is robust against overfitting and can handle multicollinearity better than Lasso or Ridge alone.
  
  - Tuning of Parameters: The optimal values for Lamda & α are typically found through a cross-validation process, which involves testing different combinations of these parameters to find the one that results in the best model performance.

### 4. Pruning (for Decision Trees): Removes parts of trees that provide little power to classify instances, reducing complexity and improving model generalization.
  
  - Pruning in the context of decision trees is a technique used to reduce the complexity of the model and prevent overfitting. In machine learning, decision trees can grow very complex with many branches, which might fit the training data too closely, capturing noise and leading to poor performance on new data.
  
  - Pruning is an essential step in decision tree algorithms and is widely used in ensemble methods like Random Forests and Gradient Boosted Trees, although these methods inherently reduce the risk of overfitting through their ensemble nature.
  
  - Reducing Tree Size: Pruning involves cutting off branches (removing splits) from the tree. This reduces the size of the tree, making it simpler and more generalizable to new data.
  
  - Types of Pruning:

     - Pre-pruning (Early Stopping): Involves setting constraints during the tree-building process, such as a maximum depth, minimum number of samples required to split a node, or a minimum gain in reduction of impurity. This stops the tree from becoming too complex in the first place.

    - Post-pruning: Involves first allowing the tree to grow fully, and then removing branches that do not provide significant predictive power. This is typically done by evaluating the tree's performance on a validation set.

    - Cost Complexity Pruning (a.k.a. Weakest Link Pruning): A common method of post-pruning. It involves finding a subtree that increases the cross-validated error the least when removed. The algorithm goes through the tree, evaluates the effect of removing each subtree, and removes those that improve the model's performance on validation data.


  - Trade-off Between Bias and Variance: By pruning a tree, the variance is reduced (less overfitting) at the expense of a slight increase in bias (potentially underfitting). The goal is to find the right balance for optimal generalization.

  - Improves Interpretability: A pruned tree is often more interpretable due to its simplicity. It's easier to understand and visualize a smaller tree.

  - Enhances Generalization: Pruned trees tend to generalize better to unseen data. They are less likely to have learned noise and specific patterns present only in the training data.



### 5. Early Stopping: Halts the training process as soon as the performance on a validation set starts deteriorating, preventing the model from learning noise in the training data.

  - Early Stopping is a simple yet effective method to ensure that a machine learning model, especially a neural network, is neither underfit nor overfit, but just right for the given data.

  - Monitoring Validation Performance: During the training of a model, its performance is continually monitored on a separate validation dataset that is not used for training.

  - Stopping Criterion: Training is stopped as soon as the performance on the validation set starts to degrade or fails to improve for a specified number of epochs. This degradation is often measured in terms of an increase in validation error or a decrease in validation accuracy.

  - Balance Between Underfitting and Overfitting: Early Stopping aims to stop the training process at the point where the model is complex enough to capture the underlying patterns in the data (avoiding underfitting), but not so complex that it starts to capture noise (avoiding over fitting).

  - Saving the Best Model: Typically, the model's state at the point where it performed the best on the validation set is saved, as this represents the most generalizable version of the model.

  - No Need for a Separate Test Set: Unlike some other regularization techniques, Early Stopping does not require a separate test set for tuning hyperparameters, as the validation set serves this purpose.

  - Applicability: Early Stopping is widely used in training deep neural networks where the number of epochs required for training is not known beforehand and can vary significantly depending on the complexity of the task and the architecture of the network.

  - Practical and Efficient: It's a practical approach to avoid overfitting without the need to precisely tune regularization hyperparameters. It also often leads to reduced training time as the model is not trained for unnecessary epochs.


### 6. Dropout (for Neural Networks): Randomly drops units (and their connections) from the neural network during training, preventing over-reliance on certain features.

  - Dropout is a simple yet effective technique that has proven to be highly successful in reducing overfitting in complex neural networks. It's one of the key innovations that has enabled the success of deep learning in various challenging domains.

  - Random Deactivation of Neurons: During training, Dropout randomly deactivates a subset of neurons (and their corresponding connections) in a layer of the neural network. This means these neurons do not participate in forward and backward passes during that particular training phase (epoch).

  - Reduces Over-reliance on Certain Neurons: By deactivating neurons randomly, Dropout prevents the network from becoming overly dependent on any specific set of neurons. This can be thought of as forcing the network to learn more robust features that are useful in conjunction with many different random subsets of the other neurons.

  - Probabilistic Approach: Dropout is defined by a probability p, which is the chance that any given neuron is dropped during a training pass. This probability is a hyperparameter that can be tuned.

  - Training and Testing Phase Differences: During training, Dropout is applied, but at test time, all neurons are used (no dropout is applied). However, the output of each neuron is typically scaled by the dropout rate to balance the fact that more neurons are active during testing than training.

  - Mimics Ensemble Learning: Conceptually, Dropout is similar to training a large number of different neural networks (an ensemble) and then averaging the results. Each training iteration represents a different "thinned" network, with a different subset of neurons.

  - Improved Generalization: By preventing units from co-adapting too much, Dropout improves the generalization of the model to new, unseen data.

  - Widely Used in Deep Learning: Dropout has become a standard tool in training deep neural networks, particularly those used in tasks like image and speech recognition, where it has been shown to significantly improve performance.

### 7. Data Augmentation (for Deep Learning): Generates new training samples by altering the existing ones, increasing data diversity and thus reducing overfitting.

  - Data Augmentation is a key technique in deep learning that enhances the quantity and quality of training data, leading to better model performance, especially in fields where collecting large datasets is challenging or expensive.

  - Data Augmentation is a technique used in deep learning to increase the diversity of training data without actually collecting new data. It's particularly useful for tasks like image and speech recognition. 

  - Generating New Training Samples: Data augmentation involves creating modified versions of the existing training data. For instance, in image processing, this could mean creating new images through transformations like rotation, scaling, cropping, flipping, or changing brightness and contrast.

  - Prevents Overfitting: By expanding the dataset with varied but realistic transformations, data augmentation helps the model learn from a broader range of data patterns. This reduces the model's tendency to overfit to the specific details of the original training set.

  - Improves Model Robustness: Augmented data help in training models that are more robust and perform better on new, unseen data. The model learns to generalize from a more diverse set of examples.

  - Widely Used in Image and Speech Recognition: In image recognition, common augmentations include geometric transformations, color space adjustments, and random erasing. In speech recognition, augmentations may involve adding noise, changing pitch, or altering speed.

  - Efficient Use of Limited Data: Data augmentation is particularly valuable when the amount of available training data is limited. It artificially increases the size of the dataset without the need for additional data collection.

  - Application-Specific Techniques: The type and extent of augmentation can vary depending on the specific application and requirements. For example, in medical image analysis, careful consideration is given to ensure that augmentations do not create unrealistic images.

  - Incorporation in Training Pipeline: Data augmentation is typically incorporated into the training pipeline, so new data variations are generated on-the-fly during training. This keeps the memory requirements manageable, as not all variations need to be stored at once.

---

