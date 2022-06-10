# 20-Newsgroups-Classification
This repositary contains my Major Project for the course Pattern Recognition and Machine Learning. 
# Problem Statement
This dataset is a collection of newsgroup documents. The 20 newsgroups collection has become a popular
data set for experiments in text applications of machine learning techniques, such as text classification
and text clustering.
# Description
We have basically used 8 classifcation models along with other essential Machine Learning  concepts.We used some functions from the scratch like TF-IDF and Naive Bayes. Ultimate goal of this project is to classify the data into respectable categories. 

# Description of DataSet
In this dataset, duplicate messages have been removed and the original messages only contain "From" and
"Subject" headers (18828 messages total).
Each new message in the bundled file begins with these four headers:
Newsgroup: alt.newsgroup
Document_id: xxxxxx
From: Cat
Subject: Meow Meow Meow
The Newsgroup and Document_id can be referenced against list.csv
Organization
Each newsgroup file in the bundle represents a single newsgroup
Each message in a file is the text of some newsgroup document that was posted to that newsgroup.
This is a list of the 20 newsgroups:
- comp.graphics
- comp.os.ms-windows.misc
- comp.sys.ibm.pc.hardware
- comp.sys.mac.hardware
- comp.windows.x rec.autos
- rec.motorcycles
- rec.sport.baseball
- rec.sport.hockey sci.crypt
- sci.electronics
- sci.med
- sci.space
- misc.forsale talk.politics.misc
- talk.politics.guns
- talk.politics.mideast talk.religion.misc
- alt.atheism
- Soc.religion.christian
# Challenges in Text Classification
Text data is often messed up. It contains hyperlinks, special symbols, etc., which need to be removed so
that we can use it for training ML models. This step is called the "preprocessing step" in ML. In this
project, we have not considered the sequence words, but rather focused on the frequency words.
Considering the sequence of words and applying techniques such as LSTM would give even more
accuracy than the techniques that use a frequency-based approach.
# Models Used
* Decision Tree Classifier
- It uses a tree-like data structure. When we have to classify an input the comparison starts from the root node and ends at one of the leaf nodes. At the leaf node the decision is made. The main idea behind Decision trees is that entropy should get reduced as we move down the tree.
* Random Forest Classifier
- This uses the concept of creating Bootstrap samples from the dataset. For each of the Bootstrap samples corresponding Decision trees are created. Then the class is assigned based on majority voting.
* Linear Support Vector Machines
- In SVM we try to find an optimal decision hyperplane along with decision margin. We can control the value of ‘C’ which decides the extent of misclassification allowed.
* Neural Networks
- In Neural Networks there are many individual neurons connected to each other. The network thus formed can learn non-linear data. This is possible only because of activation functions.
* K Means Clustering
- In KMeans Clustering we first assume centroids. Then based on the euclidean distance we determine clusters for each sample point. Then we recompute the centroid and repeat the process. We stop clustering after centroids get stable.
* Logistic Regression
- It can be thought of as a single neuron of Neural Network with activation function as Sigmoid. As this is a multiclass classification problem we use one vs all approach to predict the probabilities for each class.
* Naive Bayes
-In Naive Bayes the assumption is that each of features are independent and so p(x1,x2/w) = p(x1/w)*p(x2/w). Here we have used Multinomial Naive Bayes.
* Light GBM
LightGBM (Light Gradient Boosting Machine) gradient boosting framework. It is based on decision tree algorithms and can be used for classifications.
* Hyper-parameter tuning for models -
- GridSearchCv - In Grid Search we consider a list of values for each of the considered parameters and perform Cross Validation using the selected value during each iteration. 
# Concepts Used
* Data Preprocessing
* TF-IDF
* Feature Selection & ruduction
* Model Training
* Hyperparameter tuning
* Performance measure

# Results & Analysis
![image](https://user-images.githubusercontent.com/103515662/173074729-4f6ac22d-353b-4c5f-a2ae-84f4c6f3f0d5.png)

# Demo To Classify A User Specified News Article
We can specify custom inputs in the Custom Input Section of the notebook and it will classify it and
return the predicted class.
For example if the input given to the system was “The University of Tennessee pitcher threw a 105.5 mph
fastball in the eighth inning of the team’s win over Auburn. It was the fastest pitch ever thrown in college
baseball history, breaking his own record he set in March when he fired a 104 mph fastball against South
Carolina. The pitch just barely missed breaking Aroldis Chapman’s MLB record 105.8 mph pitch in
2010.” The output we got was rec.sport.baseball which is the correct output.

![image](https://user-images.githubusercontent.com/103515662/173075471-c9cee23c-783a-400b-8a6f-72de0ab3b5bf.png)

# Conclusion
From the above observation we conclude that SVM is the best classifier and it also took the least time to
train among all the models. This may be because of the fact that NLP data is typically high dimensional
and it is sparse, which means that different classes will be present in different areas of feature space. Thus
finding a hyperplane for classification is easier and it helps in generalization.
Tree based models (Decision Tree, Random Forest, Light GBM) took significantly more time to train.
Quality of clustering is not good because of the fact that the number of words which are not useful in
classification are significantly greater than the words which are helpful in classification. Thus there is
huge overlap among the clusters.








