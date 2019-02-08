# Naive-Bayes-using-Apache-Spark
Implementation of non-blocking Naive Bayes Classifier

In this project report, an overview of Apache Spark and its advantages over other streaming databases was discussed along with the properties that made it beneficial for my Naïve Bayes Classifier implementation. I gave a brief description of the dataset I used for modeling the classifier. I designed and implemented the Naïve Bayes Classifier purely in SQL using Spark SQL as the DSMS and python as the facilitator while achieving an accuracy of 77.3% in predicting the test data tuples.
I generalized my implementation for all datasets by converting the training data set into verticalized data set. Also, I ensured a non-blocking implementation of Naïve Bayes Classifier. The prediction of each test tuple is output immediately after its processing and the accuracy of prediction is output after periodic intervals of batches. 

The Dataset

Titanic Dataset: The titanic dataset gives the values of four categorical attributes for each of the 2201 people on board the Titanic when it struck an iceberg and sank. 

The attributes are 
•	classtype (first class 1st, second class 2nd, third class 3rd, crewmember) 
•	age (adult or child)
•	sex (male or female)  
•	survival (yes or no)

The question of interest for this natural dataset is how survival relates to the other attributes. The real interest is in interpretation and success at prediction would appear to be closely related to the discovery of interesting features of the relationship. Note that there are only sixteen possible combinations of input attributes for this prediction task, so the interesting behavior will be that with small training sets.

Usage: prediction of survival

Dataset format: .data file

Number of attributes: 4

Number of cases: 2,201


Link: (http://www.cs.toronto.edu/~delve/data/titanic/desc.html)
