#execfile('/home/pratik/Downloads/240b.py')
#rm   metastore_db/*.lc

from pyspark.sql import SQLContext,Row

from pyspark.sql.functions import array, col, explode, struct, lit

#Function to return a verticalized dataframe (key-value pairs)
def to_long(df, by):

    # Filter dtypes and split into column names and type description
    cols, dtypes = zip(*((c, t) for (c, t) in df.dtypes if c not in by))
    # Spark SQL supports only homogeneous columns
    assert len(set(dtypes)) == 1, "All columns have to be of the same type"

    # Create and explode an array of (column_name, column_value) structs
    kvs = explode(array([
      struct(lit(c).alias("key"), col(c).alias("val")) for c in cols
    ])).alias("kvs")

    #Returning a verticalized dataframe where we converted the class attributes into key-value pairs.
    return df.select(by + [kvs]).select(by + ["kvs.key", "kvs.val"])


sqlContext = SQLContext(sc)
#Importing the titanic dataset
data = sc.textFile("/home/pratik/Downloads/Dataset.data")
#Dividing the data into training and testing data (70-30 split)
weights = [.70, .30]
seed = 40
rawTrainData,rawTestData = data.randomSplit(weights,seed)

#Splitting the training data lines and dividing into classes
partsTrain = rawTrainData.map(lambda l: l.split())
titanicTrain = partsTrain.map(lambda p: Row(classtype=p[0], age=p[1], sex=p[2], survival=p[3]))
#Creating a dataframe for the training data in SQL
schemaTitanicTrain = sqlContext.createDataFrame(titanicTrain)
schemaTitanicTrain.registerTempTable("titanicTrain")

#Splitting the testing data lines and dividing into classes
partsTest = rawTestData.map(lambda l: l.split())
titanicTest = partsTest.map(lambda p: Row(classtype=p[0], age=p[1], sex=p[2], survival=p[3]))
#Creating a dataframe for the testing data in SQL
schemaTitanicTest = sqlContext.createDataFrame(titanicTest)
schemaTitanicTest.registerTempTable("titanicTest")

#Calculating total number of training and testing set tuples
totalTrainCount = titanicTrain.count()
totalTestCount = titanicTest.count()

print("Training dataset")
print(schemaTitanicTrain.show())
print("Testing dataset")
print(schemaTitanicTest.show())

#Creating a vertical dataframe by storing all the features into two columns of key-value pairs
verticalTitanicTrainDF=to_long(schemaTitanicTrain, ["survival"])
verticalTitanicTrainDF.registerTempTable("verticalTitanicTrainDF")
print "Verticalized Training Dataset"
print verticalTitanicTrainDF.show()

#Counts of the conditional probability of each feature wrt to its class attribute
counts = sqlContext.sql("SELECT key, val, survival, count(*) AS Freq FROM verticalTitanicTrainDF GROUP BY survival,key,val")
counts.registerTempTable("freq")

print ("Frequency Table")
print counts.show()

#Setting values of the column labels to colNames
colNames = ["age", "classtype", "sex", "survival"]

#Setting the values of the class variable to survivalClass
survivalClass = ["yes", "no"]

#Starting the stream of test data on our implementation of Naive Bayes Classifier
correctClassified=0
iterate=0

#Giving each row of test dataset as input to the Naive Bayes Classifier
for row in titanicTest.collect():
	results = []
	print("Test Data Tuple")
	print row
	for target in survivalClass:
		
		frequency = []

		#Calculating the number of occurences for 'yes' and 'no' ie; the class variable for each attribute/feature
		for i in range(0, len(row)-1):
			c1=sqlContext.sql("SELECT Freq FROM freq WHERE key = '{0}' and val = '{1}' and survival = '{2}'".format(colNames[i], str(row[i]), target)).collect()
			frequency.append(c1)
		
		#Calculating the sum of frequencies of all 'yes' and 'no'
		targetClass = sqlContext.sql("SELECT sum(Freq) FROM freq WHERE survival = '{0}' and key = '{1}' GROUP BY key, survival".format(target, colNames[0])).collect()
		#print targetClass

		#Calculating the class Probability ie; P(yes) or P(no) in our titanic dataset example
		classProb = float(float(targetClass[0][0])/float(totalTrainCount))
		
		#Calculating the probability of the test case being classified into a particular class using Bayes Theorem
		#Basically we multiply all the conditional probabilities and the class probability
		for freq in frequency:
			classProb *= float(float((freq[0][0])/float(targetClass[0][0])))
		results.append(classProb)

	#Index of the class in which the test tuple has been classified (yes/no)
	index  = results.index(max(results))
	
	#Checking whether the model correctly classified the data or not.
	#Keeping a count of correctly classified test cases in the variable correctClassified
	if row[len(row) - 1] == survivalClass[index]:
		correctClassified += 1


	#As soon as the test tuple has been classified into the class, it is output.
	#Hence this is how we are ensuring that our Naive Bayes Implementation is a Non Blocking one
	print "Row " + str(iterate) + " done"
	iterate+=1
	print "Prediction : " + str(survivalClass[index])

	#Calculating the accuracy after a periodic interval of 100 tuples
	if (iterate%100 == 0):
		print "Accuracy till now : " + str(float(correctClassified)/float(iterate))

#Calculating the printing the accuracy of our Naive Bayes Classifier.
#Accuracy can be found out by dividing the total number of correctly classified test cases divided by the total number of test cases.
print "Accuracy : " + str(float(float(correctClassified)/float(totalTestCount)))
