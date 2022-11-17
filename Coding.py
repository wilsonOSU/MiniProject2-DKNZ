import os
print("username:" , os.getlogin())
from datetime import date
today = date.today()
print("Today's date:", today)
import findspark
findspark.init()
from pyspark.sql import SparkSession
from pyspark.conf import SparkConf
from pyspark.sql.types import * 
import pyspark.sql.functions as F
from pyspark.sql.functions import col, asc,desc,countDistinct,isnan,count,when
import matplotlib.pyplot as plt
import numpy as np
import seaborn as sns
from pyspark.sql import SQLContext
from pyspark.mllib.stat import Statistics
import pandas as pd
from pyspark.sql.functions import udf
from pyspark.ml.feature import OneHotEncoder, StringIndexer, VectorAssembler,StandardScaler
from pyspark.ml.evaluation import BinaryClassificationEvaluator
from pyspark.ml.classification import LogisticRegression
from pyspark.ml import Pipeline
from sklearn.metrics import confusion_matrix
from pandas.plotting._misc import scatter_matrix

spark=SparkSession.builder \
.master ("local[*]")\
.appName("minicase2")\
.getOrCreate()

sc=spark.sparkContext
sqlContext=SQLContext(sc)

import os
os.getcwd()

df=spark.read \
 .option("header","True")\
 .option("inferSchema","True")\
 .option("sep",";")\
 .csv("C:\\Users\\divin\\Downloads\\XYZ_Bank_Deposit_Data_Classification.csv")
print("There are",df.count(),"rows",len(df.columns),
      "columns" ,"in the data.") 

spark = SparkSession.builder.appName("churn").getOrCreate()
df.printSchema()

df = df.withColumnRenamed("emp.var.rate","emp_var_rate").withColumnRenamed("cons.price.idx","cons_price_index")\
    .withColumnRenamed("cons.conf.idx","cons_confidence_index").withColumnRenamed("nr.employed","number_employees")

###EDA    
#Distinct Values
df.agg(*(countDistinct(col(c)).alias(c) for c in df.columns)).toPandas().transpose()
#Null Values
df.select([count(when(isnan(c), c)).alias(c) for c in df.columns]).toPandas().transpose()

##Summary Stats
double_features = [t[0] for t in df.dtypes if t[1] == 'double']
int_features = [t[0] for t in df.dtypes if t[1] == 'int']
numeric_features = double_features + int_features
df.select(numeric_features).describe().toPandas().transpose()



##Correlation 
numeric_data = df.select(numeric_features).toPandas()
axs = scatter_matrix(numeric_data, figsize=(8, 8));
n = len(numeric_data.columns)
for i in range(n):
    v = axs[i, 0]
    v.yaxis.label.set_rotation(0)
    v.yaxis.label.set_ha('right')
    v.set_yticks(())
    h = axs[n-1, i]
    h.xaxis.label.set_rotation(90)
    h.set_xticks(())
plt.show()

##Skewness and Kurtosis
from pyspark.sql.functions import col, skewness, kurtosis
df.select(skewness('age'),kurtosis('age')).show()
df.select(skewness('emp_var_rate'),kurtosis('emp_var_rate')).show()
df.select(skewness('cons_price_index'),kurtosis('cons_price_index')).show()
df.select(skewness('cons_confidence_index'),kurtosis('cons_confidence_index')).show()
df.select(skewness('euribor3m'),kurtosis('euribor3m')).show()
df.select(skewness('duration'),kurtosis('duration')).show()
df.select(skewness('campaign'),kurtosis('campaign')).show()
df.select(skewness('pdays'),kurtosis('pdays')).show()
df.select(skewness('previous'),kurtosis('previous')).show()




###Distributions
fig = plt.figure(figsize=(2,10))
st = fig.suptitle("Distribution of Features", fontsize=40, verticalalignment = 'center')
for col,num in zip(df.toPandas().describe().columns, range(1,11)):
    ax = fig.add_subplot(3,4,num)
    ax.hist(df.toPandas()[col])
    plt.style.use('dark_background')
    plt.grid(False)
    plt.xticks(rotation=45, fontsize=10)
    plt.yticks(fontsize=10)
    plt.title(col.upper(), fontsize=10)
plt.tight_layout()
st.set_y(0.95)
fig.subplots_adjust(top=0.85, hspace=0.4)
plt.show()

#Pearson Correlation
numeric_features = [t[0] for t in df.dtypes if t[1] != 'string']
numeric_features_df=df.select(numeric_features)
numeric_features_df.toPandas().head()

col_names =numeric_features_df.columns
features = numeric_features_df.rdd.map(lambda row: row[0:])
corr_mat=Statistics.corr(features, method="pearson")
corr_df = pd.DataFrame(corr_mat)
corr_df.index, corr_df.columns = col_names, col_names
corr_df

html = corr_df.to_html()
text_file = open("minicase2pearson.html", "w")
text_file.write(html)
text_file.close()

#indexer that provides index value for the categorical variables
indexer = StringIndexer(inputCols=['job','marital','education','default','housing','loan','contact','month','day_of_week','poutcome'
],outputCols=['job_index','marital_index','education_index','default_index','housing_index','loan_index','contact_index','month_index','day_of_week_index','poutcome_index'
])
df1 = indexer.fit(df).transform(df)
df1.show()

#one hot encode all the categorical variables
encoder = OneHotEncoder().setInputCols(['job_index','marital_index','education_index','default_index','housing_index','loan_index','contact_index','month_index','day_of_week_index','poutcome_index'
]).setOutputCols(['job_encoded','marital_encoded','education_encoded','default_encoded','housing_encoded','loan_encoded','contact_encoded','month_encoded','day_of_week_encoded','poutcome_encoded'])
df2 =encoder.fit(df1).transform(df1)
df2.toPandas().head()

#encode the label (target) variable
label_indexer = StringIndexer().setInputCol("y").setOutputCol("label")

df3 = label_indexer.fit(df2).transform(df2)

df3.select("y","label").toPandas().head()


#vector assembler, put everything into one dataframe column

assembler = VectorAssembler()\
         .setInputCols (['age','duration','campaign','pdays','previous','marital_encoded','education_encoded','default_encoded','housing_encoded',\
                         'loan_encoded','contact_encoded','month_encoded','day_of_week_encoded','job_encoded','poutcome_encoded',\
                         'emp_var_rate','cons_price_index','cons_confidence_index','number_employees','euribor3m'])\
         .setOutputCol ("vectorized_features")
        
# In case of missing you can skip the invalid ones
df3=assembler.setHandleInvalid("skip").transform(df3)
df3.toPandas().head()

#standard scaler, scales the features to be similar ranges

scaler = StandardScaler()\
         .setInputCol ("vectorized_features")\
         .setOutputCol ("features")
        
scaler_model=scaler.fit(df3)
df4=scaler_model.transform(df3)
pd.set_option('display.max_colwidth', 40)
df4.select("vectorized_features","features").toPandas().head(5)


#split data into test/train
finalized_data = df4.select('features','label')
train, test = finalized_data.randomSplit([0.7, 0.3],seed=2022)
print("Training Dataset Count: " + str(train.count()))
print("Test Dataset Count: " + str(test.count()))

#show the target variable split in the train group
train.groupby("label").count().show()

#logistic regression model
lr = LogisticRegression(labelCol='label')
lrn = lr.fit(train)
#create an evaluator instance
evaluator = BinaryClassificationEvaluator()


#show model performance for both test & train
predictions = lrn.transform(test)
print("Test Area Under ROC: " + str(evaluator.evaluate(predictions, {evaluator.metricName: "areaUnderROC"})))
print("Test Area Under PR: " + str(evaluator.evaluate(predictions, {evaluator.metricName: "areaUnderPR"})))
roc = lrn.summary.roc.toPandas()
plt.plot(roc['FPR'],roc['TPR'])
plt.ylabel('False Positive Rate')
plt.xlabel('True Positive Rate')
plt.title('ROC Curve - Train')
plt.show()


#show the precision & recall
pr = lrn.summary.pr.toPandas()
plt.plot(pr['recall'],pr['precision'])
plt.ylabel('Precision')
plt.xlabel('Recall')
plt.title('Precision/Recall - Train')
plt.show()


#give me the important features for this model

from pyspark.sql.window import Window
from pyspark.sql.functions import row_number,lit
#show me the weights of all the features
weights = lrn.coefficients
weights = [(float(w),) for w in weights]
weightsDF = sqlContext.createDataFrame(weights,['Feature Weight'])
#create an index to join the feature names to
w = Window.orderBy(lit('A'))
weightsDF = weightsDF.withColumn("index",row_number().over(w))

#pull the feature names & merge them together & create a dataframe
numeric_metadata = df4.select("vectorized_features").schema[0].metadata.get('ml_attr').get('attrs').get('numeric')
binary_metadata = df4.select("vectorized_features").schema[0].metadata.get('ml_attr').get('attrs').get('binary')
merge_list = numeric_metadata + binary_metadata 
feature_df = spark.createDataFrame(merge_list)
#create index for this dataframe
w = Window.orderBy(lit('A'))
feature_df = feature_df.withColumn("index",row_number().over(w))

#join these two dataframes together
important_features = feature_df.join(weightsDF,feature_df.index==weightsDF.index)
#only show features that do not have a weight of 0
important_features.filter(col('Feature Weight')!=0).show()


#trying out a decision tree
#not using the standardized features because this is a tree-based model, also makes it easier to understand
from pyspark.ml.classification import DecisionTreeClassifier

train, test = df4.randomSplit([0.7, 0.3],seed=2022)

dt = DecisionTreeClassifier(featuresCol = 'vectorized_features', labelCol = 'label', maxDepth = 5)
dtModel = dt.fit(train)
predictions = dtModel.transform(test)

print("Test Area Under ROC: " + str(evaluator.evaluate(predictions, {evaluator.metricName: "areaUnderROC"})))
print("Test Area Under PR: " + str(evaluator.evaluate(predictions, {evaluator.metricName: "areaUnderPR"})))
#played around with several different max depths, 5 seemed to be the best, but the ROC is still poor

#random forest
#AUC for base random forest performs well vs. the decision tree, but still worse than logistic regression
from pyspark.ml.classification import RandomForestClassifier
rf = RandomForestClassifier(featuresCol = 'vectorized_features', labelCol = 'label')
rfModel = rf.fit(train)
predictions = rfModel.transform(test)
evaluator = BinaryClassificationEvaluator()
print("Test Area Under ROC: " + str(evaluator.evaluate(predictions, {evaluator.metricName: "areaUnderROC"})))
print("Test Area Under PR: " + str(evaluator.evaluate(predictions, {evaluator.metricName: "areaUnderPR"})))


#gradient boosting
#performs the best with AUC of .9459
from pyspark.ml.classification import GBTClassifier
gbt = GBTClassifier(maxIter=10,featuresCol = 'vectorized_features', labelCol = 'label')
gbtModel = gbt.fit(train)
predictions = gbtModel.transform(test)

evaluator = BinaryClassificationEvaluator()
print("Test Area Under ROC: " + str(evaluator.evaluate(predictions, {evaluator.metricName: "areaUnderROC"})))
print("Test Area Under PR: " + str(evaluator.evaluate(predictions, {evaluator.metricName: "areaUnderPR"})))


#tried out a neural network, cannot calculate auc though, area under precision/recall is also lackluster
#we will not be going with this model
from pyspark.ml.classification import MultilayerPerceptronClassifier
from pyspark.ml.evaluation import MulticlassClassificationEvaluator

# specify layers for the neural network:
# input layer of size 53 (features)
# and output of size 2 (classes)
layers = [53, 5, 4, 2]

# create the trainer and set its parameters
trainer = MultilayerPerceptronClassifier(maxIter=100, layers=layers, blockSize=128, seed=1234,featuresCol = 'vectorized_features', labelCol = 'label')
# train the model
mlp = trainer.fit(train)

# compute accuracy on the test set
predictions = mlp.transform(test)
evaluator = BinaryClassificationEvaluator()
print("Test Area Under ROC: " + str(evaluator.evaluate(predictions, {evaluator.metricName: "areaUnderROC"})))
print("Test Area Under PR: " + str(evaluator.evaluate(predictions, {evaluator.metricName: "areaUnderPR"})))


#compare the AUC of all of our models together
from pyspark.ml import evaluation 
models = [(lrn, "logistic regression"), 
          (dtModel, "decision tree"), 
          (rfModel, "random forest"), 
          (gbtModel, "gradient boosting"), 
          (mlp, "multilayer perceptron")]

evaluator = evaluation.BinaryClassificationEvaluator()

for model, name in models:
    print(f"AUC of {name}: {evaluator.evaluate(model.transform(test))}")

#changing this cell to markdown as it takes a very long time to run, parameters outputted from this have been
#used in the gradient boosting model below
#lets select the gradient boosting model and move forward with some parameter tuning to make it more accurate
from pyspark.ml.tuning import ParamGridBuilder, CrossValidator

paramGrid = (ParamGridBuilder()
             .addGrid(gbt.maxDepth,[3,5,8])
             .addGrid(gbt.maxBins,[10,15,20])
             .addGrid(gbt.maxIter,[10,20])
             .build())
cv = CrossValidator(estimator=gbt,estimatorParamMaps=paramGrid,
                   evaluator=evaluator,numFolds=5)

cvModel = cv.fit(train)
predictions = cvModel.transform(test)
evaluator.evaluate(predictions)

#show me the parameters we actually ended up using for the "best model" from the parameter tuning
#these are used in the below model, 20 max iterations, 20 max bins, 5 max depth
bestModel = cvModel.bestModel
best_model_depth = bestModel._java_obj.getMaxDepth()
best_model_bins = bestModel._java_obj.getMaxBins()
best_model_iterations = bestModel._java_obj.getMaxIter()
print(best_model_depth)
print(best_model_bins)
print(best_model_iterations)

#gradient boosting with the parameters outputted from the parameter tuned model
from pyspark.ml.classification import GBTClassifier
gbt = GBTClassifier(maxIter=20,maxBins=20,maxDepth=5,featuresCol = 'vectorized_features', labelCol = 'label')
gbtModel = gbt.fit(train)
predictions = gbtModel.transform(test)

evaluator = BinaryClassificationEvaluator()
print("Test Area Under ROC: " + str(evaluator.evaluate(predictions, {evaluator.metricName: "areaUnderROC"})))
print("Test Area Under PR: " + str(evaluator.evaluate(predictions, {evaluator.metricName: "areaUnderPR"})))

a = gbtModel.featureImportances
important_features = pd.DataFrame(a.toArray())
important_features = important_features.reset_index()
print(important_features)


#give a new index column
feature_df = feature_df.withColumn("index_new",row_number().over(w)-1)
#convert to pandas dataframe
feature_names = feature_df.toPandas()
#merge with previous feature importances
combined_important_features = pd.merge(important_features,feature_names,how='inner',left_on='index',right_on='idx')
#rename columns
combined_important_features = combined_important_features.rename(columns={combined_important_features.columns[1]:'feature_importance'})
#sort dataframe
combined_important_features = combined_important_features.sort_values(by='feature_importance',ascending=False)
#show the top 10 most important features
print(combined_important_features.head(20))


#make a plot that shows the 10 most important features
import seaborn as sns
import matplotlib.pyplot as plt
sns.barplot(x = 'feature_importance',y = 'name',data = combined_important_features.head(10))
plt.xlabel('Feature Name')
plt.ylabel('Feature Importance')
plt.title("Important Features in Tuned Gradient Boosted Model")
plt.show()

##my prescriptive thoughts would be that we need to tell bankers to try and keep the potential customers engaged by having longer conversations with them, ensure that the bank is staffed appropriately (number_employees) to ensure that when customers do come into the bank, they are able to be helped. The other two important factors do seem to be out of scope for the bank, although they may be able to just ensure they have good rates (euribor3m) and forecast demand based upon the cons_confidence_index

#create a pickle for this model
gbtModel.save('C:\\Users\\divin\\')




