#!/usr/bin/env python
# coding: utf-8

# # Imports and Supportive Functions

# In[1]:


#Load Libraries
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
get_ipython().run_line_magic('matplotlib', 'inline')
import seaborn as sns
import warnings
warnings.filterwarnings('ignore')


# In[2]:


# function to call roc curve
from sklearn.metrics import roc_curve, roc_auc_score

def plot_roc_curve(y_true, y_prob, title='ROC Curve'):
    fpr, tpr, thresholds = roc_curve(y_true, y_prob)
    roc_auc = roc_auc_score(y_true, y_prob)
    
    plt.figure(figsize=(8, 6))
    plt.plot(fpr, tpr, color='darkorange', lw=2, label='ROC curve (AUC = %0.2f)' % roc_auc)
    plt.plot([0, 1], [0, 1], color='navy', lw=2, linestyle='--')
    plt.xlim([0.0, 1.0])
    plt.ylim([0.0, 1.05])
    plt.xlabel('False Positive Rate')
    plt.ylabel('True Positive Rate')
    plt.title(title)
    plt.legend(loc="lower right")
    plt.show()


# In[3]:


#Load Dataset
df=pd.read_csv('./water_potability.csv')
x = df.drop('Potability', axis=1).to_numpy()
y = df['Potability'].to_numpy()


# In[4]:


df.describe()


# In[5]:


#Create Profile Report
 
#Importing package
import pandas_profiling as pp
from IPython.display import IFrame

# Perform Exploratory Data Analysis using pandas-profiling
water_potabilityReport = pp.ProfileReport(df)
water_potabilityReport.to_file('water_potabilityReport.html')
display(IFrame('WaterPotablityReportFP-lbfgs.html', width=900, height=350))


# In[6]:


from sklearn.ensemble import IsolationForest

# The prediction returns 1 if sample point is inlier. If outlier prediction returns -1
clf_all_features = IsolationForest(random_state=100)
clf_all_features.fit(x)

#Predict if a particular sample is an outlier using all features for higher dimensional data set.
y_pred = clf_all_features.predict(x)
y_pred2 =np.array(list(map(lambda x: x == 1, y_pred)))

# Exclude suggested outlier samples for improvement of prediction power/score
x_mod = x[y_pred2, ]
y_mod = y[y_pred2, ]

#Size of Datasets
print('Original Train Dataset Size : {}'.format(len(x)))
print('New Train Dataset Size      : {}'.format(len(x_mod)))


# In[7]:


# Create Train and Test Datasets
from sklearn.model_selection import train_test_split
x_train, x_test, y_train, y_test = train_test_split(x_mod, y_mod, test_size=0.20,stratify=y_mod,random_state=100)

#Scale the Data
from sklearn.preprocessing import StandardScaler
sc = StandardScaler()
x_train2 = sc.fit_transform(x_train)
x_test2 = sc.transform(x_test)

x_2 = sc.transform(x)


# In[8]:


#Construct some pipelines 
from sklearn.linear_model import LogisticRegression
from sklearn.naive_bayes import GaussianNB
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler

#Create Pipeline

pipelines =[]

pipe_logreg = Pipeline([('scl', StandardScaler()),
                    ('clf', LogisticRegression(solver='lbfgs',class_weight='balanced',
                                               random_state=100))])
pipelines.insert(0,pipe_logreg)
# Create Pipeline for Gaussian Naive Bayes
pipe_gnb = Pipeline([
    ('scl', StandardScaler()),
    ('clf', GaussianNB())
])
pipelines.insert(1, pipe_gnb) 
#Set grid search params 

modelparas =[]

param_gridlogreg = {'clf__C': [0.1, 1, 5, 20], 
                    'clf__penalty': ['l1','none'],'clf__max_iter': [90,150,2000]}
modelparas.insert(0,param_gridlogreg)
modelparas.insert(1,{})


# In[9]:


#Define Plot for learning curve

from sklearn.model_selection import learning_curve

def plot_learning_curves(model):
    train_sizes, train_scores, test_scores = learning_curve(estimator=model,
                                                            X=x_train2, 
                                                            y=y_train,
                                                            train_sizes= np.linspace(0.1, 1.0, 10),
                                                            cv=10,
                                                            scoring='recall_weighted',random_state=100)
    train_mean = np.mean(train_scores, axis=1)
    train_std = np.std(train_scores, axis=1)
    test_mean = np.mean(test_scores, axis=1)
    test_std = np.std(test_scores, axis=1)
    
    plt.plot(train_sizes, train_mean,color='blue', marker='o', 
             markersize=5, label='training recall')
    plt.fill_between(train_sizes, train_mean + train_std, train_mean - train_std,
                     alpha=0.15, color='blue')

    plt.plot(train_sizes, test_mean, color='green', linestyle='--', marker='s', markersize=5,
             label='validation recall')
    plt.fill_between(train_sizes, test_mean + test_std, test_mean - test_std,
                     alpha=0.15, color='green')
    plt.grid(True)
    plt.xlabel('Number of training samples')
    plt.ylabel('Recall')
    plt.legend(loc='best')
    plt.ylim([0.5, 1.01])
    plt.show()


# In[10]:


#Plot Learning Curve
print('Logistical Regression - Learning Curve')
plot_learning_curves(pipe_logreg)
print('GNB Learning Curve')
plot_learning_curves(pipe_gnb)


# In[11]:


#Define Gridsearch Function

from sklearn.model_selection import GridSearchCV
from sklearn.metrics import classification_report, confusion_matrix
from sklearn.model_selection import RepeatedKFold

def Gridsearch_cv(model, params):
    
    #Cross-validation Function
    cv2=RepeatedKFold(n_splits=10, n_repeats=5, random_state=100)
        
    #GridSearch CV
    gs_clf = GridSearchCV(model, params, cv=cv2,scoring='recall_weighted')
    gs_clf = gs_clf.fit(x_train2, y_train)
    model = gs_clf.best_estimator_
    
    # Use best model and test data for final evaluation
    y_pred = model.predict(x_test2)

    #Identify Best Parameters to Optimize the Model
    bestpara=str(gs_clf.best_params_)
    
    #Output Validation Statistics
    target_names=['0','1']
    print('\nOptimized Model')
    print('\nModel Name:',str(pipeline.named_steps['clf']))
    print('\nBest Parameters:',bestpara)
    print('\n', confusion_matrix(y_test,y_pred))  
    print('\n',classification_report(y_test,y_pred,target_names=target_names)) 
        
    #Transform the variables into binary (0,1) - ROC Curve
    from sklearn import preprocessing
    Forecast1=pd.DataFrame(y_pred)
    Outcome1=pd.DataFrame(y_test)
    lb1 = preprocessing.LabelBinarizer()
    OutcomeB1 =lb1.fit_transform(Outcome1)
    ForecastB1 = lb1.fit_transform(Forecast1)
    
    #Setup the ROC Curve
    from sklearn.metrics import roc_curve, auc
    from sklearn import metrics
    fpr, tpr, threshold = metrics.roc_curve(OutcomeB1, ForecastB1)
    roc_auc = metrics.auc(fpr, tpr)
    print('ROC Curve')
    #Plot the ROC Curve
    plt.title('Receiver Operating Characteristic')
    plt.plot(fpr, tpr, 'b', label = 'AUC = %0.2f' % roc_auc)
    plt.legend(loc = 'lower right')
    plt.plot([0, 1], [0, 1],'r--')
    plt.xlim([0, 1])
    plt.ylim([0, 1])
    plt.ylabel('True Positive Rate')
    plt.xlabel('False Positive Rate')
    plt.show() 


# In[12]:


# Loop through the pipelines and model parameters
for pipeline, modelpara in zip(pipelines, modelparas):
    Gridsearch_cv(pipeline, modelpara)


# In[13]:


# Make Ensamble MOdels
from sklearn.ensemble import AdaBoostClassifier, RandomForestClassifier
from xgboost import XGBClassifier
ensemble_models = []
para = []
# Create pipelines for different models
pipeline_adaboost = Pipeline([
    ('scl', StandardScaler()),
    ('clf', AdaBoostClassifier(random_state=100))
])
ensemble_models.insert(0,pipeline_adaboost)

pipeline_randomforest = Pipeline([
    ('scl', StandardScaler()),
    ('clf', RandomForestClassifier(random_state=100))
])
ensemble_models.insert(1,pipeline_randomforest)

pipeline_xgboost = Pipeline([
    ('scl', StandardScaler()),
    ('clf', XGBClassifier(random_state=100))
])
ensemble_models.insert(2,pipeline_xgboost)
param_grid_adaboost = {
    'clf__n_estimators': [50, 100, 150],
    'clf__learning_rate': [0.01, 0.1, 1.0]
}
para.insert(0,param_grid_adaboost)

param_grid_randomforest = {
    'clf__n_estimators': [50, 100, 150],
    'clf__max_depth': [None, 10, 20]
}
para.insert(1,param_grid_randomforest)

param_grid_xgboost = {
    'clf__n_estimators': [50, 100, 150],
    'clf__max_depth': [3, 4, 5],
    'clf__learning_rate': [0.01, 0.1, 0.2]
}
para.insert(2,param_grid_xgboost)


# In[14]:


for pipeline, modelpara in zip(ensemble_models, para):
    Gridsearch_cv(pipeline, modelpara)


# In[15]:


#voting model
nb = GaussianNB()
rf = RandomForestClassifier(max_depth= 5, n_estimators = 150, random_state=100)
from sklearn.ensemble import VotingClassifier
# Create the voting classifier
voting_classifier = VotingClassifier(estimators=[('rf', rf), ('nb', nb)], voting='soft')

# Train the voting classifier
voting_classifier.fit(x_train2, y_train)

# Make predictions
y_pred = voting_classifier.predict(x_test2)

# Calculate accuracy
rpt = classification_report(y_test, y_pred)
print(rpt)
plot_roc_curve(y_test, y_pred, title='ROC Curve')


# In[17]:


base_models = [
    GaussianNB(), RandomForestClassifier(max_depth= 5, n_estimators = 150, random_state=100),XGBClassifier(random_state=100)
]

predictions = np.zeros((x_test2.shape[0], len(base_models)))
meta_model = LogisticRegression()

# Create an array to hold the predictions of base models on the test set
predictions = np.zeros((x_test.shape[0], len(base_models)))

# Train the base models and make predictions on the test set
for i, model in enumerate(base_models):
    model.fit(x_train2, y_train)
    y_pred = model.predict(x_test2)
    predictions[:, i] = y_pred

# Train the meta-model on the predictions of base models
meta_model.fit(predictions, y_test)

# Make final predictions using the stacking model
final_predictions = meta_model.predict(predictions)

# Evaluate the stacking model
print(classification_report(y_test, final_predictions))
plot_roc_curve(y_test,final_predictions,title='ROC Curve')


# # Random forest is giving best accuracy with better prediction rate.

# In[ ]:




