import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
import glob
import pickle

# Define methods to transform single csv file to wider form
def data_transform(path):
    # Read csv file
    df = pd.read_csv(path)
    df.columns = ['M/Z', 'Intensity']
    
    # Select mass window between 200 and 10000
    df = df[(df['M/Z']>200) & (df['M/Z']<10000)]
    
    # Baseline substration. Substract the spectra with the median of lowest 20% intensity
    median = np.median(np.sort(df.Intensity)[:(len(df.Intensity)//5)])
    df.Intensity = df.Intensity.apply(lambda x: x-median if (x-median)>=0 else 0)
    
    # Rescale by dividing the intensity with the median of top 5% intensity and take the rootsqure
    top_median = np.median(np.sort(df.Intensity)[::-1][:(int(len(df.Intensity)*0.05))])
    df.Intensity = df.Intensity.apply(lambda x: x/top_median)
    df.Intensity = np.sqrt(df.Intensity)
    
    # Pivot table and return the resulting data frame
    df_transform= df.pivot_table(columns='M/Z')
    df_transform = df_transform.reset_index().iloc[:,1:]
    
    return df_transform

# Define methods to transform and concat all dataframe to one single dataframe
def transform_combine_spectra(folder):    
    csv_list = glob.glob('/Users/qiancai/Downloads/2019NSOpenDataContest/JNCI_Data_7-3-02/'+ folder + '*', recursive=True)
    df = pd.DataFrame()
    for csv_file in csv_list:
        temp = data_transform(csv_file)
        df_update = pd.concat((df, temp), ignore_index=True)
        if np.any(df.isnull()):
            print('There is mismatch of data points!')
            break
        else:
            df = df_update
    return df

# Spectra data for cancer group
cancer_group_prostate = transform_combine_spectra('Cancer/')
# Spectra data for non cancer group
control_group_prostate = transform_combine_spectra('Control/')
# Store into pickle file
with open('pickle/prostate.pickle','wb') as f:
    pickle.dump([cancer_group_prostate, control_group_prostate], f)

# Define method of creating data and labels
def create_data_label(cancer_group, control_group):
    data = pd.concat((cancer_group, control_group), ignore_index=True)
    label = np.concatenate((np.repeat(1, cancer_group.shape[0]), np.repeat(-1, control_group.shape[0])))
    return data, label

# Define method of plotting spectra heatmap
def heatmap(cancer_group, control_group):
    plt.figure(figsize=[20,10])

    # Cancer group spectra
    plt.subplot(211)
    plt.title('Cancer group', fontsize = 20)

    plt.imshow(cancer_group, aspect='auto')
    plt.xticks(np.arange(0, 9200, 1000), cancer_group.columns[0:9200:1000])
    plt.yticks(np.arange(0,len(cancer_group)+1, 10))
    plt.xlabel('M/Z')
    plt.ylabel('Samples')
    plt.colorbar()

    # Non-cancer group spectra
    plt.subplot(2,1,2)
    plt.title('Control group', fontsize = 20)
    plt.imshow(control_group, aspect='auto')
    plt.xticks(np.arange(0, 9200, 1000), control_group.columns[0:9200:1000])
    plt.yticks(np.arange(0,len(control_group),10))
    plt.xlabel('M/Z')

    plt.ylabel('Samples')
    plt.colorbar()

    plt.tight_layout()
    plt.savefig('Figures/Sample comparison.png')
    
# Create data and labels for robotic prepared samples, hand prepared ovarian samples and prostate samples    
prostate_data, prostate_label = create_data_label(cancer_group_prostate, control_group_prostate)    



print('Prostate Samples')
print('='*100)
heatmap(cancer_group_prostate, control_group_prostate)

from sklearn.decomposition import PCA
pca = PCA()
pca_hand = PCA()
pca_prostate = PCA()

# Transform data and save to pickle files
prostate_data_transform = pca_prostate.fit_transform(prostate_data)
with open('pickle/pca_prostate.pickle', 'wb') as f:
    pickle.dump(pca_prostate, f)
with open('pickle/prostate_transformed_x_y.pickle', 'wb') as f:
    pickle.dump([prostate_data_transform, prostate_label], f)

# Show the cumulative explained variance
plt.figure(figsize=[5,5])
plt.plot(np.cumsum(pca_prostate.explained_variance_ratio_)[:100], 'g')
plt.plot([0,100], [0.95, 0.95], color = 'r', linewidth = 3, alpha = 0.5)
plt.annotate('95% Variance Explained', xy=(0,1), xytext=(0,1), color = 'r')
plt.legend(loc = 'best')
plt.xlabel('Number of Components')
plt.ylabel('Explained Variance')    

# Scatter plot of PC1 and PC2
plt.title('Prostate Samples')
plt.scatter(prostate_data_transform[:,0], prostate_data_transform[:,1], c=prostate_label)
plt.plot([-10,20],[-8, 15],'r--',linewidth = 3)
plt.xlabel('PC1')
plt.colorbar()

plt.tight_layout()
plt.savefig('Comparison of Samples_pca.png')



# Define methods for generate performance report of model
def report(model, y, predict):
    print('Report of ' + model)
    print('=================================================================================')
    print('Accuracy of the model:{}'.format(accuracy_score(y, predict)))
    print('AUC score:            {}'.format(roc_auc_score(y, predict)))
    print('F1 score:             {}'.format(f1_score(y, predict)))
    print('Confusion Matrix:')
    print(confusion_matrix(y, predict))
    
# Import random forest from sklearn.ensemble
from sklearn.ensemble import RandomForestClassifier, VotingClassifier

# Select number of features (from high importance to low importance) with total importance more than 95%
def feature_number(feature_importance):
    feature_importance_sort = np.sort(feature_importance)[::-1]
    sums = 0
    for i, j in enumerate(feature_importance_sort):
        sums += j
        if sums>0.95:
            return(i+1)

# Fit the model using Random Forest
rf_prostate = RandomForestClassifier(random_state=5)
rf_prostate.fit(prostate_data, prostate_label)

# Plot the cumulative explained variance of features
feature_importance_prostate = rf_prostate.feature_importances_

plt.figure(figsize=(5,5))
plt.plot(np.cumsum(np.sort(feature_importance_prostate)[::-1][:100]), 'g')

plt.plot([0,100], [0.95, 0.95], color = 'r', linewidth = 3, alpha = 0.5)
plt.annotate('95% Variance Explained', xy=(0,1), xytext=(0,1), color = 'r')
plt.xlabel('Number of Features')
plt.ylabel('Explained Variance')
plt.legend(loc='best')

# Select indexes of top features explaining 95% of variance
important_feature_index_prostate = np.argsort(feature_importance_prostate)[::-1][:(feature_number(feature_importance_prostate))]

# Write important feature index to pickle files
for i,j in zip(
    ['pickle/feature_index_prostate.pickle'],
    [important_feature_index_prostate]):    
    
    with open(i, 'wb') as f:
        pickle.dump(j, f)
        
# Split the data to training and testing data
from sklearn.model_selection import train_test_split

x_train_prostate, x_test_prostate, y_train_prostate, y_test_prostate = train_test_split(prostate_data.iloc[:,important_feature_index_prostate], prostate_label, test_size = 0.3, random_state = 10)

# Import necessary sklearn package for ML
from sklearn.svm import SVC
from sklearn.metrics import accuracy_score, classification_report,confusion_matrix, roc_auc_score, f1_score, roc_curve
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import make_pipeline
from sklearn.model_selection import GridSearchCV
#from pandas_ml import ConfusionMatrix 

# make pipeline using standard scale and svc
pipeline=make_pipeline(StandardScaler(), SVC(probability=True))
param_grid = {'svc__C': 10.**np.arange(-3,3), 'svc__gamma': 10.**np.arange(-3,3)}

# Create gridsearch object to tune C and gammas values
gs = GridSearchCV(pipeline, param_grid=param_grid)

# Prostate data fit and predict
gs.fit(x_train_prostate, y_train_prostate)
clf_svm_prostate = gs.best_estimator_

with open('pickle/svc_prostate.pickle', 'wb') as f:
    pickle.dump(gs.best_estimator_, f)
    
svc_predict_prostate = gs.predict(x_test_prostate)

# Generate performance report
report('SVM with selected features for prostate prediction', y_test_prostate, svc_predict_prostate)

# Tune parameters for random forest, using n_estimators and max_features to tune
param_grid = {'n_estimators': 10**np.arange(3), 
              'max_features': ['auto','sqrt','log2']}
gs = GridSearchCV(RandomForestClassifier(), param_grid=param_grid)
# Fit and predict prostate data
gs.fit(x_train_prostate, y_train_prostate)
clf_rf_prostate = gs.best_estimator_

with open('pickle/rf_prostate.pickle', 'wb') as f:
    pickle.dump(gs.best_estimator_, f)
    
rf_predict_prostate = gs.predict(x_test_prostate)

report('Random Forest with selected features for prostate prediction', y_test_prostate, rf_predict_prostate)

# Tune parameters for random forest, using n_neighbors to tune
from sklearn.neighbors import KNeighborsClassifier
param_grid_knn = {'n_neighbors':np.arange(1,30)}
gs = GridSearchCV(KNeighborsClassifier(), param_grid=param_grid_knn)

# Fit and predict prostate
gs.fit(x_train_prostate, y_train_prostate)
clf_knn_prostate = gs.best_estimator_

with open('pickle/knn_prostate.pickle', 'wb') as f:
    pickle.dump(gs.best_estimator_, f)
    
knn_predict_prostate = gs.predict(x_test_prostate)
knn_predict_proba_prostate = gs.predict_proba(x_test_prostate)

# Generate knn performance report for ovarian and prostate data
report('knn prostate', y_test_prostate, knn_predict_prostate)

comparison = pd.DataFrame({'SVM':[accuracy_score(y_test_prostate, svc_predict_prostate), 
                                  roc_auc_score(y_test_prostate, svc_predict_prostate),
                                  f1_score(y_test_prostate, svc_predict_prostate)], 
                           'Random Forest':[accuracy_score(y_test_prostate, svc_predict_prostate), 
                                            roc_auc_score(y_test_prostate, svc_predict_prostate),
                                            f1_score(y_test_prostate, svc_predict_prostate)], 
                           'KNN':[accuracy_score(y_test_prostate, knn_predict_prostate),
                                  roc_auc_score(y_test_prostate, knn_predict_prostate),
                                  f1_score(y_test_prostate, knn_predict_prostate)],
                           'Score': ['Accuracy', 'AUC', 'F1-Score'],
                           'Data':  np.repeat(['Prostate'],3)}).set_index(['Data','Score'])
print(comparison)

# Plot the histogram for probability distribution of cancer predicting
plt.figure(figsize=(10,5))
plt.subplot(1,2,1)
plt.hist(svc_predict_proba_ovarian[:,1], label='SVM')
plt.ylabel('Frequency')
plt.xlabel('Probability of predicting as cancer')
plt.legend(loc = 'best')

# Plot the histogram for probability distribution of predicting female
plt.subplot(1,2,2)
plt.hist(rf_predict_proba_ovarian[:,1], label='Random Forest', color = 'g')
plt.xlabel('Probability of predicting as cancer')
plt.yticks([])
plt.legend(loc = 'best')

plt.tight_layout()


