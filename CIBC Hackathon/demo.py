import pandas as pd
import matplotlib.pyplot as plt
from sklearn.cluster import DBSCAN
import numpy as np
from mpl_toolkits.mplot3d import Axes3D

#Set up feature vectors
def generate_feature_1(df):
    df['norm_dollar'] = df.groupby('medicalProcedureCode', group_keys=False).apply(lambda g: (g['dollarAmountClaim'] - g['dollarAmountClaim'].mean()) / g['dollarAmountClaim'].std())
    df=df.fillna(0)

    return df['norm_dollar']

#value_pm_normed is a vector of len(visits)

#generate_feature_1(df_list[0]) 

def generate_feature_2(df):
    # Average money spend for all previous visit in this provider from this patient
    #df['provider_medical_concat'] = df['providerType'].astype(str).str.cat(df['medicalProcedureCode'].astype(str))
    #df['patient']= df['patientFamilyID'].astype(str) + "|" + df['paitientFamilyMemberID'].astype(str) + "|" + df['providerID'].astype(str) 
    #df_avg_dollar = df.groupby(['patientFamilyID','paitientFamilyMemberID','providerID']).agg({'dollarAmountClaim':'mean'})
    df = df[~df['dollarAmountClaim'].isnull()]
    df['patient'] = df['patientFamilyID'].astype(str) + "|" + df['patientFamilyMemberID'].astype(str) + "|" + df['providerID'].astype(str) 
    temp = df.groupby(['patient']).agg({'dollarAmountClaim':'mean'}).reset_index()
    df = pd.merge(df, temp, on='patient')
    
    #df['average spent']=df_avg_dollar[]
    #df_avg_dollar = df.groupby('patient').agg({'dollarAmountClaim':'mean'})
    #
    return (df.dollarAmountClaim_y-df.dollarAmountClaim_y.mean())/df.dollarAmountClaim_y.std()

def generate_feature_3(df):
    df['providerID_medical'] = df['providerID'].astype(str) + df['medicalProcedureCode'].astype(str)
    df['patientID_medical'] = df['patientFamilyID'].astype(str) + (df['patientFamilyMemberID'].astype(str)) + (df['medicalProcedureCode'].astype(str))

    df['provider_count'] = df.groupby('providerID_medical')['providerID_medical'].transform('count')
    df['patient_count'] = df.groupby('patientID_medical')['patientID_medical'].transform('count')
    
    df['provider_normed'] = df.groupby('medicalProcedureCode', group_keys=False)\
    .apply(lambda g: (g['provider_count'] - g['provider_count'].mean()) / g['provider_count'].std())
    df['patient_normed'] = df.groupby('medicalProcedureCode', group_keys=False)\
    .apply(lambda g: (g['patient_count'] - g['patient_count'].mean()) / g['patient_count'].std())
    df = df.fillna(1)
    df['final'] = df['provider_normed'] * df['patient_normed']
    return df['final']
 
    

#generate_feature_2(df_list[0])

if __name__ =="__main__":
    columns = ['patientFamilyID',
            'patientFamilyMemberID',
            'providerID',
            'providerType',
            'stateCode',
            'dateOfService',
            'medicalProcedureCode',
            'dollarAmountClaim']
    df = pd.read_csv('~/desktop/cibc hackathon/claims_final.csv', names = columns)
    
    gb = df.groupby('providerType')
    
    df_list = df['providerType'].unique()
    
    df_list = [df[df['providerType']==i] for i in df_list]
    print(df_list[0].shape[0])
    
    output = np.zeros((1300, 6), dtype=object)
    providertype = df
    
    output_index = 0
   
    df = df_list[0]
    series_f1 = generate_feature_1(df).values
    series_f2 = generate_feature_2(df).values
    series_f3 = generate_feature_3(df).values
    
    X = np.zeros((len(series_f1), 3))
    X[:,0] = series_f1
    X[:,1] = series_f2
    X[:,2] = series_f3
    
    
    ### Parameters ###
    
    
    
    db = DBSCAN(eps=0.85, min_samples=5).fit(X)
    
    
    ##################
    core_samples_mask = np.zeros_like(db.labels_, dtype=bool)
    core_samples_mask[db.core_sample_indices_] = True
    labels = db.labels_
    n_clusters_ = len(set(labels)) - (1 if -1 in labels else 0)
    
    # Black removed and is used for noise instead.
    unique_labels = set(labels)

    
    # 'patientFamilyID','paitientFamilyMemberID','providerID','providerType','dateOfService', 'rank'


    
    colors = [plt.cm.Spectral(each)
            for each in np.linspace(0, 1, len(unique_labels))]
            
    fig = plt.figure()
    ax = fig.add_subplot(111, projection='3d')
    for k, col in zip(unique_labels, colors):
        if k == -1:
            # Black used for noise.
            col = [0, 0, 0, 1]
    
        class_member_mask = (labels == k)
    
        xy = X[class_member_mask & core_samples_mask]
        
        ax.plot(xy[:, 0], xy[:, 1], xy[:, 2], 'o', markerfacecolor=tuple(col),
                markeredgecolor='k', markersize=14)
    
        xy = X[class_member_mask & ~core_samples_mask]
        ax.plot(xy[:, 0], xy[:, 1], xy[:, 2],'o', markerfacecolor=tuple(col),
                markeredgecolor='k', markersize=6)
    
    ax.set_xlabel('Feature 1')
    ax.set_ylabel('Feature 2')
    ax.set_zlabel('Feature 3')
    ax.text2D(0.85, 0.05, 'Provider Type 122399961', transform=ax.transAxes)
    plt.title('Estimated number of clusters: %d' % n_clusters_)
    plt.show()
            


    