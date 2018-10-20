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
    for df in df_list:

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
        
        if (n_clusters_ == 0):
            centre = np.zeros((1 , 3))
            dist_to_cluster = np.zeros(X.shape)
            centre[0][0] = np.sum(X[:,0])/len(X)
            centre[0][1] = np.sum(X[:,1])/len(X)
            centre[0][2] = np.sum(X[:,2])/len(X)
            X_dist = np.sum(np.square(X-centre), axis=1)
            
            sorted_index = np.argsort(X_dist)[::-1]
        
        else:
            clusters_centre = np.zeros((len(unique_labels) - 1 , 3))
            
            dist_to_cluster = np.zeros((X.shape[0], clusters_centre.shape[0]))
            for i in range(len(unique_labels) - 1):
                X_sameLabels = X[np.where(labels==i)]
                clusters_centre[i][0] = np.sum(X_sameLabels[:,0])/len(X_sameLabels)
                clusters_centre[i][1] = np.sum(X_sameLabels[:,1])/len(X_sameLabels)
                clusters_centre[i][2] = np.sum(X_sameLabels[:,2])/len(X_sameLabels)
                dist_to_cluster[:, i] = np.sum(np.square(X-clusters_centre[i]), axis=1)
            
            
            X_dist = np.min(dist_to_cluster, axis=1)
            sorted_index = np.argsort(X_dist)[::-1] #first is the most distance = highest suspicion
        
        # 'patientFamilyID','paitientFamilyMemberID','providerID','providerType','dateOfService', 'rank'
        
        if len(sorted_index)<100:
            selected_entries = df.values[sorted_index[:len(sorted_index)]]
            output[output_index:output_index+len(selected_entries), 0] = selected_entries[:, 0]
            output[output_index:output_index+len(selected_entries), 1] = selected_entries[:, 1]
            output[output_index:output_index+len(selected_entries), 2] = selected_entries[:, 2]
            output[output_index:output_index+len(selected_entries), 3] = selected_entries[:, 5]
            output[output_index:output_index+len(selected_entries), 4] = selected_entries[:, 3]
            output[output_index:output_index+len(selected_entries), 5] = np.arange(1, 1+ len(selected_entries))
            output_index += len(selected_entries)
        
        else:
            selected_entries = df.values[sorted_index[:100]]
            output[output_index:output_index+100, 0] = selected_entries[:, 0]
            output[output_index:output_index+100, 1] = selected_entries[:, 1]
            output[output_index:output_index+100, 2] = selected_entries[:, 2]
            output[output_index:output_index+100, 3] = selected_entries[:, 5]
            output[output_index:output_index+100, 4] = selected_entries[:, 3]
            output[output_index:output_index+100, 5] = np.arange(1, 101)
        
            output_index += 100
    #end loop
    
    output = output[output[:, 5].argsort()] #sort into ranks
    output = output[~np.all(output == 0, axis=1)] #remove 0 rows
    np.savetxt('file2.csv', output, delimiter=',')


    