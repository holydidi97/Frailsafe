import csv
import numpy as np
from numpy import NaN, empty, float64, int64, zeros
from numpy.lib.arraysetops import isin
import pandas as pd
import re
from sklearn.model_selection import train_test_split
from sklearn.cluster import KMeans
from sklearn.metrics import silhouette_samples, silhouette_score
import matplotlib.pyplot as plt 
from sklearn.neighbors import KNeighborsClassifier 
import seaborn as sns


#import data set 
# Clinical Dataset 
df = pd.read_csv('clinical_dataset.csv',delimiter=";")   
print(df.columns)
print(df.describe())
print(df.median())


df['fried'] = df['fried'].replace(['Frail','Pre-frail','Non frail'],[2,1,0])
df = df.replace(['Yes','No'],[1,0])
df = df.replace([999.0,999],[NaN,NaN])


#dictionary for unique nominal features
col = []
for i in df.columns:
    col.append(i)


temp = {}
for i in col:
    t = []
    for k in df[i].unique():
        if (type(k) is not float64 and type(k) is not int64 and type(k) is not int or k==999):
            t.append(k)
    if t:
        temp[i] = t
      
print(t)

#convert nominal features to numerical and replace erroneous values with NaN
df['gender'] = df['gender'].replace(['F','M'],[1,0])     
df['vision'] = df['vision'].replace(['Sees moderately','Sees well','Sees poorly'],[2,1,0])
df['audition'] = df['audition'].replace(['Hears well','Hears moderately','Hears poorly'],[2,1,0])
df['balance_single'] = df['balance_single'].replace(['>5 sec','<5 sec','test non realizable'],[2,1,NaN])
df['gait_optional_binary'] = df['gait_optional_binary'].replace([False,True],[0,1])
df['gait_speed_slower'] = df["gait_speed_slower"].replace(['Test not adequate'],[NaN])  
df['sleep'] = df['sleep'].replace(['No sleep problem', 'Occasional sleep problem', 'Permanent sleep problem'],[2,1,0])
df['health_rate'] = df['health_rate'].replace(['3 - Medium', '4 - Good', '5 - Excellent', '2 - Bad', '1 - Very bad'],[3,4,5,2,1])  
df['health_rate_comparison'] = df['health_rate_comparison'].replace(['3 - About the same', '2 - A little worse', '4 - A little better', '5 - A lot better', '1 - A lot worse'],[3,2,4,5,1])
df['activity_regular'] = df['activity_regular'].replace(['> 2 h and < 5 h per week', '< 2 h per week', '> 5 h per week'],[2,1,0])
df['smoking'] = df['smoking'].replace(['Never smoked', 'Past smoker (stopped at least 6 months)', 'Current smoker'],[2,1,0])

print(df.dtypes)
s = []

for i in col:
    df[i] = df[i].replace([NaN],df[i].mean())  #όχι τιμές μέσα στο dataset που είναι κενές
    for k in df[i].unique():
        if k is NaN:
            print(df[i].unique())
        
print(df.head(20))

list_drop = ['weight_loss','exhaustion_score','gait_speed_slower','grip_strength_abnormal','low_physical_activity']
df = df.drop(list_drop, axis=1)

print(df)
print(df.columns.values) # The names of all the columns in the data.

#Normalizing the Variables
from sklearn.preprocessing import StandardScaler
scaler = StandardScaler()
scaled_features = scaler.fit_transform(df.drop('fried', axis=1))
df_features = pd.DataFrame(scaled_features, columns= df.columns[:-1])
print(df_features.head())

#train test split (70-30% train-test)
from sklearn.model_selection import train_test_split
X_train, X_test,y_train, y_test = train_test_split(scaled_features, df['fried'],test_size=0.30) 

#Building KNN 
from sklearn.neighbors import KNeighborsClassifier
knn = KNeighborsClassifier(n_neighbors=20)
knn.fit(X_train, y_train)

#Predictions and Model Evaluation 
from sklearn.metrics import accuracy_score
pred = knn.predict(X_test)
accuracy_score(y_test,pred)

error_rate = []

for i in range(1,40):
    knn = KNeighborsClassifier(n_neighbors=i)
    knn.fit(X_train,y_train)
    pred_i = knn.predict(X_test)
    error_rate.append(np.mean(pred_i !=y_test))

#testing diff values for n 
knn = KNeighborsClassifier(n_neighbors=20)
knn.fit(X_train,y_train)
pred = knn.predict(X_test)

print('WITH K=20')
print('/n')
print(accuracy_score(y_test,pred))

# Beacons Dataset
df1 = pd.read_csv('beacons_dataset.csv',delimiter=";")
print(df1['room'].unique())

#correcting rooom labels to make them homogenous
df1['room'] = df1['room'].replace(['Kitcen','Kitchen2','Kithen','kitchen','Kiychen','Kitch','Kitcheb','Kitvhen','Kichen'],['Kitchen','Kitchen','Kitchen','Kitchen','Kitchen','Kitchen','Kitchen','Kitchen','Kitchen'])
df1['room'] = df1['room'].replace(['Livingroom1','Luvingroom1','LivingRoom2','Livingroom2','Sitingroom','Sittingroom','Leavingroom','LivingRoom','TV'],['Livingroom','Livingroom','Livingroom','Livingroom','Livingroom','Livingroom','Livingroom','Livingroom','Livingroom'])
df1['room'] = df1['room'].replace(['Liningroom','Sittigroom','Sittingroom','Sittinroom','Leavivinroom','Livingroon','LivibgRoom','Livroom','LivibgRoom','Living'],['Livingroom','Livingroom','Livingroom','Livingroom','Livingroom','Livingroom','Livingroom','Livingroom','Livingroom','Livingroom'])
df1['room'] = df1['room'].replace(['SittingRoom','LeavingRoom','SeatingRoom','livingroom','LuvingRoom','Living','SittingOver'],['Livingroom','Livingroom','Livingroom','Livingroom','Livingroom','Livingroom','Livingroom'])
df1['room'] = df1['room'].replace(['Bathroom','Bathroom-1','Barhroom','Barhroom ','Bathroon','Bathroin','Bsthroom','Bathroim','Bqthroom','Baghroom','Washroom','Bathroom1'],['Bathroom','Bathroom','Bathroom','Bathroom','Bathroom','Bathroom','Bathroom','Bathroom','Bathroom','Bathroom','Bathroom','Bathroom'])
df1['room'] = df1['room'].replace(['bedroom','Bed','Chambre','Bedroom1st','Bedroom-1','Bedroom1','Bedroom2'],['Bedroom','Bedroom','Bedroom','Bedroom','Bedroom','Bedroom','Bedroom'])
df1['room'] = df1['room'].replace(['Garden'],['Outdoor'])
df1['room'] = df1['room'].replace(['DinerRoom','DiningRoom','DinningRoom','Dinerroom','DinnerRoom'],['Diningroom','Diningroom','Diningroom','Diningroom','Diningroom'])
df1['room'] = df1['room'].replace(['Office1st','Office2','Office1','Office-2','Desk','Workroom'],['Office','Office','Office','Office','Office','Office'])
df1['room'] = df1['room'].replace(['Laundry','LaundryRoom'],['Laundryroom','Laundryroom'])
df1['room'] = df1['room'].replace(['Box-1'],['Box'])

#droping wrong and NaN room labels 
df1 = df1.drop(index=df1[df1['room']=='T'].index)
df1 = df1.drop(index=df1[df1['room']=='three'].index)
df1 = df1.drop(index=df1[df1['room']=='Three'].index)
df1 = df1.drop(index=df1[df1['room']=='Two'].index)
df1 = df1.drop(index=df1[df1['room']=='One'].index)
df1 = df1.drop(index=df1[df1['room']=='K'].index)
df1 = df1.drop(index=df1[df1['room']=='Four'].index)
df1 = df1.drop(index=df1[df1['room']=='Left'].index)
df1 = df1.drop(index=df1[df1['room']=='Right'].index)
df1 = df1.drop(index=df1[df1['room']=='Guard'].index)
df1 = df1.drop(index=df1[df1['room']=='2ndRoom'].index)
df1 = df1.dropna(axis = 0,how ='any', thresh = None, subset = None, inplace=False)
 


#print(df1['room'].value_counts())

s1= []
pattern = re.compile('[0-9]{4,4}')

#remove erroneous users
for k in df1['part_id'].unique():
    if not(bool(pattern.fullmatch(k))):
        s1.append(k)


for i in s1:
    df1 = df1.drop(index=df1[df1['part_id']==i].index)



#finding percentage of time in all the rooms for each user 

# For %
#pd.set_option('display.float_format', '{:.2%}'.format)
df2 = df1.groupby('part_id')['room'].value_counts(normalize=True).to_frame()
print(df2)

s3=[]
for i in df1['part_id'].unique():
    temp = df1.loc[df1['part_id'] == i]
    temp = temp.iloc[:,[3]]
    m=temp.value_counts(normalize=True).rename_axis('room').reset_index(name='counts')
    s3.append({'part_id': np.float64(i),'Kitchen':m[m['room']=='Kitchen']['counts'].unique(),'Livingroom': m[m['room']=='Livingroom']['counts'].unique(),
                'Bathroom':m[m['room']=='Bathroom']['counts'].unique(),'Bedroom':m[m['room']=='Bedroom']['counts'].unique()})
    

#print(s3)


df3 = pd.DataFrame(s3)


print(df3)

#merging the two preprocessed datasets              
df4 = pd.merge(df,df3,on='part_id')
print(df4)



#Clustering
from sklearn.cluster import k_means
x = df.iloc[:,:]
kmeans = KMeans(3)
kmeans.fit(x)
#clustering results
cluster_labels = kmeans.fit_predict(x)
print(cluster_labels)

range_n_clusters = [2, 3, 4, 5, 6]

for n_clusters in range_n_clusters:
         clusterer = KMeans(n_clusters=n_clusters, random_state=10)
         cluster_labels = clusterer.fit_predict(x)
         silhouette_avg = silhouette_score(x, cluster_labels)
         print("For n_clusters =",n_clusters,"The average silhouette_score is :",silhouette_avg, )




#export dataframe to csv
df4.to_csv('new_clinical_dataset.csv')

