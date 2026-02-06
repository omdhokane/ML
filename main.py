import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.preprocessing  import StandardScaler 
from sklearn.metrics import silhouette_score
from sklearn.cluster import KMeans
from kneed import KneeLocator

df=pd.read_excel("Mall_Customers.xlsx")
print(df)

x=df[['Age', 'Annual Income (k$)',
       'Spending Score (1-100)']]

scaler=StandardScaler()
x_scale=scaler.fit_transform(x)

df_scale=pd.DataFrame(x_scale,columns=['Age', 'Annual Income (k$)',
       'Spending Score (1-100)'])

wcss=[]
for k in range(1,11):
    kmeans=KMeans(n_clusters=k,init="k-means++")
    kmeans.fit(df_scale)
    wcss.append(kmeans.inertia_)
    
print(wcss)
plt.plot(range(1,11),wcss)
plt.xticks(range(1,11))
plt.xlabel("cluster")
plt.ylabel("wcss")
plt.title("elbow method")
plt.show()
kl=KneeLocator(range(1,11),wcss,curve="convex",direction="decreasing")
print("The value of cluster is ", kl.elbow)

kmean=KMeans(n_clusters=4,random_state=42)
label=kmean.fit_predict(df_scale)

df_scale["cluster"]=label
print(df_scale)

plt.scatter(df_scale["Annual Income (k$)"],
            df_scale["Spending Score (1-100)"],
            c=df_scale["cluster"])
plt.xlabel("Annual Income (k$)")
plt.ylabel("Spending Score (1-100)")
plt.title("Customer Segmentation (K-Means)")
plt.show()

