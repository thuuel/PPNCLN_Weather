#!/usr/bin/env python
# coding: utf-8

# In[1]:


import numpy as np
import pandas as pd 
import warnings
import matplotlib.pyplot as plt
warnings.filterwarnings("ignore")


# In[2]:


df=pd.read_csv(r"C:\Users\admin\OneDrive\Documents\weather_data.csv")
df.set_index('Date').sort_index()

columns_of_interest = ['TempAvgF','DewPointAvgF', 'HumidityAvgPercent', 'SeaLevelPressureAvgInches', 'VisibilityAvgMiles', 'WindAvgMPH', 'PrecipitationSumInches']
data = df[columns_of_interest]
events = df[['Events']].replace(' ', 'None')


# Xem qua thành phần của bộ dữ liệu chúng ta sẽ dùng để phân tích

# In[3]:


events


# In[4]:


data


# Vẽ đồ thị thể hiện số lượng các sự kiện thời tiết đã xảy ra trong bộ dữ liệu

# In[5]:


ax = events.Events.value_counts().plot(kind='bar', figsize=(8,4), color = plt.cm.Set2(range(len(events.Events.unique()))))
ax.set_title("Weather events in dataset", fontsize=18)


# Lọc ra những sự kiện thời tiết đơn lẻ

# In[6]:


unique_events = set()
for value in events.Events.value_counts().index:
    splitted = [x.strip() for x in value.split(',')]
    unique_events.update(splitted)
unique_events


# Gán giá trị True cho các sự kiện thời tiết xảy ra tương ứng với từng hàng, ngược lại thì False

# In[7]:


def create_event_occurrence_df(events, unique_events):
    single_events = pd.DataFrame()
    for event_type in unique_events:
        event_occurred = events.str.contains(event_type)
        single_events = pd.concat([single_events, pd.DataFrame(data={event_type: event_occurred.values})], join='outer', axis=1)
    return single_events

single_events = create_event_occurrence_df(events.Events, unique_events)
single_events


# Vẽ đồ thị thể hiện số lượng của từng sự kiện thời tiết riêng lẻ trong bộ dữ liệu

# In[8]:


ax = single_events.sum().sort_values(ascending=False).plot.bar(figsize=(8,5), color = plt.cm.Set2(range(len(events.Events.unique()))))
ax.set_title("Single weather events in dataset", fontsize=18)


# Trong bộ dữ liệu đang sử dụng, có thể thấy ở cột PrecipitationSumInches có các giá trị T bên cạnh những số cụ thể.
# Điều này có thể hiểu là vào ngày hôm đó có mưa nhưng không biết cụ thể là bao nhiêu.

# In[9]:


precipitation = data[pd.to_numeric(data.PrecipitationSumInches, errors='coerce').isnull()].PrecipitationSumInches.value_counts()
precipitation


# Kiểm tra xem bộ dữ liệu được sử dụng có bao nhiêu hàng không phải là số.

# In[10]:


def isColumnNotNumeric(columns_of_interest, data):
    result = np.zeros(data.shape[0], dtype=bool)
    for column_name in columns_of_interest:
        result = result | pd.to_numeric(data[column_name], errors='coerce').isnull()
    return result

def getDataFrameWithNonNumericRows(dataFrame):
    return data[isColumnNotNumeric(columns_of_interest, data)]

non_numeric_rows_count = getDataFrameWithNonNumericRows(data).shape[0]

print("Non numeric rows: {0}".format(non_numeric_rows_count))


# Chuyển đổi các dòng có T trong cột PrecipitationSumInches thành số 0.
# Đồng thời, tạo một cột mới tên PrecipitationTrace để lưu trữ các giả trị T này (gán 1 cho những dòng có T, 0 cho những dòng còn lại)

# In[11]:


def numberOrZero(value):
    try:
        parsed = float(value)
        return parsed
    except:
        return 0

#Find rows indices with "T" values
has_precipitation_trace_series = isColumnNotNumeric(['PrecipitationSumInches'], data).astype(int)
data = data.assign(PrecipitationTrace=has_precipitation_trace_series.values)
data['PrecipitationSumInches'] = data['PrecipitationSumInches'].apply(numberOrZero)
data.iloc[0:10,:]


# Từ các output trên, có thể thấy, ngoài cột T thì có nhiều cột khác chứa giá trị không phải số và đã được chuyển thành null.
# Vì thế, phải cân nhắc xử lí để mô hình chạy hiệu quả.

# In[12]:


getDataFrameWithNonNumericRows(data)


# In[13]:


row_indices_for_missing_values = getDataFrameWithNonNumericRows(data).index.values
data_prepared = data.drop(row_indices_for_missing_values)
events_prepared = single_events.drop(row_indices_for_missing_values)
print("Data rows: {0}, Events rows: {1}".format(data_prepared.shape[0], events_prepared.shape[0]))


# Vì mô hình của học máy không giám sát nhóm sử dụng được chạy trên các dữ liệu dạng số.
# Vì thế, phải kiểm tra xem loại dữ liệu của các cột và ép kiểu nếu cần.

# In[14]:


data_prepared.dtypes


# In[15]:


data_prepared = data_prepared.apply(pd.to_numeric)
data_prepared.dtypes


# In[16]:


data_prepared


# Bắt đầu chuẩn hóa dữ liệu để huấn luyện mô hình

# In[17]:


from sklearn import preprocessing
data_values = data_prepared.values #returns a numpy array
min_max_scaler = preprocessing.MinMaxScaler()

data_prepared = pd.DataFrame(min_max_scaler.fit_transform(data_prepared), columns=data_prepared.columns, index=data_prepared.index)


# In[18]:


data_prepared.head()


# In[19]:


events_prepared.head()


# Chia dữ liệu thành 2 tập riêng biệt để huấn luyện và kiểm thử

# In[20]:


from sklearn.model_selection import train_test_split

random_state = 42
X_train, X_test = train_test_split(data_prepared, test_size=0.2, random_state=random_state)
y_train, y_test = train_test_split(events_prepared, test_size=0.2, random_state=random_state)

clusters_count = len(unique_events)


# In[21]:


#hàm gán tên từng sự kiện với tên các cụm tương ứng
def create_cluster_mapping(events_prepared, resultDf):
    event_names_ordered = events_prepared.sum().sort_values(ascending=False).index
    clusters_ordered = resultDf.iloc[:,0].value_counts().index
    cluster_category_mapping = {}
    clusters_count = len(clusters_ordered)
    for i in range(clusters_count):
        cluster_category_mapping.update({clusters_ordered[i]:event_names_ordered[i]})
    return cluster_category_mapping

#xem giá trị  dự đoán có đúng vs gtri thực tế
def check_accuracy(X, y):    
    comparison = X == y
    val_counts = comparison.all(axis=1).value_counts()
    percentageCorrect = val_counts.at[True] / X.shape[0] * 100
    return percentageCorrect
y_train_col_ordered = y_train.reindex(sorted(y_train.columns), axis=1)


# Sử dụng những thuật toán phân cụm và so sánh với kết quả thực tế. Từ đó, đưa ra thuật toán cho kết quả gần với thực tế nhất.

# In[22]:


from sklearn.cluster import KMeans
warnings.filterwarnings("ignore")
kmeans = KMeans(n_clusters=clusters_count).fit(X_train)
resultDf1 = pd.DataFrame(kmeans.labels_)
resultDf1.index = X_train.index
ax = resultDf1.iloc[:,0].value_counts().plot.bar(color = plt.cm.Set2(range(len(events.Events.unique()))))
ax.set_title("K-Means clustering", fontsize=16)


# In[23]:


resultDf1.replace(create_cluster_mapping(y_train, resultDf1), inplace=True)
result_kmean = create_event_occurrence_df(resultDf1[0], unique_events)
result_kmean.index = resultDf1.index
result_kmean.reindex(sorted(result_kmean.columns), axis=1)
result_kmean.sort_index()


# In[24]:


result_kmean = result_kmean.reindex(sorted(result_kmean.columns), axis=1)
a = check_accuracy(result_kmean.sort_index(), y_train_col_ordered.sort_index())


# In[25]:


from sklearn.cluster import SpectralClustering
warnings.filterwarnings("ignore")
sc = SpectralClustering(n_clusters=clusters_count).fit(X_train)
resultDf2 = pd.DataFrame(sc.labels_)
resultDf2.index = X_train.index
ax = resultDf2.iloc[:,0].value_counts().plot.bar(color = plt.cm.Set2(range(len(events.Events.unique()))))
ax.set_title("Spectral Clustering", fontsize=16)


# In[26]:


resultDf2.replace(create_cluster_mapping(y_train, resultDf2), inplace=True)
result_spectral = create_event_occurrence_df(resultDf2[0], unique_events)
result_spectral.index = resultDf2.index
result_spectral.reindex(sorted(result_spectral.columns), axis=1)
result_spectral.sort_index()


# In[27]:


result_spectral = result_spectral.reindex(sorted(result_spectral.columns), axis=1)
b = check_accuracy(result_spectral.sort_index(), y_train_col_ordered.sort_index())


# In[28]:


from sklearn.cluster import DBSCAN
dbscan = DBSCAN(eps=0.25, min_samples=4).fit(X_train)
resultDf3 = pd.DataFrame(dbscan.labels_)
resultDf3.index = X_train.index
ax = resultDf3.iloc[:,0].value_counts().plot.bar(color = plt.cm.Set2(range(len(events.Events.unique()))))
ax.set_title("DBSCAN", fontsize=16)


# In[29]:


resultDf3.replace(create_cluster_mapping(y_train, resultDf3), inplace=True)
result_dbscan = create_event_occurrence_df(resultDf3[0], unique_events)
result_dbscan.index = resultDf3.index
result_dbscan.reindex(sorted(result_dbscan.columns), axis=1)
result_dbscan.sort_index()


# In[30]:


result_dbscan = result_dbscan.reindex(sorted(result_dbscan.columns), axis=1)
c = check_accuracy(result_dbscan.sort_index(), y_train_col_ordered.sort_index())


# In[31]:


from sklearn.cluster import AgglomerativeClustering
ac = AgglomerativeClustering(n_clusters=clusters_count, linkage="average").fit(X_train)
resultDf = pd.DataFrame(ac.labels_)
ax = resultDf.iloc[:,0].value_counts().plot.bar(color = plt.cm.Set2(range(len(events.Events.unique()))))
ax.set_title("Agglomerative Clustering", fontsize=16)
resultDf4 = resultDf.copy()
resultDf4.index = X_train.index


# In[32]:


resultDf4.replace(create_cluster_mapping(y_train, resultDf4), inplace=True)
result_agglo = create_event_occurrence_df(resultDf4[0], unique_events)
result_agglo.index = resultDf4.index
result_agglo.reindex(sorted(result_agglo.columns), axis=1)
result_agglo.sort_index()


# In[33]:


result_agglo = result_agglo.reindex(sorted(result_agglo.columns), axis=1)
d = check_accuracy(result_agglo.sort_index(), y_train_col_ordered.sort_index())


# In[34]:


accuracy_df = {'K-means': a, 'Spectral': b, 'DBSCAN': c, 'Agglomerative': d}
accuracy_df = pd.DataFrame(accuracy_df, index = ['Accuracy'])
accuracy_df


# Có thể thấy, thuật toán phân cụm Agglomerative đem lại kết quả phân cụm gần với thực tế nhất.

# In[35]:


fig, ax = plt.subplots(1, 2, figsize=(15, 5))
events_prepared.sum().sort_values(ascending=False).plot.bar(ax=ax[0], title="Real events that happened", color = plt.cm.Set2(range(len(events.Events.unique()))))
resultDf.iloc[:,0].value_counts().plot.bar(ax=ax[1], title="Bar obtained from agglomerative clustering", color = plt.cm.Set2(range(len(events.Events.unique()))))


# Thực hiện gán tên cụm vào tên sự kiện thời tiết tương ứng. 
# Sau đó, sử dụng lý thuyết phân cụm của thuật toán Agglomerative để xem rằng liệu 1 ngày có thể có 2 sự kiện thời tiết hay không.

# In[36]:


cluster_category_mapping = create_cluster_mapping(y_test, resultDf)


# In[37]:


#tọa độ tâm từng cụm
cluster_centers_mapping = {}
for key in cluster_category_mapping:
    cluster_indices = resultDf.loc[resultDf[0] == key].index
    cluster_data = X_train.iloc[cluster_indices]
    mean = cluster_data.mean(axis=0).values
    #print("\n" + cluster_category_mapping[key])
    #print(mean)
    cluster_centers_mapping.update({key:mean})
cluster_centers_mapping


# In[38]:


#tính khoảng cách từ 1 điểm đến tâm từng cụm
def get_distances_from_cluster(data_frame):
    cluster_distance = np.zeros((data_frame.shape[0], clusters_count))
    #khoảng cách euclidean
    for i in range(data_frame.shape[0]):
        for key in cluster_category_mapping:
            dist = np.linalg.norm(data_frame.iloc[[i]].values[0]-cluster_centers_mapping[key])
            cluster_distance[i,key] = dist
            #print(dist)
    column_names = [cluster_category_mapping[k] for k in cluster_category_mapping]
    #column_names

    return pd.DataFrame(cluster_distance, index=data_frame.index, columns=column_names)

distancesDf = get_distances_from_cluster(X_train)
distancesDf.sort_index().head()


# In[39]:


def classify_events(distances_dataFrame):
    return distances_dataFrame.apply(lambda x: x<x.min()*1.01, axis=1)


# Đánh giá mô hình phân cụm dựa trên tỷ lệ giữa số dòng dự báo đúng và tổng số dòng của 2 tập X_train và X_test

# In[40]:


classification_result = classify_events(distancesDf)
X_train_col_ordered = classification_result.reindex(sorted(classification_result.columns), axis=1)
y_train_col_ordered = y_train.reindex(sorted(y_train.columns), axis=1)
a = check_accuracy(X_train_col_ordered, y_train_col_ordered)

fig, ax = plt.subplots(1, 2, figsize=(15, 5))
y_train_col_ordered.sum().plot.bar(ax=ax[0], title="Real events that happened", color = plt.cm.Set2(range(len(events.Events.unique()))))
ax = X_train_col_ordered.sum().plot.bar(ax=ax[1], title="Predicted events", color = plt.cm.Set2(range(len(events.Events.unique()))))
#resultDf.iloc[:,0].value_counts().plot.bar(ax=ax[1], title="Histogram obtained from agglomerative clustering")
print(f'Chỉ số Accuracy của mô hình là {a}')


# In[41]:


distancesDf = get_distances_from_cluster(X_test)
classification_result = classify_events(distancesDf)
X_test_col_ordered = classification_result.reindex(sorted(classification_result.columns), axis=1)
y_test_col_ordered = y_test.reindex(sorted(y_train.columns), axis=1)
a = check_accuracy(X_test_col_ordered, y_test_col_ordered)

fig, ax = plt.subplots(1, 2, figsize=(15, 5))
y_test_col_ordered.sum().plot.bar(ax=ax[0], title="Real events that happened", color = plt.cm.Set2(range(len(events.Events.unique()))))
X_test_col_ordered.sum().plot.bar(ax=ax[1], title="Predicted events", color = plt.cm.Set2(range(len(events.Events.unique()))))
print(f'Chỉ số Accuracy của mô hình là {a}')

