import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.cluster import MiniBatchKMeans

# 1. Read data
data = pd.read_csv('OnlineRetail.zip', encoding='unicode_escape')
df = pd.read_csv('clean_data.zip')
df_RFM = pd.read_csv('data_RFM.csv')
df_scale = pd.read_csv('data_scale.csv')

df['TotalAmount'] = df['Quantity'] * df['UnitPrice']

# Model
mbk = MiniBatchKMeans(n_clusters=3, random_state=0)
mbk.fit(df_scale.iloc[:,1:])
# Result
centroids = mbk.cluster_centers_
labels = mbk.labels_
df_result = df_RFM.copy()
df_result['Group'] = pd.Series(labels, index=df_RFM.index)
# # Visualization
fig, (ax0, ax1, ax2) = plt.subplots(1, 3, figsize=(15, 7), tight_layout=True)
ax0.scatter(df_result.Recency, df_result.Frequency, c=labels, s=30, cmap='rainbow')
ax0.set_xlabel('Recency')
ax0.set_ylabel('Frequency')
ax1.scatter(df_result.Frequency, df_result.Monetary, c=labels, s=30, cmap='rainbow')
ax1.set_xlabel('Frequency')
ax1.set_ylabel('Monetary')
ax2.scatter(df_result.Recency, df_result.Monetary, c=labels, s=30, cmap='rainbow')
ax2.set_xlabel('Recency')
ax2.set_ylabel('Monetary')

# Explaining the groups
rfm_group = df_result.groupby('Group').agg({
    'Recency': 'mean',
    'Frequency': 'mean',
    'Monetary': ['mean', 'count']}).round(0)
rfm_group.columns = rfm_group.columns.droplevel()
rfm_group.columns = ['RecencyMean','FrequencyMean','MonetaryMean', 'Count']
rfm_group['Percent'] = round((rfm_group['Count']/rfm_group.Count.sum())*100, 2)
rfm_group = rfm_group.reset_index()
rfm_group['Group'] = 'Group '+ rfm_group['Group'].astype('str')

fig1, ax = plt.subplots(figsize=(15, 7))
sns.scatterplot(data=rfm_group, x='RecencyMean', y='MonetaryMean', size='FrequencyMean', hue='Group', ax=ax, sizes=(30, 530))

dct_group = {0 : 'Khách hàng thường',
             1 : 'Khách hàng VIP',
             2 : 'Khách hàng thân thiết'}
df_result_group = df_result.copy()
df_result_group['Group'] = df_result_group['Group'].apply(lambda x: dct_group[x])

# GUI
st.title('Data Science Project 1')
st.write('## Customers Segmentation')

menu = ['Giới thiệu', 'Tiền xử lý dữ liệu', 'Chọn số nhóm', 'Mô hình tối ưu', 'Tìm nhóm']
choice = st.sidebar.selectbox('Menu', menu)

if choice == 'Giới thiệu':
    st.subheader('Giới thiệu')
    st.write('''
    - Công ty X chủ yếu bán các sản phẩm là quà tặng dành cho những dịp đặc biệt. Nhiều khách hàng của công ty là khách hàng bán buôn.
    - Công ty X mong muốn có thể bán được nhiều sản phẩm hơn cũng như giới thiệu sản phẩm đến đúng đối tượng khách hàng, chăm sóc và làm hài lòng khách hàng.
    ''')
    st.markdown('**Mục tiêu/ vấn đề:** Xây dựng hệ thống phân cụm khách hàng dựa trên các thông tin do công ty cung cấp từ đó có thể giúp công ty xác định các nhóm khách hàng khác nhau để có chiến lược kinh doanh, chăm sóc khách hàng phù hợp.')

elif choice == 'Tiền xử lý dữ liệu':
    st.subheader('Tiền xử lý dữ liệu')
    st.write('#### 1. Dữ liệu')
    st.table(data.head(3))
    st.table(data.tail(3))

    st.write('#### 2. Phân tích RFM')
    st.write('''
    - Recency (R): được tính dựa trên `InvoiceDate`. 
    - Frequency (F): được tính dựa trên `InvoiceNo`.
    - Monetary Value (M): được tính dựa trên `Quantity` và `UnitPrice`. 
    ''')
    st.write('##### Dữ liệu RFM:')
    st.table(df_RFM.head())

elif choice == 'Chọn số nhóm':
    st.subheader('Chọn số nhóm')
    k = st.slider('Chọn số nhóm:', min_value=2, max_value=10, step=1)
    # Model
    mbk = MiniBatchKMeans(n_clusters=k, random_state=0)
    mbk.fit(df_scale.iloc[:,1:])
    # Result
    centroids = mbk.cluster_centers_
    labels = mbk.labels_
    df_result = df_RFM.copy()
    df_result['Group'] = pd.Series(labels, index=df_RFM.index)
    # Groups
    st.write('##### Số khách hàng trong mỗi nhóm:')
    st.dataframe(df_result.groupby('Group').size().rename('Số khách hàng'))
    # Visualization
    fig, (ax0, ax1, ax2) = plt.subplots(1, 3, figsize=(15, 7), tight_layout=True)
    ax0.scatter(df_result.Recency, df_result.Frequency, c=labels, s=30, cmap='rainbow')
    ax0.set_xlabel('Recency')
    ax0.set_ylabel('Frequency')
    ax1.scatter(df_result.Frequency, df_result.Monetary, c=labels, s=30, cmap='rainbow')
    ax1.set_xlabel('Frequency')
    ax1.set_ylabel('Monetary')
    ax2.scatter(df_result.Recency, df_result.Monetary, c=labels, s=30, cmap='rainbow')
    ax2.set_xlabel('Recency')
    ax2.set_ylabel('Monetary')
    st.pyplot(fig)
    # Explaining the groups
    st.write('##### Đặc trưng của từng nhóm:')
    rfm_group = df_result.groupby('Group').agg({
        'Recency': 'mean',
        'Frequency': 'mean',
        'Monetary': ['mean', 'count']}).round(0)
    rfm_group.columns = rfm_group.columns.droplevel()
    rfm_group.columns = ['RecencyMean','FrequencyMean','MonetaryMean', 'Count']
    rfm_group['Percent'] = round((rfm_group['Count']/rfm_group.Count.sum())*100, 2)
    rfm_group = rfm_group.reset_index()
    rfm_group['Group'] = 'Group '+ rfm_group['Group'].astype('str')
    st.table(rfm_group)
    fig1, ax = plt.subplots(figsize=(15, 7))
    sns.scatterplot(data=rfm_group, x='RecencyMean', y='MonetaryMean', size='FrequencyMean', hue='Group', ax=ax)
    st.pyplot(fig1)

elif choice == 'Mô hình tối ưu':
    st.subheader('Mô hình tối ưu')
    # Groups
    st.write('##### Số nhóm: 3.')
    st.write('##### Số khách hàng trong mỗi nhóm:')
    st.dataframe(df_result.groupby('Group').size().rename('Số khách hàng'))
    st.pyplot(fig)
    # Explaining the groups
    st.write('##### Đặc trưng của từng nhóm:')
    st.table(rfm_group)
    st.pyplot(fig1)
    st.markdown('''
    - Nhóm 0: 3979 khách, chi tiêu không nhiều và mua sắm không thường xuyên --> **Khách hàng thường**
    - Nhóm 1: 18 khách, chi tiêu nhiều và mua sắm thường xuyên nhất --> **Khách hàng VIP**
    - Nhóm 2: 342 khách, chi tiêu khá nhiều và mua sắm khá thường xuyên --> **Khách hàng thân thiết**
    ''')


else:
    st.subheader('Tìm nhóm')
    customer_id = st.number_input('Nhập ID khách hàng:', step=1)
    if customer_id in list(df_result_group['CustomerID']):
        cluster = df_result_group.loc[df_result_group['CustomerID']==customer_id, 'Group'].values[0]
        result = 'Khách hàng ID = ' + str(customer_id) + ' thuộc nhóm ' + str(cluster) + '.'
    else:
        result = 'Không tìm thấy khách hàng ID = ' + str(customer_id) + '.'
    st.write(result)