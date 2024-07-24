import streamlit as st
from streamlit_option_menu import option_menu
import pickle

# LIBRARY
import pandas as pd
import math
from sklearn.preprocessing import LabelEncoder, MinMaxScaler
import numpy as np
from imblearn.over_sampling import SMOTE
import seaborn as sns
import matplotlib.pyplot as plt

from sklearn.model_selection import train_test_split, KFold, cross_val_score
from sklearn.svm import SVC
from sklearn.metrics import confusion_matrix, accuracy_score
from sklearn.feature_selection import mutual_info_classif


st.set_page_config(
    page_title="Klasifikasi Stroke dengan SVM",
    page_icon='https://raw.githubusercontent.com/Amel039/Stroke-Classification/main/stroke.jpg',
    
    initial_sidebar_state="expanded",
    layout="centered"
   
)

st.write("""<h1 style="text-align: center;">KLASIFIKASI PENYAKIT STROKE DENGAN SELEKSI FITUR INFORMATION GAIN MENGGUNAKAN METODE SUPPORT VECTOR MACHINE </h1><br>""", unsafe_allow_html=True)

with st.container():
    with st.sidebar:
        selected = option_menu(
            st.write("""<h3 style="text-align: center;"> SAN SAYIDUS SOLATAL A`LA <p>200411100032</p></h3>""", unsafe_allow_html=True),
            ["Home", "Description", "Dataset", "Prepocessing" ,"Modeling", "Implementation"],
            icons=['house', 'file-earmark-font', 'bar-chart', 'gear', 'arrow-down-square', 'check2-square'],
            menu_icon="cast", default_index=0,
            styles={
                "container": {"padding": "0!important", "background-color": "#6495ED"},
                "icon": {"color": "white", "font-size": "18px"},
                "nav-link": {"font-size": "16px", "text-align": "left", "margin": "0px", "color": "white"},
                "nav-link-selected": {"background-color": "#00008B"}
            }
        )


    if selected == "Home":
        st.write("""<div style="text-align: justify;">
        Klasifikasi merupakan salah satu teknik data mining yang dapat digunakan untuk prediksi kelas dari suatu kelompok data. Klasifikasi penyakit stroke dengan menggunakan fitur Information Gain. Support Vector Machine (SVM) merupakan metode supervised learning yang sering digunakan untuk klasifikasi dan regresi. 
        SVM juga dapat mengatasi masalah klasifikasi dan regresi untuk data linier maupun non-linier Penelitian ini bertujuan untuk meningkatkan akurasi klasifikasi metode SVM pada data penyakit stroke dengan menggunakan seleksi fitur information gain. Penggunaan metode SVM dengan waktu pengujiannya yang singkat 
        diperlukan juga untuk memperkecil beban komputasinya. Salah satu cara untuk memperkecil beban komputasi data sebelum dilakukan uji dengan menggunakan SVM adalah dengan melakukan seleksi fitur Information Gain. Oleh karena itu, peneliti tertarik untuk meneliti terkait klasifikasi penyakit stroke dengan menggunakan 
        metode SVM menggunakan fitur Information Gain. Penelitian ini merujuk pada penelitian sebelumnya yang telah membandingkan metode SVM dengan metode lainnya dapat menghasilkan akurasi yang kurang tinggi sehingga diperlukan ekstraksi fitur agar dapat meningkatkan nilai akurasi akurasi terbaik daripada peneliti yang menggunakan 
        metode lainnya dengan metode Information Gain dapat mengurangi dimensi fitur pada klasifikasi dengan menerapkan teknik scoring dalam melakukan pembobotan menggunakan maksimal entropy.<br>
      </div>
      """, unsafe_allow_html=True)
        
    elif selected == "Description":
        st.subheader("""Pengertian""")
        st.write("""<div style="text-align: justify;">
        Dataset yang digunakan pada penelitian ini merupakan data Prediksi Stroke. Dataset ini berjumlah 5110 data dengan 11 fitur dan 1 label berisi 2 kelas yakni 1 (stroke) dan 0 (tidak stroke). Dataset ini berisi jumlah diagnosis stroke 249 orang dan tidak stroke sebanyak 4861 orang. Dan di dalam dataset terdapat nilai kosong (missing value) sebanyak 201 data.
        Dataset stroke ini berasal dari penelitian thesis Oluwafemi Emmanuel Zachariah tentang “Prediksi Penyakit Stroke dengan Data Demografis Dan Perilaku Menggunakan Algoritma SVM” dari Sheffield Hallam University yang diambil dari catatan kesehatan dari berbagai rumah sakit di Bangladesh yang dapat diakses melalui link berikut  ini https://www.kaggle.com/datasets/fedesoriano/stroke-prediction-dataset/.<br>
        </div>
        """, unsafe_allow_html=True)
        

        st.subheader("""Kegunaan Dataset""")
        st.write("""
        Data ini akan digunakan untuk melakukan prediksi atau klasifikasi penderita Stroke.
        """)

        st.subheader("""Fitur""")
        st.markdown(
            """
            Fitur-fitur yang terdapat pada dataset:
            -   ID	Merupakan id unik tiap pasien bertipe Numerik
            -   Gender	Merupakan fitur jenis kelamin pasien. berisi dari kategori yaitu female (perempuan), male (laki-laki)  dan other (lainnya) bertipe	Kategorik
            -   Age	Merupakan fitur umur pasien	Numerik
            -   Hypertension,	Merupakan fitur yang berisi apakah pasien tersebut memiliki penyakit jantung atau tidak. Berisi kategori 0 dan kategori 1 bertipe Kategorik
            -   heart_disease	Fitur  apakah pasien tersebut memiliki penyakit hipertensi (1) atau tidak (0) bertipe Kategorik
            -   Ever_married	Fitur berisi apakah pasien sudah pernah menikah atau tidak, terdiri dari 2 kategori yaitu kategori Yes dan No bertipe Kategorik
            -   Work_type	Fitur tipe pekerjaan berisi 5 fitur yakni Private (pribadi) , self- employed (bekerja sendiri), children (anak-anak) , Govt_job (pekerjaan pemerintahan) , Never_worked (tidak pernah bekerja) bertipeKategorik
            -   Residence_type	Fitur jenis tempat tinggal berisi berisi 2 kategori Urban (Perkotaan) dan Rural (Pedesaan) bertipe	Kategorik
            -   avg_glucose_level	Fitur berisi rata rata tingkat kadar glukosa dalam darah pada pasien bertipe Numerik 
            -   bmi	Fitur bmi (body mass index) pasien bertipe Numerik
            -   smoking_status	fitur 	berisi status merokok pasien, Never smoke (tidak pernah merokok), Unknown (tidak diketahui), formery smoked (sebelumnya merokok), smoked (merokok) bertipe Kategorik
            -   stroke	Fitur berisi label diagnosis stroke berisi 2 kategori 1 (stroke) orang dan 0 (tidak) bertipe Kategorik

            """
        )

        st.subheader("""Sumber Dataset""")
        st.write("""
        Sumber data di dapatkan melalui website kaggle.com, Berikut merupakan link untuk mengakses sumber dataset.
        <br>
        <a href="https://www.kaggle.com/datasets/fedesoriano/stroke-prediction-dataset/"> Kunjungi Sumber Data di Kaggle</a>
        <br>
        <a href="https://raw.githubusercontent.com/SanS2A/dataset/main/healthcare-dataset-stroke-data.csv"> Lihat Dataset Stroke</a>""", unsafe_allow_html=True)
      
        st.subheader("""Tipe Data""")
        st.write("""
        Tipe data yang di gunakan pada dataset ini adalah numerik dan kategorikal.
        """)

    elif selected == "Dataset":
        # Memuat data
        df = pd.read_csv('https://raw.githubusercontent.com/SanS2A/dataset/main/healthcare-dataset-stroke-data.csv')

        # Menampilkan header dan dataset
        st.header("Analisa Data Stroke")
        st.subheader("Dataset Stroke")
        st.dataframe(df, width=1000)

        # Mengkategorikan kolom age
        df['age_group'] = pd.cut(df['age'], bins=[0, 17, 65, 79, float('inf')], labels=['0-17', '18-65', '66-79', '>79'])
        # Mengkategorikan kolom avg_glucose_level
        df['avg_group'] = pd.cut(df['avg_glucose_level'], bins=[0, 100, 200, float('inf')], labels=['0-100', '101-200', '>200'])
        # Mengkategorikan kolom bmi
        df['bmi_group'] = pd.cut(df['bmi'], bins=[0, 24.9, 29.9, float('inf')], labels=['0-24.9', '25-29.9', '>29.9'])

        # Menghitung jumlah nilai kosong pada setiap kolom
        jumlah_nilai_kosong = df.isna().sum()

        # Mendefinisikan variabel yang ingin ditampilkan jumlah nilainya
        variables = ['gender', 'age_group', 'hypertension', 'heart_disease', 'ever_married',
                    'work_type', 'Residence_type', 'avg_group', 'bmi_group', 'smoking_status']

        # Membuat figure
        fig, axes = plt.subplots(5, 2, figsize=(15, 20))
        axes = axes.flatten()

        # Menghitung total data
        total_data = len(df)

        # Iterasi melalui setiap variabel
        for i, var in enumerate(variables):
            # Plot histogram untuk variabel saat ini
            sns.histplot(df, x=var, ax=axes[i], color='skyblue')

            # Menambahkan jumlah total data dan jumlah nilai kosong di label sumbu y
            axes[i].set_ylabel(f"\nTotal Data: {total_data}\nMissing: {jumlah_nilai_kosong[var]}")

            # Menghitung dan menambahkan teks jumlah nilai kategori di tengah diagram batangnya
            value_counts = df[var].value_counts()
            total_count = value_counts.sum()
            for idx, (category, count) in enumerate(value_counts.items()):
                axes[i].text(category, count / 2, str(count), ha='center', va='center', fontsize=12)
        plt.tight_layout()

        # Tampilkan dengan Streamlit
        st.pyplot(fig)
       
    elif selected == "Prepocessing":
        df = pd.read_csv('https://raw.githubusercontent.com/SanS2A/dataset/main/healthcare-dataset-stroke-data.csv')
        st.subheader('Hasil Penghapusan Kolom ID ')
        df = df.drop('id', axis=1)
        st.dataframe(df, width=800)

        st.subheader("""Cek missing value:""")
        mis = df.isnull().sum().reset_index()
        mis.columns = ['Fitur', 'Jumlah Missing Values']
        st.dataframe(mis, width=400)

        # Imputasi Missing Values dengan mean
        st.subheader("""Penanganan data kosong:""")
        st.write("""Imputasi dengan metode Mean mengisi missing data dalam suatu variable dengan rata-rata dari semua nilai yang diketahui pada suatu variabel.""")
        df['bmi'].fillna(math.floor(df['bmi'].mean()),inplace=True)
        mis = df.isnull().sum().reset_index()
        mis.columns = ['Fitur', 'Jumlah Missing Values']
        st.dataframe(mis, width=400)

        st.subheader("""Label Encoding""")
        st.write("""Dilakukan perubahan data dari kategorikal menjadi numerik berdasarkan penomoran kategori secara berurutan.""")
        encode = LabelEncoder()
        # transformasi data
        df['gender'] = encode.fit_transform(df['gender'].values)
        df['ever_married'] = encode.fit_transform(df['ever_married'].values)
        df['work_type'] = encode.fit_transform(df['work_type'].values)
        df['Residence_type'] = encode.fit_transform(df['Residence_type'].values)
        df['smoking_status'] = encode.fit_transform(df['smoking_status'].values)
        st.dataframe(df)
        df.to_csv('data_.csv', index=False)
        # Normalisasi
        st.subheader('Normalisasi data')
        st.write("""Min-Max Scaler mengubah nilai data menjadi rentang nilai 0 hingga 1. Dengan Min-Max Scaler, semua nilai dalam atribut akan berada dalam rentang 0 hingga 1.""")
        columns_to_normalize = ['age', 'bmi', 'avg_glucose_level']
        scaler = MinMaxScaler()
        scaled = scaler.fit_transform(df[columns_to_normalize])
        df[columns_to_normalize] = scaled
        st.dataframe(df)

        # Simpan DataFrame setelah semua operasi pemrosesan
        df.to_csv('data_preprocess.csv', index=False)
        # # Menampilkan hasil imputasi
        # st.dataframe(df_imputed, width=600)

        #SMOTE    
        st.subheader('Synthetic Minority Over-sampling Technique (SMOTE)')
        st.write("""Teknik SMOTE bekerja dengan menambahkan data pada kelas minor untuk menyeimbangkan jumlah data yang sama dengan kelas mayor dengan cara membangkitkan data sintesis.""")
        # x adalah atribut yang mempengaruhi
        # y adalah label itu sendiri
        X = df.drop(['stroke'],axis=1)
        y = df['stroke']

        # Visualisasi data sebelum SMOTE
        st.subheader('Distribusi Data Sebelum SMOTE')
        fig_before, ax_before = plt.subplots(figsize=(8, 6))  # Perkecil ukuran diagram
        sns.countplot(x=y, ax=ax_before)
        ax_before.set_title('Distribusi Label Sebelum SMOTE')
        for p in ax_before.patches:
            ax_before.annotate(f'{p.get_height()}', (p.get_x() + p.get_width() / 2, p.get_height()), ha='center', va='bottom')
        st.pyplot(fig_before)

#SMOTE
        smote = SMOTE(k_neighbors=3,random_state=42)
        # sampling smote
        X_resampled, y_resampled = smote.fit_resample(X,y)
        
       # Visualisasi data setelah SMOTE
        st.subheader('Distribusi Data Setelah SMOTE')
        fig_after, ax_after = plt.subplots(figsize=(8, 6))  # Perkecil ukuran diagram
        sns.countplot(x=y_resampled, ax=ax_after)
        ax_after.set_title('Distribusi Label Setelah SMOTE')
        for p in ax_after.patches:
            ax_after.annotate(f'{p.get_height()}', (p.get_x() + p.get_width() / 2, p.get_height()), ha='center', va='bottom')
        st.pyplot(fig_after)

        # Simpan DataFrame setelah SMOTE
        df.to_csv('SMOTE.csv', index=False)

    elif selected == "Modeling":
        with st.form("modeling"):
            st.subheader('Modeling')
            st.write("Pilihlah model yang akan dilakukan pengecekkan akurasi:")
            SVM = st.checkbox('SVM')
            SVM_IG = st.checkbox('SVM IG')
            SVM_SMOTE = st.checkbox('SVM SMOTE')
            SVM_SMOTE_IG = st.checkbox('SVM SMOTE IG')
            submitted = st.form_submit_button("Submit")
            def save_model(model, params, feature_names, filename):
                model_info = {
                    'model': model,
                    'params': params,
                    'feature_names': feature_names
                }

            if SVM:
                st.subheader('SKENARIO 1 : SVM Basic')
                st.subheader('SVM kernel RBF')
                # Fungsi untuk memuat data dengan caching

                data = pd.read_csv('https://raw.githubusercontent.com/SanS2A/dataset/main/nonSMOTE.csv')
                data = data.drop(['Unnamed: 0'], axis=1)

                # Menghilangkan kolom 'stroke' dan membuat variabel target
                X = data.drop('stroke', axis=1)
                y = data['stroke']

                # Memisahkan dataset menjadi data pelatihan dan pengujian
                x_train, x_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

                # Membuat objek k_fold
                k_fold = KFold(n_splits=5)
                # Menentukan parameter yang akan diuji
                params = {
                    'C': [0.1, 1, 10, 100],
                    'gamma': [0.01, 0.1, 1, 10, 100]
                }
                # Variasi parameter
                param_combinations = [(C, gamma) for C in params['C'] for gamma in params['gamma']]

                # Untuk menyimpan hasil cross-validation
                results = []
                # Melakukan cross-validation untuk setiap kombinasi parameter
                for i, (C, gamma) in enumerate(param_combinations):
                    # Inisialisasi model SVM dengan kernel polynomial
                    svm_rbf = SVC(kernel='rbf', C=C, gamma=gamma)

                    # Melakukan cross-validation
                    scores = cross_val_score(svm_rbf, x_train, y_train, cv=k_fold, scoring='accuracy')

                    # Menyimpan hasil iterasi
                    results.append({
                        'params': {'C': C, 'gamma': gamma},
                        'scores': scores,
                        'iteration': i+1  # Menyimpan nomor iterasi
                    })
                    # Menampilkan hasil akurasi dengan 5-fold pada tiap iterasi
                    st.write(f"Iterasi ke - {i+1} :")
                    st.write(f"Hyperparameter : C = {C}, gamma = {gamma}")
                    st.write("Detail Akurasi Tiap Fold :")
                    for j, score in enumerate(scores):
                        st.write(f"Fold ke - {j+1} : {score}")
                    st.write()

               # Menentukan model terbaik berdasarkan akurasi fold tertinggi
                best_result = max(results, key=lambda x: max(x['scores']))
                best_params = best_result['params']
                best_iteration = best_result['iteration']
                best_fold_index = best_result['scores'].tolist().index(max(best_result['scores'])) + 1
                st.write(f"Model terbaik pada iterasi ke - {best_iteration} : C = {best_params['C']}, gamma = {best_params['gamma']}")
                st.write(f"Akurasi fold tertinggi pada fold ke - {best_fold_index} : {max(best_result['scores'])}")

################
                st.subheader('SVM kernel Polynomial')

                # Menentukan parameter yang akan diuji
                params = {
                    'C': [0.1, 1, 10, 100],
                    'degree': [1, 2, 3, 4, 5]
                }
                # Variasi parameter
                param_combinations = [(C, degree) for C in params['C'] for degree in params['degree']]

                # Untuk menyimpan hasil cross-validation
                results = []
                # Melakukan cross-validation untuk setiap kombinasi parameter
                for i, (C, degree) in enumerate(param_combinations):
                    # Inisialisasi model SVM dengan kernel polynomial
                    svm_poly = SVC(kernel='poly', C=C, degree=degree)

                    # Melakukan cross-validation
                    scores = cross_val_score(svm_poly, x_train, y_train, cv=k_fold, scoring='accuracy')

                    # Menyimpan hasil iterasi
                    results.append({
                        'params': {'C': C, 'degree': degree},
                        'scores': scores,
                        'iteration': i+1  # Menyimpan nomor iterasi
                    })
                    # Menampilkan hasil akurasi dengan 5-fold pada tiap iterasi
                    st.write(f"Iterasi ke - {i+1} : ")
                    st.write(f"Hyperparameter : C = {C}, degree = {degree}")
                    st.write("Detail Akurasi Tiap Fold : ")
                    for j, score in enumerate(scores):
                        st.write(f"Fold ke - {j+1} : {score}")
                    st.write("")

                # Menentukan model terbaik berdasarkan akurasi fold tertinggi
                best_result = max(results, key=lambda x: max(x['scores']))
                best_params = best_result['params']
                best_iteration = best_result['iteration']
                best_fold_index = best_result['scores'].tolist().index(max(best_result['scores'])) + 1
                st.write(f"Model terbaik pada iterasi ke - {best_iteration} : C = {best_params['C']}, degree = {best_params['degree']}")
                st.write(f"Akurasi fold tertinggi pada fold ke - {best_fold_index} : {max(best_result['scores'])}")

################################################
            elif SVM_IG:
                st.subheader('SKENARIO 2 : INFORMATION GAIN :')

                data = pd.read_csv('https://raw.githubusercontent.com/SanS2A/dataset/main/nonSMOTE.csv')
                data = data.drop(['Unnamed: 0'], axis=1)
                # Memisahkan fitur dan label
                X = data.drop('stroke', axis=1)
                y = data['stroke']
                # Menghitung Information Gain untuk setiap fitur dengan random_state untuk hasil yang konsisten
                info_gain = mutual_info_classif(X, y, random_state=13)
                # Membuat DataFrame untuk menampilkan hasil
                info_gain_df = pd.DataFrame({'Fitur': X.columns, 'Information Gain': info_gain})
                info_gain_df = info_gain_df.sort_values(by='Information Gain', ascending=False)
                # Menampilkan tabel hasil Information Gain
                info_gain_df['Rank'] = range(1, len(info_gain_df) + 1)
                st.write("Peringkat Information Gain (dari besar ke kecil) untuk semua fitur :")
                st.write(info_gain_df)

                # Memisahkan dataset menjadi data pelatihan dan pengujian
                x_train, x_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
                # Membuat objek k_fold
                k_fold = KFold(n_splits=5)
                # Mendefinisikan parameter yang akan diuji
                C = 100
                degree = 4

                # Untuk menyimpan hasil cross-validation
                results = []
                # Loop untuk mencoba berbagai kombinasi fitur mulai dari 5 hingga semua fitur
                for K in range(5, len(info_gain_df) + 1):
                    selected_features = info_gain_df['Fitur'][:K]

                    # Filter matriks fitur berdasarkan fitur yang dipilih
                    X_selected = X[selected_features]

                    # Melakukan cross-validation dengan parameter yang ditentukan
                    svm_poly = SVC(kernel='poly', C=C, degree=degree)
                    scores = cross_val_score(svm_poly, x_train[selected_features], y_train, cv=k_fold, scoring='accuracy')

                    best_score = max(scores)
                    best_fold = scores.tolist().index(best_score) + 1  # +1 karena indeks dimulai dari 0

                    results.append({
                        'num_features': K,
                        'features': selected_features,
                        'params': {'C': C, 'degree': degree},
                        'scores': scores,
                        'best_score': best_score,
                        'best_fold': best_fold,
                    })

                    # Menampilkan hasil akurasi dengan 5-fold untuk jumlah fitur saat ini
                    st.write(f"Jumlah fitur : {K}")
                    st.write(f"Hyperparameter : C = {C}, degree = {degree}")
                    st.write("Detail Akurasi Tiap Fold : ")
                    for j, score in enumerate(scores):
                        st.write(f"Fold ke - {j+1} : {score}")
                    st.write()


################################################ SMOTE + SVM
            elif SVM_SMOTE:
                st.subheader('SKENARIO 3 : SVM + SMOTE')
                st.subheader('SVM kernel RBF')
                # Menghilangkan kolom 'stroke' dan membuat variabel target
                data = pd.read_csv('https://raw.githubusercontent.com/SanS2A/dataset/main/SMOTE.csv')
                data = data.drop(['Unnamed: 0'], axis=1)
                X = data.drop('stroke', axis=1)
                y = data['stroke']

                # Memisahkan dataset menjadi data pelatihan dan pengujian
                x_train, x_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

                # Membuat objek k_fold
                k_fold = KFold(n_splits=5)
                # Menentukan parameter yang akan diuji
                params = {
                    'C': [0.1, 1, 10, 100],
                    'gamma': [0.01, 0.1, 1, 10, 100]
                }
                # Variasi parameter
                param_combinations = [(C, gamma) for C in params['C'] for gamma in params['gamma']]

                # Untuk menyimpan hasil cross-validation
                results = []
                # Melakukan cross-validation untuk setiap kombinasi parameter
                for i, (C, gamma) in enumerate(param_combinations):
                    # Inisialisasi model SVM dengan kernel polynomial
                    svm_rbf = SVC(kernel='rbf', C=C, gamma=gamma)

                    # Melakukan cross-validation
                    scores = cross_val_score(svm_rbf, x_train, y_train, cv=k_fold, scoring='accuracy')

                    # Menyimpan hasil iterasi
                    results.append({
                        'params': {'C': C, 'gamma': gamma},
                        'scores': scores,
                        'iteration': i+1  # Menyimpan nomor iterasi
                    })
                    # Menampilkan hasil akurasi dengan 5-fold pada tiap iterasi
                    st.write(f"Iterasi ke - {i+1} :")
                    st.write(f"Hyperparameter : C = {C}, gamma = {gamma}")
                    st.write("Detail Akurasi Tiap Fold :")
                    for j, score in enumerate(scores):
                        st.write(f"Fold ke - {j+1} : {score}")
                    st.write()

               # Menentukan model terbaik berdasarkan akurasi fold tertinggi
                best_result = max(results, key=lambda x: max(x['scores']))
                best_params = best_result['params']
                best_iteration = best_result['iteration']
                best_fold_index = best_result['scores'].tolist().index(max(best_result['scores'])) + 1
                st.write(f"Model terbaik pada iterasi ke - {best_iteration} : C = {best_params['C']}, gamma = {best_params['gamma']}")
                st.write(f"Akurasi fold tertinggi pada fold ke - {best_fold_index} : {max(best_result['scores'])}")

################ SMOTE
                st.subheader('SVM kernel Polynomial')

                # Menentukan parameter yang akan diuji
                params = {
                    'C': [0.1, 1, 10, 100],
                    'degree': [1, 2, 3, 4, 5]
                }
                # Variasi parameter
                param_combinations = [(C, degree) for C in params['C'] for degree in params['degree']]

                # Untuk menyimpan hasil cross-validation
                results = []
                # Melakukan cross-validation untuk setiap kombinasi parameter
                for i, (C, degree) in enumerate(param_combinations):
                    # Inisialisasi model SVM dengan kernel polynomial
                    svm_poly = SVC(kernel='poly', C=C, degree=degree)

                    # Melakukan cross-validation
                    scores = cross_val_score(svm_poly, x_train, y_train, cv=k_fold, scoring='accuracy')

                    # Menyimpan hasil iterasi
                    results.append({
                        'params': {'C': C, 'degree': degree},
                        'scores': scores,
                        'iteration': i+1  # Menyimpan nomor iterasi
                    })
                    # Menampilkan hasil akurasi dengan 5-fold pada tiap iterasi
                    st.write(f"Iterasi ke - {i+1} : ")
                    st.write(f"Hyperparameter : C = {C}, degree = {degree}")
                    st.write("Detail Akurasi Tiap Fold : ")
                    for j, score in enumerate(scores):
                        st.write(f"Fold ke - {j+1} : {score}")
                    st.write("")

                # Menentukan model terbaik berdasarkan akurasi fold tertinggi
                best_result = max(results, key=lambda x: max(x['scores']))
                best_params = best_result['params']
                best_iteration = best_result['iteration']
                best_fold_index = best_result['scores'].tolist().index(max(best_result['scores'])) + 1
                st.write(f"Model terbaik pada iterasi ke - {best_iteration} : C = {best_params['C']}, degree = {best_params['degree']}")
                st.write(f"Akurasi fold tertinggi pada fold ke - {best_fold_index} : {max(best_result['scores'])}")


################################################ SMOTE + IG + SVM          
            elif SVM_SMOTE_IG:
                st.subheader('SKENARIO 4 : SVM + SMOTE + INFORMATION GAIN :')

                data = pd.read_csv('https://raw.githubusercontent.com/SanS2A/dataset/main/SMOTE.csv')
                data = data.drop(['Unnamed: 0'], axis=1)
                # Memisahkan fitur dan label
                X = data.drop('stroke', axis=1)
                y = data['stroke']
                # Menghitung Information Gain untuk setiap fitur dengan random_state untuk hasil yang konsisten
                info_gain = mutual_info_classif(X, y, random_state=13)
                # Membuat DataFrame untuk menampilkan hasil
                info_gain_df = pd.DataFrame({'Fitur': X.columns, 'Information Gain': info_gain})
                info_gain_df = info_gain_df.sort_values(by='Information Gain', ascending=False)
                # Menampilkan tabel hasil Information Gain
                info_gain_df['Rank'] = range(1, len(info_gain_df) + 1)
                st.write("Peringkat Information Gain (dari besar ke kecil) untuk semua fitur :")
                st.write(info_gain_df)

                # Memisahkan dataset menjadi data pelatihan dan pengujian
                x_train, x_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
                # Membuat objek k_fold
                k_fold = KFold(n_splits=5)
                # Mendefinisikan parameter yang akan diuji
                C = 10
                gamma = 100

                # Untuk menyimpan hasil cross-validation
                results = []
                # Loop untuk mencoba berbagai kombinasi fitur mulai dari 5 hingga semua fitur
                for K in range(5, len(info_gain_df) + 1):
                    selected_features = info_gain_df['Fitur'][:K]

                    # Filter matriks fitur berdasarkan fitur yang dipilih
                    X_selected = X[selected_features]

                    # Melakukan cross-validation dengan parameter yang ditentukan
                    svm_rbf = SVC(kernel='rbf', C=C, gamma=gamma)
                    scores = cross_val_score(svm_rbf, x_train[selected_features], y_train, cv=k_fold, scoring='accuracy')

                    best_score = max(scores)
                    best_fold = scores.tolist().index(best_score) + 1  # +1 karena indeks dimulai dari 0

                    results.append({
                        'num_features': K,
                        'features': selected_features,
                        'params': {'C': C, 'gamma': gamma},
                        'scores': scores,
                        'best_score': best_score,
                        'best_fold': best_fold,
                    })

                    # Menampilkan hasil akurasi dengan 5-fold untuk jumlah fitur saat ini
                    st.write(f"Jumlah fitur : {K}")
                    st.write(f"Hyperparameter : C = {C}, gamma = {gamma}")
                    st.write("Detail Akurasi Tiap Fold : ")
                    for j, score in enumerate(scores):
                        st.write(f"Fold ke - {j+1} : {score}")
                    st.write()
               
                    
################################################      
    elif selected == "Implementation":
            st.write("Pilihlah model untuk melakukan klasifikasi:")
            option = st.radio("Pilih Metode:", ['SVM', 'SVM_SMOTE'])

            with open('best_model_skenario_1.pkl', 'rb') as f:
                best_model_skenario_1 = pickle.load(f)
            with open('best_model_skenario_3.pkl', 'rb') as f:
                best_model_skenario_3 = pickle.load(f)
            # with open('scaler.pkl', 'rb') as f:
            #     scaler = pickle.load(f)

            if option == 'SVM':
                with st.form(key='SVM'):
                    st.subheader('Masukkan Data Anda')
                    gender = st.radio("Gender", ('Male', 'Female'))
                    gender = 1 if gender == "Male" else 0

                    age = st.number_input('Masukkan Umur Pasien')

                    hypertension = st.radio("Hypertency", ('Yes','No'))
                    hypertension = 1 if hypertension == "Yes" else 0

                    heart_disease = st.radio("Heart Disease", ('Yes','No'))
                    heart_disease = 1 if heart_disease == "Yes" else 0

                    ever_married = st.radio("Ever Married", ('Yes','No'))
                    ever_married = 1 if ever_married == "Yes" else 0

                    work_type = st.selectbox('Select a Work Type', options=['Govt_job', 'Never_worked', 'Private', 'Self_employed', 'childern'])
                    work_type_dict = {'Govt_job': 0, 'Never_worked': 1, 'Private': 2, 'Self_employed': 3, 'childern': 4}
                    work_type = work_type_dict[work_type]

                    Residence_type = st.radio("Residence Type", ('Rural', 'Urban'))
                    Residence_type = 0 if Residence_type == "Rural" else 1

                    avg_glucose_level = st.number_input('Average Glucose Level')
                    bmi = st.number_input('BMI')

                    smoking_status = st.selectbox('Select a smoking status', options=['Unknown', 'Formerly smoked', 'never smoked', 'smokes'])
                    smoking_status_dict = {'Unknown': 0, 'Formerly smoked': 1, 'never smoked': 2, 'smokes': 3}
                    smoking_status = smoking_status_dict[smoking_status]

                    df = pd.read_csv('data_.csv')
                    st.dataframe(df)
                    x = df.drop(columns=['stroke'])
                    x_normalized = x.copy() 
                            
                    #Normalisasi data input
                    df_min_bmi = x_normalized['bmi'].min()
                    df_max_bmi =  x_normalized['bmi'].max()
                            
                    df_min_age = x_normalized['age'].min()
                    df_max_age = x_normalized['age'].max()
                                    
                    df_min_avg = x_normalized['avg_glucose_level'].min()
                    df_max_avg = x_normalized['avg_glucose_level'] .max()
                            
                    # Make a copy of x to keep the original data
                    age_norm = float((age - df_min_age) / (df_max_age - df_min_age))
                    avg_norm = float((avg_glucose_level - df_min_avg) / (df_max_avg - df_min_avg))
                    bmi_norm = float((bmi - df_min_bmi) / (df_max_bmi - df_min_bmi))
                                
                    inputs = np.array([[gender, age_norm, hypertension, heart_disease, ever_married, work_type, Residence_type, avg_norm, bmi_norm, smoking_status]])

                    st.subheader('Hasil Klasifikasi dengan SVM')
                    cek_hasil = st.form_submit_button("Cek Hasil Klasifikasi")
                    if cek_hasil:
                        st.write(inputs)

                        # Melakukan prediksi
                        input_pred = best_model_skenario_1.predict(inputs)
                        
                        st.subheader('Hasil Prediksi')
                        st.write(input_pred)
                        if input_pred == 1:
                            st.error('Anda Terkena Stroke')
                        else:
                            st.success('Anda tidak terkena Stroke')
                                        
            else:
                with st.form(key='SVM_SMOTE'):
                    st.subheader('Masukkan Data Anda')

                    gender = st.radio("Gender", ('Male', 'Female'))
                    gender = 1 if gender == "Male" else 0

                    age = st.number_input('Masukkan Umur Pasien')

                    hypertension = st.radio("Hypertency : 0 = No, 1 = Yes", ('Yes', 'No'))
                    hypertension = 1 if hypertension == "Yes" else 0

                    heart_disease = st.radio("Heart Disease : 0 = No, 1 = Yes", ('Yes', 'No'))
                    heart_disease = 1 if heart_disease == "Yes" else 0

                    ever_married = st.radio("Ever Married", ('Yes', 'No'))
                    ever_married = 1 if ever_married == "Yes" else 0

                    work_type = st.selectbox('Select a Work Type', options=['Govt_job', 'Never_worked', 'Private', 'Self_employed', 'childern'])
                    work_type_dict = {'Govt_job': 0, 'Never_worked': 1, 'Private': 2, 'Self_employed': 3, 'childern': 4}
                    work_type = work_type_dict[work_type]

                    Residence_type = st.radio("Residence Type", ('Rural', 'Urban'))
                    Residence_type = 0 if Residence_type == "Rural" else 1

                    avg_glucose_level = st.number_input('Average Glucose Level')
                    bmi = st.number_input('BMI')

                    smoking_status = st.selectbox('Select a smoking status', options=['Unknown', 'Formerly smoked', 'never smoked', 'smokes'])
                    smoking_status_dict = {'Unknown': 0, 'Formerly smoked': 1, 'never smoked': 2, 'smokes': 3}
                    smoking_status = smoking_status_dict[smoking_status]

                    df = pd.read_csv('data_.csv')
                    # st.dataframe(df)
                    x = df.drop(columns=['stroke'])
                    x_normalized = x.copy() 
                            
                    # Normalisasi data input
                    df_min_bmi = x_normalized['bmi'].min()
                    df_max_bmi =  x_normalized['bmi'].max()
                            
                    df_min_age = x_normalized['age'].min()
                    df_max_age = x_normalized['age'].max()
                                    
                    df_min_avg = x_normalized['avg_glucose_level'].min()
                    df_max_avg = x_normalized['avg_glucose_level'] .max()
                            
                    # Make a copy of x to keep the original data
                    age_norm = float((age - df_min_age) / (df_max_age - df_min_age))
                    avg_norm = float((avg_glucose_level - df_min_avg) / (df_max_avg - df_min_avg))
                    bmi_norm = float((bmi - df_min_bmi) / (df_max_bmi - df_min_bmi))
                                      
                    inputs = np.array([[gender, age_norm, hypertension, heart_disease, ever_married, work_type, Residence_type, avg_norm, bmi_norm, smoking_status]])

                    st.subheader('Hasil Klasifikasi dengan SVM')
                    cek_hasil = st.form_submit_button("Cek Hasil Klasifikasi")
                    if cek_hasil:
                        st.write(inputs)

                        # Melakukan prediksi
                        input_pred = best_model_skenario_3.predict(inputs)

                        st.subheader('Hasil Prediksi')
                        st.write(input_pred)
                        if input_pred == 1:
                            st.error('Anda Terkena Stroke')
                        else:
                            st.success('Anda tidak terkena Stroke')        