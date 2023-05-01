# Laporan Proyek Machine Learning - Sylla Ayu Kusumahati
## Domain Proyek

Domain proyek yang dipilih dalam proyek machine learning ini adalah mengenai **telekomunikasi** dengan judul proyek "Prediksi Data Churn pada Pelanggan Telco".

- Latar Belakang
  Sebagai makhluk sosial komunikasi merupakan hal yang penting. Komunikasi yang semakin dibutuhkan ini membuat perusahaan telekomunikasi juga semakin banyak berkembang. Hal ini membuat perusahaan harus terus melakukan berbagai inovasi untuk mendorong persaingan usaha yang sangat ketat. Terbukanya persaingan ini menjadi tantangan sangat penting bagi para perusahaan.

  Dengan adanya komunikasi yang mudah dan banyaknya perusahaan telekomunikasi membuat pelanggan berhak menentukan berlangganan pada penyedia jasa yang ingin digunakan. Hal ini membuat grafik Customer Churn naik turun.  Customer Churn, juga dikenal sebagai pergantian pelanggan, adalah hilangnya pelanggan dengan berganti pada perusahaan telekomunikasi lainnya. Adapun alasan terbesar karena tertarik dengan penawaran kompetitor. 

  Dalam persaingan pasar saat ini mayoritas pelanggan menginginkan produk yang sesuai kebutuhan, dan mendapatkan pelayanan yang lebih baik dengan harga lebih murah. Penting bagi perusahaan telekomunikasi untuk memperhatikan Churn. Karena mendapatkan pelanggan baru lebih membutuhkan biaya berlipat-lipat dari pada mempertahankan pelanggan lama. Sehingga mengurangi kerugian yang bisa terjadi pada perusahaan.

  Dengan menggunakan prediksi churn dapat memahami perilaku konsumen dan pada gilirannya memprediksi asosiasi pelanggan apakah mereka akan meninggalkan perusahaan atau tidak. Untuk prediksi ini dibutuhkan suatu metode *machine learning* yaitu Klasifikasi untuk melakukan prediksi dengan jumlah data yang lumayan besar.
    
 
    
## Business Understanding
Perusahaan telekomunikasi adalah penyedia layanan yang sangat dibutuhkan oleh masyrakat saat ini. Dengan banyaknya perusahaan telekomunikasi ini pasti akan ada banyak persaingan dalam mendapatkan pelanggan tetap. Menurut Lu & Ph (2002), Pada persaingan pasar saat ini, laju Churn pada perusahaan telekomunikasi cukup besar yaitu 30-35% per tahun dari total pelanggan.

Sistem Telco adalah solusi terdepan di pasar yang memungkinkan penyedia layanan untuk membuat dan mengoperasikan jaringan cerdas berkualitas tinggi, terjamin layanan, tingkat operator, dan cerdas. Mereka membawa lebih dari 40 tahun pengalaman untuk desain dan pengembangan solusi komunikasi jaringan telekomunikasi canggih berkinerja tinggi.

Telco menyediakan kemampuan untuk diferensiasi layanan yang memungkinkan bentuk-bentuk baru produksi pendapatan, memaksimalkan profitabilitas jaringan. Penyedia layanan, besar dan kecil, bergantung pada penyampaian solusi canggih kami yang konsisten, memungkinkan mereka untuk tetap berada di depan krisis kapasitas sambil menjaga total biaya kepemilikan tetap minimum. [[1]](https://www.telco.com/about-us/)

Perusahaan ingin mempertahankan pelanggan lama alih-alih mendapatkan pelanggan baru. Karena seperti pengalaman yang ada, untuk mendapatkan pelanggan baru membutuhkan biaya 5-10 kali lipat dibanding jika mempertahankan pelanggan lama. Dan hal itu akan membuat kerugian pada perusahaan tersebut.

Oleh karena itu, penting bagi sebuah perusahaan telekomunikasi mengetahui dan dapat memprediksi churn  mereka. Prediksi akan digunakan untuk memaksimalkan kinerja yang harus diberikan kepada pelangga sebelum mereka berpindah perusahaan lain. Tentunya untuk menghindari kerugian dan memperoleh profit.
### Problem Statements
Berdasarkan kondisi yang telah diuraikan sebelumnya, perusahaan akan mengembangkan sebuah sistem prediksi churn untuk menjawab permasalahan berikut.
- Dari serangkaian fitur yang ada, fitur apa yang paling perpengaruh terhadap churn?
- Apakah churn meningkat atau menurun?

### Goals
Untuk menjawab pertanyaan tersebut, akan dibuat predictive modeling  dengan tujuan atau goals sebagai berikut:
- Mengetahui fitur yang paling berkorelasi dengan churn.
- Membuat model *machine learning* untuk mengklasifikasikan data churn dan tidak churn memiliki tingkat akurasi 70%.

### Solution statements
Solusi yang dapat dilakukan untuk memenuhi tujuan dari proyek ini diantaranya :
- Untuk pra-pemrosesan data dapat dilakukan beberapa teknik, diantaranya :
    * Mengisi data yang kosong dengan nilai rata rata atau **(mean substitution)**
    * Melakukan **pembagian dataset** menjadi dua bagian dengan rasio 80% untuk data latih dan 20% untuk data uji
    * Melakukan **proses encoding** fitur kategori, teknik yang dilakukan adalah LabelEncoder
    * Melakukan **standardisasi** data pada semua fitur data.
    Poin pra-pemrosesan data akan dibahas lebih lanjut pada bagian *Data Preparation.*

- Untuk pembuatan model dipilih penggunaan model dengan algoritma **K-Nearest Neighbor**, **RandomForest**, **Ada Boost**. Dan nantinya **K-Nearest Neighbor** akan dipilih sebagai model *baseline*. Algoritma tersebut dipilih karena mudah digunakan dan juga cocok untuk kasus klasifikasi. Berikut adalah penjelasan mengenai masing-masing algoritma :
  1.  **K-Nearest Neighbor** adalah algoritma pembelajaran mesin terawasi yang sederhana dan mudah diterapkan yang dapat digunakan untuk menyelesaikan masalah klasifikasi. Cara kerja algoritma ini adalah sebagai berikut (diterjemahkan dari [[2]](https://towardsdatascience.com/machine-learning-basics-with-the-k-nearest-neighbors-algorithm-6a6e71d01761)):  
        * Muat datanya
        * Inisialisasi nilai K (banyak tetangga/kelompok)
        * Pada setiap datanya :
            * Hitung euclidian distance antara contoh kueri dan contoh yang ada pada data tersebut dengan rumus seperti berikut ini : ![Rumus Euclidian Distance](https://raw.githubusercontent.com/syllaayu/FotoSya/main/FOTO/ss.png)
            * Tambahkan jarak dan urutan dari contoh pada koleksi yang berururutan
        * Pilih entri K paling awal pada koleksi yang berurutan
        * Dapatkan label dari dari entri K yang dipilih
        * Apabila kasus regresi, kembalikan nilai rata-ratanya. Apabila kasus klasifikasi, kembalikan labelnya.
    
     Selain itu, berikut ini merupakan kelebihan dan kekurangan algoritma dari K-Nearest Neighbor (diterjemahkan dari [[2]](https://towardsdatascience.com/machine-learning-basics-with-the-k-nearest-neighbors-algorithm-6a6e71d01761)):
    - **Kelebihan**
      * Algoritmanya mudah digunakan dan sederhana
      * Algoritmanya sangat fleksibel, dapat diimplementasikan pada kasus klasifikasi, regresi dan pencarian
    - **Kekurangan** :
    Algoritma menjadi lebih lambat secara signifikan karena jumlah contoh dan/atau prediktor/variabel yang meningkat.

2. **Random Forest** adalah salah satu algoritma supervised learning. Ia dapat digunakan untuk menyelesaikan masalah klasifikasi dan regresi. Cara kerja algoritma berdasarkan wikipedia [[3]](https://id.wikipedia.org/wiki/Random_forest) adalah sebagai berikut :
     - Memecah data sampel yang ada kedalam *decision tree* secara acak. Untuk vektor acak dinyatakan dengan X = ( X1 , ...,  Xp ), X mewakili input yang bernilai nyata dan variabel acak Y mewakili respons bernilai riil.
     - Setelah pohon terbentuk,maka akan dilakukan voting pada setiap kelas dari data sampel. 
     - Kemudian, mengkombinasikan vote dari setiap kelas kemudian diambil vote yang paling banyak.

     Selain itu, berikut ini merupakan kelebihan dan kekurangan algoritma dari Random Forest (berdasarkan artikel dari [[3]](http://learningbox.coffeecup.com/05_2_randomforest.html)):
     * **Kelebihan**:
       * Menghasilkan eror yang lebih rendah.
       * Memberikan hasil yang bagus dalam klasifikasi.
       * Metode yang efektif untuk mengestimasi hilangnya data.
     * **Kekurangan**:
       * Waktu pemrosesan yang lama karena menggunakan data yang banyak dan membangun model tree yang banyak pula untuk membentuk random trees karena menggunakan single processor.
       * Interpretasi yang sulit dan membutuhkan mode penyetelan yang tepat untuk data. 
  
    3. **Ada Boost** adalah salah satu metode adaptive boosting yang terkenal algoritma ini bertujuan untuk meningkatkan performa atau akurasi prediksi. Cara kerja algoritma ini adalah sebagai berikut (diterjemahkan dari [[4]](https://towardsdatascience.com/machine-learning-part-17-boosting-algorithms-adaboost-in-python-d00faac6c464)):
        * Awalnya, semua kasus dalam data latih memiliki weight atau bobot yang sama.
        * Bangun pohon keputusan dengan setiap fitur, klasifikasikan data dan evaluasi hasilnya
        * Hitung signifikansi pohon dalam klasifikasi akhir. Rumus lanjutan untuk menghitung jumlah ialah :
        ![image](https://raw.githubusercontent.com/syllaayu/FotoSya/main/FOTO/jjj.png)
        * Perbarui bobot sampel sehingga pohon keputusan berikutnya akan memperhitungkan kesalahan yang dibuat oleh pohon keputusan sebelumnya
        * Bentuk kumpulan data baru
        * Ulangi langkah 2 sampai 5 sampai jumlah iterasi sama dengan jumlah yang ditentukan oleh hyperparameter (yaitu jumlah estimator)
        * Gunakan hutan pohon keputusan untuk membuat prediksi pada data di luar set pelatihan
    - **Kelebihan**
        - AdaBoost mudah diimplementasikan.
        - Ini secara berulang mengoreksi kesalahan pengklasifikasi lemah dan meningkatkan akurasi dengan menggabungkan peserta didik yang lemah.
    - **Kekurangan**
        - AdaBoost sensitif terhadap data derau.
        - Ini sangat dipengaruhi oleh pencilan karena mencoba menyesuaikan setiap poin dengan sempurna.
            
* Untuk melakukan peningkatan performa model baseline tersebut dikembangkan dengan pengaturan hyperparameter [GridSearchCV](https://scikit-learn.org/0.24/modules/generated/sklearn.model_selection.GridSearchCV.html). GridSearchCV adalah metode yang efektif untuk menyesuaikan parameter dalam supervised learning dan meningkatkan performa generalisasi model.

Cara kerja algoritma ini adalah sebagai berikut (diterjemahkan dari [[5]](https://medium.com/@adiptamartulandi/tuning-hyperparameters-logistic-regression-menggunakan-grid-search-ucupstory-fb1ab9db082a)):
- Mengkombinasikan nilai yang kita masukan pada Hyperparameters.
   Contohnya adalah ketika kita ingin mencari kombinasi dari Hyperparameters A = [1,2] dan B=[3,4] maka Grid Search akan mencari seluruh kombinasi dari A dan B yaitu [1,3],[1,4],[2,3],[2,4] dan memilih kombinasi terbaik berdasarkan nilai dari CV Score yang paling tinggi
    
- Kelebihan *Grid Search Cross Validation* mempermudah kita dalam menguji coba setiap model dan parameter model machine learning tanpa harus mencoba melakukan validasi secara manual satu persatu. Penerapan *Grid Search Cross Validation* yang disandingkan dengan pemahaman dan intuisi yang baik terkait model *machine learning* dan data yang digunakan akan memberikan hasil prediksi yang akurat dan optimal.

    
Poin pra-pemrosesan data akan dibahas lebih lanjut pada bagian Data Preparation.

# Data Understanding
![image](https://raw.githubusercontent.com/syllaayu/FotoSya/main/FOTO/ddd.png)

Informasi dataset dapat dilihat pada tabel dibawah ini :
Jenis | Keterangan
------------ | -------------
Sumber | 	[Kaggle Dataset : Telco Customer Churn](https://www.kaggle.com/blastchar/telco-customer-churn)
Lisensi | Data files © Original Authors
Kategori | Telekomunikasi
Rating Penggunaan | 8.8 
Jenis dan Ukuran Berkas | CSV (978 kB)

 Dataset ini berguna sebagai prediksi perilaku untuk mempertahankan pelanggan. Juga dapat menganalisa semua data pelanggan yang relevan dan mengembangkan retensi pelanggan yang terfokus. Setiap baris pada dataset mewakili pelanggan, setiap kolom berisi atribut pelanggan. Dataset ini memiliki 7043 baris data mentah milik pelanggan dan 21 fitur. Terdapat 3 buah fitur numerik. Dan 17 fitur non-numerik. Sedangkan, TotalChargers merupakan fitur target.
 
 Untuk penjelasan mengenai variabel-variable pada data telco churn customer dapat dilihat pada poin-poin berikut :
1. `customerID`  ID Pelanggan
2. `gender` jenis kelamin (perempuan, laki-laki)
3. `SeniorCitizen` Apakah pelanggan adalah warga senior atau tidak (1, 0)
4. `PartnerWhether` pelanggan memiliki mitra atau tidak (Ya, Tidak)
5. `Dependents` Apakah pelanggan memiliki tanggungan atau tidak (Ya, Tidak)
6. `tenure` Jumlah bulan pelanggan telah tinggal di perusahaan
7. `PhoneService` Apakah pelanggan memiliki layanan telepon atau tidak (Ya, Tidak)
8. `MultipleLines` Apakah pelanggan memiliki banyak saluran atau tidak (Ya, Tidak, Tidak ada layanan telepon)
9. `InternetService` Penyedia layanan internet pelanggan (DSL, Fiber optic, No)
10. `OnlineSecurity` Apakah pelanggan memiliki keamanan online atau tidak (Ya, Tidak, Tidak ada layanan internet)
11. `OnlineBackup` Apakah pelanggan memiliki keamanan online atau tidak (Ya, Tidak, Tidak ada layanan internet)
12. `DeviceProtection` Apakah pelanggan memiliki perlindungan perangkat atau tidak (Ya, Tidak, Tidak ada layanan internet)
13. `TechSupport` Apakah pelanggan memiliki dukungan teknis atau tidak (Ya, Tidak, Tidak ada layanan internet
14. `StreamingTV` Apakah pelanggan memiliki TV streaming atau tidak (Ya, Tidak, Tidak ada layanan internet)
15. `StreamingMovies` Apakah pelanggan memiliki streaming film atau tidak (Ya, Tidak, Tidak ada layanan internet)
16. `Contract` Jangka waktu kontrak pelanggan (Bulan-ke-bulan, Satu tahun, Dua tahun)
17. `PaperlessBilling` Apakah pelanggan memiliki paperless billing atau tidak (Ya, Tidak)
18. `PaymentMethod` Metode pembayaran pelanggan (Cek elektronik, Cek pos, Transfer bank (otomatis), Kartu kredit (otomatis))
19. `MonthlyCharges` Jumlah yang dibebankan ke pelanggan setiap bulan
20. `TotalCharges` Jumlah total yang dibebankan kepada pelanggan
21. `Churn` Apakah pelanggan churn atau tidak (Ya atau Tidak)

Berikut visualisasi pada presentase ` Churn` :
![images](https://raw.githubusercontent.com/syllaayu/FotoSya/main/FOTO/vis1.png)

![images](https://raw.githubusercontent.com/syllaayu/FotoSya/main/FOTO/vis2.png)

Kemudian terdapat juga visualisasi data untuk kolom dengan fitur numerik seperti pada gambar dibawah ini :
Distribusi fitur numerik pada **tenure**
![image](https://raw.githubusercontent.com/syllaayu/FotoSya/main/FOTO/dis1.png)

Distribusi fitur numerik pada **MonthlyCharges**
![image](https://raw.githubusercontent.com/syllaayu/FotoSya/main/FOTO/dis2.png)

Distribusi fitur numerik pada **TotalCharges**
![image](https://raw.githubusercontent.com/syllaayu/FotoSya/main/FOTO/dis3.png)

Distribusi data pada kolom dengan fitur numerik dengan fungsi pairplot()
![image](https://raw.githubusercontent.com/syllaayu/FotoSya/main/FOTO/plot.png)

Terakhir visualisasi data untuk kolom dengan fitur kategori seperti pada gambar dibawah ini :
![image](https://raw.githubusercontent.com/syllaayu/FotoSya/main/FOTO/plot1.png)

![image](https://raw.githubusercontent.com/syllaayu/FotoSya/main/FOTO/plot2.png)

![image](https://raw.githubusercontent.com/syllaayu/FotoSya/main/FOTO/plot3.png)

![image](https://raw.githubusercontent.com/syllaayu/FotoSya/main/FOTO/plot4.png)

![image](https://raw.githubusercontent.com/syllaayu/FotoSya/main/FOTO/plot5.png)

![image](https://raw.githubusercontent.com/syllaayu/FotoSya/main/FOTO/plot6.png)

![image](https://raw.githubusercontent.com/syllaayu/FotoSya/main/FOTO/plot7.png)

![image](https://raw.githubusercontent.com/syllaayu/FotoSya/main/FOTO/plot8.png)

![image](https://raw.githubusercontent.com/syllaayu/FotoSya/main/FOTO/plot9.png)

![image](https://raw.githubusercontent.com/syllaayu/FotoSya/main/FOTO/plot10.png)

![image](https://raw.githubusercontent.com/syllaayu/FotoSya/main/FOTO/plot11.png)

![image](https://raw.githubusercontent.com/syllaayu/FotoSya/main/FOTO/plot12.png)

![image](https://raw.githubusercontent.com/syllaayu/FotoSya/main/FOTO/plot13.png)

![image](https://raw.githubusercontent.com/syllaayu/FotoSya/main/FOTO/plot14.png)

![image](https://raw.githubusercontent.com/syllaayu/FotoSya/main/FOTO/plot15.png)

# Data Preparation
Seperti yang sudah disebutkan sebelumnya pada bagian *Solution statements*, berikut adalah tahapan-tahapan dalam melakukan pra-pemrosesan data :
* Mengisi data yang kosong dengan nilai rata rata atau **(mean substitution)**
![image](https://raw.githubusercontent.com/syllaayu/FotoSya/main/FOTO/mean.png)
Menganggap data kosong sebagai data rata-rata, model tetap dapat memperoleh informasi dari data yang ada pada kolom lainnya. Proses yang dilakukan pertama-tama dengan cara mengambil nilai rata-rata dari kolom yang memiliki data kosong, kemudian memasukannya kepada setiap data kosong sebagai pengganti dari datanya. Semua proses tersebut dilakukan dengan slicing data dengan kondisi menggunakan pandas.

* Melakukan **proses encoding** fitur kategori
Teknik yang dilakukan adalah LabelEncode. Label encoding mengubah setiap nilai dalam kolom menjadi angka yang berurutan.

* Melakukan **pembagian dataset** menjadi dua bagian dengan rasio 80% untuk data latih dan 20% untuk data uji
Supaya performa model pada data sebenarnya dapat diuji, maka perlu melakukan pembagian menjadi dua bagian yakni pada data latih dan data uji dengan rasio 80:20. Data latih dilakukan sepenuhnya untuk melatih model, sedangkan data uji merupakan data yang belum pernah dilihat oleh model dan diharapkan model dapat memiliki performa yang sama baiknya pada data uji seperti pada data latih. Pembagian dataset dilakukan dengan modul train_test_split dari scikit-learn.

* Melakukan **standardisasi** data pada semua fitur data menggunakan StandardScaler.
Standardisasi adalah teknik transformasi yang paling umum digunakan dalam tahap persiapan pemodelan. StandardScaler melakukan proses standarisasi fitur dengan mengurangkan mean (nilai rata-rata) kemudian membaginya dengan standar deviasi untuk menggeser distribusi.  StandardScaler menghasilkan distribusi dengan standar deviasi sama dengan 1 dan mean sama dengan 0. Sekitar 68% dari nilai akan berada di antara -1 dan 1.  

# Modeling
Setelah melakukan pra-pemrosesan data yang baik pada tahap modeling akan dilakukan tiga hal, yakni tahap pelatihan model KNN, Random Forest dan AdaBoost, setelah itu tahap pembuatan model *baseline* dan pembuatan model yang dikembangkan.

* Melatih algoritma **K-Nearest Neighbor**
Pada model KNN ini menggunakan k = 10 tetangga dan metric Euclidean untuk mengukur jarak antara titik. Pada tahap ini kita hanya melatih data training dan menyimpan data testing untuk dilatih di tahap evaluasi. 

* Melatih algoritma **Random Forest**
Pada tahap Random Forest menerapkan algoritma pada dataset menggunakan library scikit-learn. Setelah model ini dijalankan (run), disimpan terlebih dahulu hasilnya untuk tahap evaluasi nanti.

* Melatih model **AdaBoost**
Pada tahap ini menerapkan algoritma pada dataset menggunakan library scikit-learn. Dengan menambahkan estimators=50, learning_rate=0.05, random_state=55.


Metrik yang akan kita gunakan pada prediksi ini adalah MSE atau Mean Squared Error yang menghitung selisih rata-rata nilai sebenarnya dengan nilai prediksi. MSE didefinisikan dalam persamaan berikut :
![image](https://raw.githubusercontent.com/syllaayu/FotoSya/main/FOTO/12.jpeg)

Hasil evaluasi pada data latih dan data test adalah sebagai berikut :
![image](https://raw.githubusercontent.com/syllaayu/FotoSya/main/FOTO/121.png)

Untuk memudahkan, berikut plot metrik tersebut dengan bar chart.

![image](https://raw.githubusercontent.com/syllaayu/FotoSya/main/FOTO/11.png)
Dari gambar di atas, terlihat bahwa, model KNN memberikan nilai eror yang paling seimbang. Model inilah yang akan kita pilih sebagai model terbaik untuk melakukan prediksi churn.

Hasil prediksi menggunakan beberapa data pada data test :
![image](https://raw.githubusercontent.com/syllaayu/FotoSya/main/FOTO/13.png)
Terlihat bahwa prediksi dengan K-Nearest Neighbor (KNN) memberikan hasil yang bagus. 

* Model *baseline*
  Pada tahap ini saya membuat model dasar dengan menggunakan modul *scikit-learn* yakni **K-Nearest Neighbor** tanpa menggunakan parameter tambahan. Lalu melakukan prediksi kepada data ujinya.

* Model yang dikembangkan
  Kemudian setelah melihat kinerja model baseline, agar dapat bekerja lebih optimal lagi maka digunakan sebuah fungsi untuk mencari hyperparameter yang optimal dengan GridSearchCV. Setelah ditemukan yang optimal, kemudian hyperparameter tersebut diterapkan ke model baseline.

Hasilnya dapat dilihat seperti pada tabel berikut ini :
![image](https://raw.githubusercontent.com/syllaayu/FotoSya/main/FOTO/14.png)

Pada model baseline nilai akurasinya sedikit buruk. Namun setelah dilakukan pengaturan *hyperparamete*r, nilai akurasi pun meningkat. Walaupun akurasi tidak meningkat secara signifikan. Untuk membuktikannya, kedua model tersebut diuji pada data uji dan di visualisasikan pada *confussion matrix* seperti berikut.

*Confussion matrix* adalah salah satu teknik yang dapat digunakan untuk mengukur kinerja suatu model khusunya kasus klasifikasi (*supervised learning*) pada machine learning. Confusion matrix berbentuk tabel matriks yang menggambarkan kinerja model klasifikasi pada serangkaian data uji yang nilai sebenarnya diketahui.[[6]](https://ksnugroho.medium.com/confusion-matrix-untuk-evaluasi-model-pada-unsupervised-machine-learning-bc4b1ae9ae3f)

Gambar dibawah ini merupakan confusion matrix dengan 4 kombinasi nilai prediksi dan nilai aktual yang berbeda. Merupakan hasil dari Model baseline dan model yang dikembangkan.

* **Model baseline**
![image](https://raw.githubusercontent.com/syllaayu/FotoSya/main/FOTO/15.png)

Pada kasus model *baseline* Cunfusion Matrix ini menunjukkan bahwa :
- `True Positive(TP)` memprediksi terdapat 892 pelanggan benar tidak churn 
- `True Negative(TN)` memprediksi terdapat 184 pelanggan benar churn
- `False Positive(FP)` memprediksi terdapat 144 pelanggan positif tidak churn dan ternyata prediksi salah, ternyata pelanggan negatif churn 
- `False Negative(FN) memprediksi terdapat 189 pelanggan negatif Churn dan ternyata prediksi salah, ternyata pelanggan positif tidak churn
- Kemudian pada akurasi ini bila dihitung dengan rumus *accuracy* maka akan menghasilkan akurasi sebesar 76% seperti gambar :
![images](https://raw.githubusercontent.com/syllaayu/FotoSya/main/FOTO/akur.png)

* **Model yang dikembangkan**
![image](https://raw.githubusercontent.com/syllaayu/FotoSya/main/FOTO/16.png)

Pada kasus model yang dikembangankan Cunfusion Matrix ini menunjukkan bahwa :
- `True Positive(TP)` memprediksi menjadi sebesar 965 pelanggan benar tidak churn 
- `True Negative(TN)` memprediksi menjadi sebesar 119 pelanggan benar churn
- `False Positive(FP)` memprediksi menjadi sebesar 71 pelanggan positif tidak churn dan ternyata prediksi salah, ternyata pelanggan negatif churn 
- `False Negative(FN) memprediksi menjadi sebesar 254 pelanggan negatif Churn dan ternyata prediksi salah, ternyata pelanggan positif tidak churn
- Kemudian pada akurasi ini bila dihitung dengan rumus *accuracy* maka akan menghasilkan akurasi sebesar 77% seperti gambar :
![img](https://raw.githubusercontent.com/syllaayu/FotoSya/main/FOTO/akuras.png)

Dengan hasil diatas, maka model yang dikembangkan merupakan model yang dipilih untuk digunakan. Karena memberi akurasi terbaik pada sistem rekomendasi. 

# Evaluation
Pada proyek ini, model yang dibuat merupakan kasus regressi dan menggunakan metriks akurasi, f1-score, recall dan precision. Pada gambar dibawah ini ditampilkan kembali hasil pengukuran model yang dikembangkan dengan metriks akurasi, f1-score, recall dan precision.

![image](https://raw.githubusercontent.com/syllaayu/FotoSya/main/FOTO/17.png)

Berdasarkan terjemahan [[7]](https://developers.google.com/machine-learning/crash-course/classification/accuracy) berikut adalah pengertian dari *accuracy*, *precision*, *recall*, *f1-score* :

* Akurasi
Akurasi adalah salah satu metrik untuk mengevaluasi model klasifikasi. Akurasi memiliki definisi sebagai berikut:
![image](https://raw.githubusercontent.com/syllaayu/FotoSya/main/FOTO/2.png)
Untuk kasus ini akurasi dapat dihitung seperti gambar dibawah :
![image](https://raw.githubusercontent.com/syllaayu/FotoSya/main/FOTO/3.png)
    Kelebihan dari metriks ini adalah sering digunakan dalam kasus pembuatan model klasifikasi baik itu klasifikasi dua kelas, atau kategori. Kekurangan dari metrik ini adalah dapat bersifat 'menyesatkan' pada data yang tidak seimbang.

* *Precission*
Merupakan rasio prediksi benar positif dibandingkan dengan keseluruhan hasil yang diprediksi positf. Perhitungan *precission* seperti gambar dibawah : 
![image](https://raw.githubusercontent.com/syllaayu/FotoSya/main/FOTO/4.png)
Kelebihan dari metriks ini berfokus pada bagaimana performa (prediksi) model terhadap label data positif, kekurangannya metriks ini tidak memperhitungkan label negatifnya.

* *Recall* (Sensitifitas)
Merupakan rasio prediksi benar positif dibandingkan dengan keseluruhan hasil yang diprediksi positf.
![image](https://raw.githubusercontent.com/syllaayu/FotoSya/main/FOTO/5.png)
Kelebihan dari metriks ini menghitung bagian negatif dari prediksi label positif (tidak seperti precision). Tetapi kekurangannya ketika semua prediksi = 1 maka recall akan bernilai 1 (tidak memperhitungkan prediksi negatif).

* *F1 Score*
F1 Score merupakan perbandingan rata-rata presisi dan recall yang dibobotkan.
![image](https://raw.githubusercontent.com/syllaayu/FotoSya/main/FOTO/6.png)
Kelebihan dari metriks ini menutup semua kekurangan yang ada pada precision dan recall. Namun kekurangannya adalah f1-score tidak memperhitungkan hasil prediksi benar pada label negatif.

# Referensi
[[1]](https://repository.its.ac.id/83058/) Saputra, Faisal Dhio, *PREDIKSI CHURN DAN STRATEGI RETENSI PADA KASUS PERUSAHAAN TELEKOMUNIKASI*. Masters thesis, Institut Teknologi Sepuluh Nopember (2021). https://repository.its.ac.id/83058/
[[2]](https://towardsdatascience.com/machine-learning-basics-with-the-k-nearest-neighbors-algorithm-6a6e71d01761) Onel Harrison. (Sep 11, 2018). *Machine Learning Basics with the K-Nearest Neighbors Algorithm*. Medium. https://towardsdatascience.com/machine-learning-basics-with-the-k-nearest-neighbors-algorithm-6a6e71d01761
[[3]](https://id.wikipedia.org/wiki/Random_forest) Random forest. Wikipedia. https://id.wikipedia.org/wiki/Random_forest
[[4]](https://towardsdatascience.com/machine-learning-part-17-boosting-algorithms-adaboost-in-python-d00faac6c464) Cory Maklin. (May 16, 2019). *AdaBoost Classifier Example In Python*. Medium. https://towardsdatascience.com/machine-learning-part-17-boosting-algorithms-adaboost-in-python-d00faac6c464
[[5]](https://medium.com/@adiptamartulandi/tuning-hyperparameters-logistic-regression-menggunakan-grid-search-ucupstory-fb1ab9db082a) Adipta Martulandi. (Oct 6, 2019). *Tuning Hyperparameters Logistic Regression Menggunakan Grid Search #UcupStory*. Medium. https://medium.com/@adiptamartulandi/tuning-hyperparameters-logistic-regression-menggunakan-grid-search-ucupstory-fb1ab9db082a
[[6]](https://ksnugroho.medium.com/confusion-matrix-untuk-evaluasi-model-pada-unsupervised-machine-learning-bc4b1ae9ae3f) Kuncahyo Setyo Nugroho. (Nov 13, 2019). *Confusion Matrix untuk Evaluasi Model pada Supervised Learning*. Medium. https://ksnugroho.medium.com/confusion-matrix-untuk-evaluasi-model-pada-unsupervised-machine-learning-bc4b1ae9ae3f
[[7]](https://developers.google.com/machine-learning/crash-course/classification/accuracy) *Machine Learning Crash Course*. Google. https://developers.google.com/machine-learning/crash-course/classification/accuracy

