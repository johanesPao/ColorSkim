# ColorSkim Machine Learning AI
!!! quote "Matthew Quick, The Good Luck of Right Now"
    *The voice that navigated was definitely that of a machine, and yet you could tell that the machine was a woman, which hurt my mind a little. How can machines have genders? The machine also had an American accent. How can machines have nationalities? This can't be a good idea, making machines talk like real people, can it? Giving machines humanoid identities?*

```python
# import modul dasar⁡⁡
import os
import random
import gc #garbage collector
import io
# import pandas, numpy dan tensorflow⁡
import pandas as pd
import numpy as np
import tensorflow as tf

# import daftar device terdeteksi oleh tensorflow
from tensorflow.python.client.device_lib import list_local_devices

# import utilitas umum tensorflow
from tensorflow.config import run_functions_eagerly # type: ignore
from tensorflow.data.experimental import enable_debug_mode # type: ignore

# import pembuatan dataset
from sklearn.model_selection import train_test_split
"""
karena struktur objek dalam tf.data.Dataset, from_tensor_slices() 
dan zip tidak dapat dipanggil secara langsung dalam modul import
"""
from_tensor_slices = tf.data.Dataset.from_tensor_slices
zip = tf.data.Dataset.zip

# import preprocessing data
from sklearn.preprocessing import OneHotEncoder, LabelEncoder
import tensorflow_hub as hub


# import pipeline scikit untuk model_0
from sklearn.pipeline import Pipeline

# import layer neural network
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.naive_bayes import MultinomialNB
from tensorflow.keras.layers import Input, Conv1D, Dense, GlobalMaxPooling1D, Bidirectional, LSTM, Dropout # type: ignore
from tensorflow.keras.layers import Concatenate # type: ignore
from tensorflow.keras.layers import TextVectorization # type: ignore
from tensorflow.keras.layers import Embedding # type: ignore

# import fungsi loss dan optimizer
from tensorflow.keras.losses import BinaryCrossentropy # type: ignore
from tensorflow.keras.optimizers import Adam # type: ignore

# import model Functional API tensorflow
from tensorflow.keras import Model # type: ignore

# import callbacks untuk tensorflow
from tensorflow.keras.callbacks import ModelCheckpoint, EarlyStopping, ReduceLROnPlateau # type: ignore

# import model terbaik, metriks dan alat evaluasi
from sklearn.metrics import accuracy_score, precision_recall_fscore_support, confusion_matrix
from tensorflow.keras.models import load_model # type: ignore

# import grafik
import matplotlib.pyplot as plt
import seaborn as sns
from tensorflow.keras.utils import plot_model # type: ignore
from scipy.stats import binned_statistic

# import display untuk menampilkan dataframe berdasar settingan tertentu (situasional)
from IPython.display import display

# import library log untuk training
import wandb as wb
from wandb.keras import WandbCallback

# import kunci untuk login wandb
from rahasia import API_KEY_WANDB # type: ignore

# set output tensorflow
run_functions_eagerly(True)
enable_debug_mode()

# set matplotlib untuk menggunakan tampilan seaborn
sns.set()
```


```python
# cek ketersediaan GPU untuk modeling
# NVidia GeForce MX250 - office
# NVidia GeForce GTX1060 - home
list_local_devices()[1]
```




    name: "/device:GPU:0"
    device_type: "GPU"
    memory_limit: 1408103015
    locality {
      bus_id: 1
      links {
      }
    }
    incarnation: 4677246905546237580
    physical_device_desc: "device: 0, name: NVIDIA GeForce MX250, pci bus id: 0000:02:00.0, compute capability: 6.1"
    xla_global_id: 416903419



## Variabel Global


```python
DIR_MODEL_CHECKPOINT = 'colorskim_checkpoint'
# kita akan mengatur toleransi_es sebagai fraksi (fraksi_toleransi) tertentu dari jumlah total epoch
# dan toleransi_rlop sebagai toleransi_es dibagi dengan jumlah kesempatan (kesempatan_rlop)
# dilakukannya reduksi pada learning_rate 
EPOCHS = 1000
UKURAN_BATCH = 32
FRAKSI_TOLERANSI = 0.01
KESEMPATAN_RLOP = 5
TOLERANSI_ES = int(EPOCHS*FRAKSI_TOLERANSI)
TOLERANSI_RLOP = int(TOLERANSI_ES/KESEMPATAN_RLOP)
FRAKSI_REDUKSI_LR = 0.1
METRIK_MONITOR = 'val_accuracy'
RANDOM_STATE = 11
# untuk mencegah overfitting, kita akan memberikan ruang yang cukup besar 
# untuk test_data dan memperkecil porsi train_data dengan jumlah epoch
# yang besar sehingga model masih memiliki waktu untuk melakukan
# training pada train_data
RASIO_TEST_TRAIN = 0.2

# wandb init
wandb = {'proyek': 'ColorSkim',
         'user': 'jpao'}

# nama model
MODEL = ['model_0_multinomial_naive_bayes',
         'model_1_Conv1D_vektorisasi_embedding',
         'model_2_Conv1D_USE_embed',
         'model_3_quadbrid_embedding']
```

## Callbacks

Beberapa *callbacks* yang akan digunakan dalam proses *training* model diantaranya:

* `WandbCallback` - *Callback* ke [wandb.ai](https://wandb.ai) untuk mencatat log dari sesi *training* model.
* `ModelCheckpoint` - Untuk menyimpan model dengan *val_accuracy* terbaik dari seluruh *epoch* dalam *training* model.
* `EarlyStopping` (ES) - *Callback* ini digunakan untuk menghentikan proses *training* model jika selama beberapa *epoch* model tidak mengalami perbaikan pada metrik *val_accuracy*-nya. *Callback* ini juga digunakan bersama dengan `ReduceLROnPlateau` dimana *patience* ES > *patience* RLOP.
* `ReduceLROnPlateau` (RLOP) - *Callback* ini digunakan untuk memperkecil *learning_rate* dari model jika tidak mengalami perbaikan *val_accuracy* selama beberapa *epoch*.

*Patience* dari ES di-set lebih tinggi dari *patience* RLOP untuk memberikan kesempatan bagi RLOP untuk memperkecil *learning_rate* beberapa kali sebelum proses *training* model dihentikan oleh ES setelah tidak berhasil mendapatkan *val_accuracy* yang lebih baik selama beberapa *epoch*.


```python

# Login ke wandb
wb.login(key=API_KEY_WANDB)

# Pembuatan fungsi callback
def wandb_callback(data_training):
    return WandbCallback(save_model=False, # model akan disimpan menggunakan callback ModelCheckpoint
                         log_weights=True, # bobot akan disimpan untuk visualisasi di wandb
                         log_gradients=True, # gradient akan disimpan untuk visualisasi di wandb
                         training_data=data_training) 
def model_checkpoint(nama_model):
    return ModelCheckpoint(filepath=os.path.join(DIR_MODEL_CHECKPOINT, nama_model),
                           verbose=0,
                           monitor=METRIK_MONITOR,
                           save_best_only=True) # model dengan 'val_accuracy' terbaik akan disimpan
def early_stopping():
    return EarlyStopping(patience=TOLERANSI_ES,
                         monitor=METRIK_MONITOR)
def reduce_lr_on_plateau():
    return ReduceLROnPlateau(factor=FRAKSI_REDUKSI_LR, # pengurangan learning_rate diset sebesar 0.1 * learning_rate
                             patience=TOLERANSI_RLOP,
                             monitor=METRIK_MONITOR,
                             verbose=0)
```

    Failed to detect the name of this notebook, you can set it manually with the WANDB_NOTEBOOK_NAME environment variable to enable code saving.
    [34m[1mwandb[0m: Currently logged in as: [33mjpao[0m ([33mpri-data[0m). Use [1m`wandb login --relogin`[0m to force relogin
    [34m[1mwandb[0m: [33mWARNING[0m If you're specifying your api key in code, ensure this code is not shared publicly.
    [34m[1mwandb[0m: [33mWARNING[0m Consider setting the WANDB_API_KEY environment variable, or running `wandb login` from the command line.
    [34m[1mwandb[0m: Appending key for api.wandb.ai to your netrc file: C:\Users\jPao/.netrc
    

## Data

Data yang dipergunakan adalah sebanyak 101,077 kata. Terdapat 2 versi data, data versi 1 hanya memiliki 56,751 kata dan data versi 2 adalah data lengkap.

* Data 1: 56,751 kata, terdiri dari 34,174 kata dengan label `bukan_warna` dan 22,577 kata dengan label `warna` atau rasio 1.51 : 1 `bukan_warna` berbanding `warna`
* Data 2: 101,077 kata, dikarenakan kekurangan *man power* untuk proses *labeling* manual, maka kita tidak akan menggunakan data ini.

`brand`, `urut_kata` dan `total_kata` akan digunakan sebagai alternatif variabel independen tambahan dalam model tertentu.


```python
""" 
Membaca data ke dalam DataFrame pandas
Merubah kolom `urut_kata` dan 'total_kata' menjadi float32
"""
data = pd.read_csv('data/setengah_dataset_artikel.csv')
data = data.astype({'urut_kata': np.float32, 'total_kata': np.float32})
# Untuk dokumentasi, gunakan format markdown untuk rendering dataframe
# Menampilkan 100 data pertama
print(data[:100].to_markdown())
```

|    | brand   | nama_artikel                       | kata       | label       |   urut_kata |   total_kata |
|---:|:--------|:-----------------------------------|:-----------|:------------|------------:|-------------:|
|  0 | ADI     | ADISSAGE-BLACK/BLACK/RUNWHT        | ADISSAGE   | bukan_warna |           1 |            4 |
|  1 | ADI     | ADISSAGE-BLACK/BLACK/RUNWHT        | BLACK      | warna       |           2 |            4 |
|  2 | ADI     | ADISSAGE-BLACK/BLACK/RUNWHT        | BLACK      | warna       |           3 |            4 |
|  3 | ADI     | ADISSAGE-BLACK/BLACK/RUNWHT        | RUNWHT     | warna       |           4 |            4 |
|  4 | ADI     | ADISSAGE-N.NAVY/N.NAVY/RUNWHT      | ADISSAGE   | bukan_warna |           1 |            4 |
|  5 | ADI     | ADISSAGE-N.NAVY/N.NAVY/RUNWHT      | N.NAVY     | warna       |           2 |            4 |
|  6 | ADI     | ADISSAGE-N.NAVY/N.NAVY/RUNWHT      | N.NAVY     | warna       |           3 |            4 |
|  7 | ADI     | ADISSAGE-N.NAVY/N.NAVY/RUNWHT      | RUNWHT     | warna       |           4 |            4 |
|  8 | ADI     | 3 STRIPE D 29.5-BASKETBALL NATURAL | 3          | bukan_warna |           1 |            6 |
|  9 | ADI     | 3 STRIPE D 29.5-BASKETBALL NATURAL | STRIPE     | bukan_warna |           2 |            6 |
| 10 | ADI     | 3 STRIPE D 29.5-BASKETBALL NATURAL | D          | bukan_warna |           3 |            6 |
| 11 | ADI     | 3 STRIPE D 29.5-BASKETBALL NATURAL | 29.5       | bukan_warna |           4 |            6 |
| 12 | ADI     | 3 STRIPE D 29.5-BASKETBALL NATURAL | BASKETBALL | warna       |           5 |            6 |
| 13 | ADI     | 3 STRIPE D 29.5-BASKETBALL NATURAL | NATURAL    | warna       |           6 |            6 |
| 14 | ADI     | 3S RUBBER X-BLACK                  | 3S         | bukan_warna |           1 |            4 |
| 15 | ADI     | 3S RUBBER X-BLACK                  | RUBBER     | bukan_warna |           2 |            4 |
| 16 | ADI     | 3S RUBBER X-BLACK                  | X          | bukan_warna |           3 |            4 |
| 17 | ADI     | 3S RUBBER X-BLACK                  | BLACK      | warna       |           4 |            4 |
| 18 | ADI     | ADILETTE-BLACK1/WHT/BLACK1         | ADILETTE   | bukan_warna |           1 |            4 |
| 19 | ADI     | ADILETTE-BLACK1/WHT/BLACK1         | BLACK1     | warna       |           2 |            4 |
| 20 | ADI     | ADILETTE-BLACK1/WHT/BLACK1         | WHT        | warna       |           3 |            4 |
| 21 | ADI     | ADILETTE-BLACK1/WHT/BLACK1         | BLACK1     | warna       |           4 |            4 |
| 22 | ADI     | ADILETTE-WHITE                     | ADILETTE   | bukan_warna |           1 |            2 |
| 23 | ADI     | ADILETTE-WHITE                     | WHITE      | warna       |           2 |            2 |
| 24 | ADI     | TANGO ROSARIO-WHITE                | TANGO      | bukan_warna |           1 |            3 |
| 25 | ADI     | TANGO ROSARIO-WHITE                | ROSARIO    | bukan_warna |           2 |            3 |
| 26 | ADI     | TANGO ROSARIO-WHITE                | WHITE      | warna       |           3 |            3 |
| 27 | ADI     | ADISSAGE-N.NAVY/N.NAVY/RUNWHT      | ADISSAGE   | bukan_warna |           1 |            4 |
| 28 | ADI     | ADISSAGE-N.NAVY/N.NAVY/RUNWHT      | N.NAVY     | warna       |           2 |            4 |
| 29 | ADI     | ADISSAGE-N.NAVY/N.NAVY/RUNWHT      | N.NAVY     | warna       |           3 |            4 |
| 30 | ADI     | ADISSAGE-N.NAVY/N.NAVY/RUNWHT      | RUNWHT     | warna       |           4 |            4 |
| 31 | ADI     | CFC SCARF-CHEBLU/WHITE             | CFC        | bukan_warna |           1 |            4 |
| 32 | ADI     | CFC SCARF-CHEBLU/WHITE             | SCARF      | bukan_warna |           2 |            4 |
| 33 | ADI     | CFC SCARF-CHEBLU/WHITE             | CHEBLU     | warna       |           3 |            4 |
| 34 | ADI     | CFC SCARF-CHEBLU/WHITE             | WHITE      | warna       |           4 |            4 |
| 35 | ADI     | DFB H JSY Y-WHITE/BLACK            | DFB        | bukan_warna |           1 |            6 |
| 36 | ADI     | DFB H JSY Y-WHITE/BLACK            | H          | bukan_warna |           2 |            6 |
| 37 | ADI     | DFB H JSY Y-WHITE/BLACK            | JSY        | bukan_warna |           3 |            6 |
| 38 | ADI     | DFB H JSY Y-WHITE/BLACK            | Y          | bukan_warna |           4 |            6 |
| 39 | ADI     | DFB H JSY Y-WHITE/BLACK            | WHITE      | warna       |           5 |            6 |
| 40 | ADI     | DFB H JSY Y-WHITE/BLACK            | BLACK      | warna       |           6 |            6 |
| 41 | ADI     | 3S PER N-S HC3P-BLACK/BLACK/WHITE  | 3S         | bukan_warna |           1 |            8 |
| 42 | ADI     | 3S PER N-S HC3P-BLACK/BLACK/WHITE  | PER        | bukan_warna |           2 |            8 |
| 43 | ADI     | 3S PER N-S HC3P-BLACK/BLACK/WHITE  | N          | bukan_warna |           3 |            8 |
| 44 | ADI     | 3S PER N-S HC3P-BLACK/BLACK/WHITE  | S          | bukan_warna |           4 |            8 |
| 45 | ADI     | 3S PER N-S HC3P-BLACK/BLACK/WHITE  | HC3P       | bukan_warna |           5 |            8 |
| 46 | ADI     | 3S PER N-S HC3P-BLACK/BLACK/WHITE  | BLACK      | warna       |           6 |            8 |
| 47 | ADI     | 3S PER N-S HC3P-BLACK/BLACK/WHITE  | BLACK      | warna       |           7 |            8 |
| 48 | ADI     | 3S PER N-S HC3P-BLACK/BLACK/WHITE  | WHITE      | warna       |           8 |            8 |
| 49 | ADI     | 3S PER AN HC 3P-WHITE/WHITE/BLACK  | 3S         | bukan_warna |           1 |            8 |
| 50 | ADI     | 3S PER AN HC 3P-WHITE/WHITE/BLACK  | PER        | bukan_warna |           2 |            8 |
| 51 | ADI     | 3S PER AN HC 3P-WHITE/WHITE/BLACK  | AN         | bukan_warna |           3 |            8 |
| 52 | ADI     | 3S PER AN HC 3P-WHITE/WHITE/BLACK  | HC         | bukan_warna |           4 |            8 |
| 53 | ADI     | 3S PER AN HC 3P-WHITE/WHITE/BLACK  | 3P         | bukan_warna |           5 |            8 |
| 54 | ADI     | 3S PER AN HC 3P-WHITE/WHITE/BLACK  | WHITE      | warna       |           6 |            8 |
| 55 | ADI     | 3S PER AN HC 3P-WHITE/WHITE/BLACK  | WHITE      | warna       |           7 |            8 |
| 56 | ADI     | 3S PER AN HC 3P-WHITE/WHITE/BLACK  | BLACK      | warna       |           8 |            8 |
| 57 | ADI     | 3S PER AN HC 3P-BLACK/BLACK/BLACK  | 3S         | bukan_warna |           1 |            8 |
| 58 | ADI     | 3S PER AN HC 3P-BLACK/BLACK/BLACK  | PER        | bukan_warna |           2 |            8 |
| 59 | ADI     | 3S PER AN HC 3P-BLACK/BLACK/BLACK  | AN         | bukan_warna |           3 |            8 |
| 60 | ADI     | 3S PER AN HC 3P-BLACK/BLACK/BLACK  | HC         | bukan_warna |           4 |            8 |
| 61 | ADI     | 3S PER AN HC 3P-BLACK/BLACK/BLACK  | 3P         | bukan_warna |           5 |            8 |
| 62 | ADI     | 3S PER AN HC 3P-BLACK/BLACK/BLACK  | BLACK      | warna       |           6 |            8 |
| 63 | ADI     | 3S PER AN HC 3P-BLACK/BLACK/BLACK  | BLACK      | warna       |           7 |            8 |
| 64 | ADI     | 3S PER AN HC 3P-BLACK/BLACK/BLACK  | BLACK      | warna       |           8 |            8 |
| 65 | ADI     | 3S PER AN HC 1P-WHITE/WHITE/BLACK  | 3S         | bukan_warna |           1 |            8 |
| 66 | ADI     | 3S PER AN HC 1P-WHITE/WHITE/BLACK  | PER        | bukan_warna |           2 |            8 |
| 67 | ADI     | 3S PER AN HC 1P-WHITE/WHITE/BLACK  | AN         | bukan_warna |           3 |            8 |
| 68 | ADI     | 3S PER AN HC 1P-WHITE/WHITE/BLACK  | HC         | bukan_warna |           4 |            8 |
| 69 | ADI     | 3S PER AN HC 1P-WHITE/WHITE/BLACK  | 1P         | bukan_warna |           5 |            8 |
| 70 | ADI     | 3S PER AN HC 1P-WHITE/WHITE/BLACK  | WHITE      | warna       |           6 |            8 |
| 71 | ADI     | 3S PER AN HC 1P-WHITE/WHITE/BLACK  | WHITE      | warna       |           7 |            8 |
| 72 | ADI     | 3S PER AN HC 1P-WHITE/WHITE/BLACK  | BLACK      | warna       |           8 |            8 |
| 73 | ADI     | 3S PER AN HC 1P-BLACK/BLACK/WHITE  | 3S         | bukan_warna |           1 |            8 |
| 74 | ADI     | 3S PER AN HC 1P-BLACK/BLACK/WHITE  | PER        | bukan_warna |           2 |            8 |
| 75 | ADI     | 3S PER AN HC 1P-BLACK/BLACK/WHITE  | AN         | bukan_warna |           3 |            8 |
| 76 | ADI     | 3S PER AN HC 1P-BLACK/BLACK/WHITE  | HC         | bukan_warna |           4 |            8 |
| 77 | ADI     | 3S PER AN HC 1P-BLACK/BLACK/WHITE  | 1P         | bukan_warna |           5 |            8 |
| 78 | ADI     | 3S PER AN HC 1P-BLACK/BLACK/WHITE  | BLACK      | warna       |           6 |            8 |
| 79 | ADI     | 3S PER AN HC 1P-BLACK/BLACK/WHITE  | BLACK      | warna       |           7 |            8 |
| 80 | ADI     | 3S PER AN HC 1P-BLACK/BLACK/WHITE  | WHITE      | warna       |           8 |            8 |
| 81 | ADI     | 3S PER CR HC 3P-WHITE/WHITE/WHITE  | 3S         | bukan_warna |           1 |            8 |
| 82 | ADI     | 3S PER CR HC 3P-WHITE/WHITE/WHITE  | PER        | bukan_warna |           2 |            8 |
| 83 | ADI     | 3S PER CR HC 3P-WHITE/WHITE/WHITE  | CR         | bukan_warna |           3 |            8 |
| 84 | ADI     | 3S PER CR HC 3P-WHITE/WHITE/WHITE  | HC         | bukan_warna |           4 |            8 |
| 85 | ADI     | 3S PER CR HC 3P-WHITE/WHITE/WHITE  | 3P         | bukan_warna |           5 |            8 |
| 86 | ADI     | 3S PER CR HC 3P-WHITE/WHITE/WHITE  | WHITE      | warna       |           6 |            8 |
| 87 | ADI     | 3S PER CR HC 3P-WHITE/WHITE/WHITE  | WHITE      | warna       |           7 |            8 |
| 88 | ADI     | 3S PER CR HC 3P-WHITE/WHITE/WHITE  | WHITE      | warna       |           8 |            8 |
| 89 | ADI     | 3S PER CR HC 3P-BLACK/BLACK/WHITE  | 3S         | bukan_warna |           1 |            8 |
| 90 | ADI     | 3S PER CR HC 3P-BLACK/BLACK/WHITE  | PER        | bukan_warna |           2 |            8 |
| 91 | ADI     | 3S PER CR HC 3P-BLACK/BLACK/WHITE  | CR         | bukan_warna |           3 |            8 |
| 92 | ADI     | 3S PER CR HC 3P-BLACK/BLACK/WHITE  | HC         | bukan_warna |           4 |            8 |
| 93 | ADI     | 3S PER CR HC 3P-BLACK/BLACK/WHITE  | 3P         | bukan_warna |           5 |            8 |
| 94 | ADI     | 3S PER CR HC 3P-BLACK/BLACK/WHITE  | BLACK      | warna       |           6 |            8 |
| 95 | ADI     | 3S PER CR HC 3P-BLACK/BLACK/WHITE  | BLACK      | warna       |           7 |            8 |
| 96 | ADI     | 3S PER CR HC 3P-BLACK/BLACK/WHITE  | WHITE      | warna       |           8 |            8 |
| 97 | ADI     | 3S PER CR HC 1P-WHITE/WHITE/BLACK  | 3S         | bukan_warna |           1 |            8 |
| 98 | ADI     | 3S PER CR HC 1P-WHITE/WHITE/BLACK  | PER        | bukan_warna |           2 |            8 |
| 99 | ADI     | 3S PER CR HC 1P-WHITE/WHITE/BLACK  | CR         | bukan_warna |           3 |            8 |
    

### Eksplorasi Data


```python
# distribusi label dalam data
print(data['label'].value_counts())
data['label'].value_counts().plot(kind='bar')
plt.gca().set_title('Distribusi Label', fontsize=18)
plt.figure(facecolor='w')
```

    bukan_warna    34174
    warna          22577
    Name: label, dtype: int64




    
![png](ColorSkim_AI_files/ColorSkim_AI_10_2.png)



```python
# distribusi label dalam brand (data hanya menunjukkan 10 teratas)
print(data[['brand', 'label']].value_counts().unstack().sort_values(by='bukan_warna', ascending=False)[:10])
data[['brand', 'label']].value_counts().unstack().sort_values(by='bukan_warna', ascending=False)[:10].plot(kind='bar')
plt.gca().set_title('Distribusi Label Berdasarkan Brand', fontsize=18)
plt.figure(facecolor='w')
```

    label  bukan_warna    warna
    brand                      
    NIK        13396.0  10807.0
    ADI        10028.0   7073.0
    PUM         4279.0   2062.0
    BBC         1174.0    367.0
    CAO          887.0     61.0
    HER          868.0    287.0
    AGL          611.0    212.0
    KIP          554.0    321.0
    STN          494.0    255.0
    WAR          404.0    298.0




    
![png](ColorSkim_AI_files/ColorSkim_AI_11_2.png)
    


### Konversi Fitur dan Label ke dalam numerik

Kita akan melakukan pengkonversian fitur dan label ke dalam bentuk numerik, dikarenakan jaringan saraf tiruan hanya dapat bekerja dalam data numerik. 

Terdapat dua jenis *encoding* untuk data yang bersifat kategorikal:

* `OneHotEncoder`
* `LabelEncoder`

**OneHotEncoder**
*Encoding* ini akan merubah data satu kolom menjadi multi-kolom dengan nilai 1 dan 0 dimana jumlah kolom sama dengan jumlah kategori, seperti berikut:

| brand | brand_NIK | brand_ADI | brand_SPE | brand_PIE | brand_... |
| --- | --- | --- | --- | --- | --- |
| NIK | 1 | 0 | 0 | 0 | ... |
| SPE | 0 | 0 | 1 | 0 | ... |
| PIE | 0 | 0 | 0 | 1 | ... |
| ADI | 0 | 1 | 0 | 0 | ... |
| SPE | 0 | 0 | 1 | 0 | ... |
| ... | ... | ... | ... | ... | ... |

**LabelEncoder**
*Encoding* ini akan merubah data pada satu kolom menjadi 0, 1, 2, 3.. dstnya sesuai dengan jumlah kategorinya, seperti berikut:

| brand | brand_label_encoded |
| --- | --- |
| NIK | 0 |
| SPE | 1 |
| PIE | 2 |
| ADI | 3 |
| SPE | 1 |
| ... | ... |

**Kapan menggunakan `OneHotEncoder` atau `LabelEncoder` dalam sebuah proses encoding?** Kita dapat menggunakan `OneHotEncoder` ketika kita tidak menginginkan suatu bentuk hubungan hirarki di dalam data kategorikal yang kita miliki. Dalam hal ini ketika kita tidak ingin jaringan saraf tiruan untuk memandang ADI (3) lebih signifikan dari NIK (0) dalam hal nilainya jika dilakukan label *encoding*, maka kita dapat menggunakan `OneHotEncoder`.
Jika kategori bersifat biner seperti 'Pria' atau 'Wanita', 'Ya' atau 'Tidak' dsbnya, penggunaan `LabelEncoder` dinilai lebih efektif.

> Dengan pertimbangan di atas dan melihat struktur data kita, maka kita akan menggunakan `OneHotEncoder` untuk kolom *brand* (fitur) dan menggunakan `LabelEncoder` untuk kolom *label* (target), kecuali untuk **Model 0** yang akan menggunakan fungsi ekstraksi fitur dengan `TfIdfVectorizer` kita hanya akan menggunakan kolom 'label' yang belum di-*encode*.


```python
# OneHotEncoding pada fitur brand
fitur_encoder = OneHotEncoder(sparse=False)
brand_encoded = fitur_encoder.fit_transform(data['brand'].to_numpy().reshape(-1, 1))
df_fitur_encoded = pd.DataFrame(brand_encoded, columns=fitur_encoder.get_feature_names_out(['brand']))

# LabelEncoding pada target label
label_encoder = LabelEncoder()
label_encoded = label_encoder.fit_transform(data['label'])
df_label_encoded = pd.DataFrame(label_encoded, columns=['label_encoded'])

# gabungkan dengan dataframe awal
data_encoded = data.copy()
data_encoded = pd.concat([data_encoded,df_fitur_encoded, df_label_encoded], axis=1)
# Menampilkan 100 data pertama
print(data_encoded[:100].to_markdown())
```

|    | brand   | nama_artikel                       | kata       | label       |   urut_kata |   total_kata |   brand_ADI |   brand_ADS |   brand_AGL |   brand_AND |   brand_ASC |   brand_BAL |   brand_BBC |   brand_BEA |   brand_CAO |   brand_CIT |   brand_CRP |   brand_DOM |   brand_FIS |   brand_GUE |   brand_HER |   brand_JAS |   brand_KIP |   brand_NEW |   brand_NFA |   brand_NFC |   brand_NFL |   brand_NIB |   brand_NIC |   brand_NIK |   brand_NPS |   brand_ODD |   brand_PBY |   brand_PSB |   brand_PTG |   brand_PUM |   brand_REL |   brand_SAU |   brand_SOC |   brand_STN |   brand_UME |   brand_VAP |   brand_WAR |   label_encoded |
|---:|:--------|:-----------------------------------|:-----------|:------------|------------:|-------------:|------------:|------------:|------------:|------------:|------------:|------------:|------------:|------------:|------------:|------------:|------------:|------------:|------------:|------------:|------------:|------------:|------------:|------------:|------------:|------------:|------------:|------------:|------------:|------------:|------------:|------------:|------------:|------------:|------------:|------------:|------------:|------------:|------------:|------------:|------------:|------------:|------------:|----------------:|
|  0 | ADI     | ADISSAGE-BLACK/BLACK/RUNWHT        | ADISSAGE   | bukan_warna |           1 |            4 |           1 |           0 |           0 |           0 |           0 |           0 |           0 |           0 |           0 |           0 |           0 |           0 |           0 |           0 |           0 |           0 |           0 |           0 |           0 |           0 |           0 |           0 |           0 |           0 |           0 |           0 |           0 |           0 |           0 |           0 |           0 |           0 |           0 |           0 |           0 |           0 |           0 |               0 |
|  1 | ADI     | ADISSAGE-BLACK/BLACK/RUNWHT        | BLACK      | warna       |           2 |            4 |           1 |           0 |           0 |           0 |           0 |           0 |           0 |           0 |           0 |           0 |           0 |           0 |           0 |           0 |           0 |           0 |           0 |           0 |           0 |           0 |           0 |           0 |           0 |           0 |           0 |           0 |           0 |           0 |           0 |           0 |           0 |           0 |           0 |           0 |           0 |           0 |           0 |               1 |
|  2 | ADI     | ADISSAGE-BLACK/BLACK/RUNWHT        | BLACK      | warna       |           3 |            4 |           1 |           0 |           0 |           0 |           0 |           0 |           0 |           0 |           0 |           0 |           0 |           0 |           0 |           0 |           0 |           0 |           0 |           0 |           0 |           0 |           0 |           0 |           0 |           0 |           0 |           0 |           0 |           0 |           0 |           0 |           0 |           0 |           0 |           0 |           0 |           0 |           0 |               1 |
|  3 | ADI     | ADISSAGE-BLACK/BLACK/RUNWHT        | RUNWHT     | warna       |           4 |            4 |           1 |           0 |           0 |           0 |           0 |           0 |           0 |           0 |           0 |           0 |           0 |           0 |           0 |           0 |           0 |           0 |           0 |           0 |           0 |           0 |           0 |           0 |           0 |           0 |           0 |           0 |           0 |           0 |           0 |           0 |           0 |           0 |           0 |           0 |           0 |           0 |           0 |               1 |
|  4 | ADI     | ADISSAGE-N.NAVY/N.NAVY/RUNWHT      | ADISSAGE   | bukan_warna |           1 |            4 |           1 |           0 |           0 |           0 |           0 |           0 |           0 |           0 |           0 |           0 |           0 |           0 |           0 |           0 |           0 |           0 |           0 |           0 |           0 |           0 |           0 |           0 |           0 |           0 |           0 |           0 |           0 |           0 |           0 |           0 |           0 |           0 |           0 |           0 |           0 |           0 |           0 |               0 |
|  5 | ADI     | ADISSAGE-N.NAVY/N.NAVY/RUNWHT      | N.NAVY     | warna       |           2 |            4 |           1 |           0 |           0 |           0 |           0 |           0 |           0 |           0 |           0 |           0 |           0 |           0 |           0 |           0 |           0 |           0 |           0 |           0 |           0 |           0 |           0 |           0 |           0 |           0 |           0 |           0 |           0 |           0 |           0 |           0 |           0 |           0 |           0 |           0 |           0 |           0 |           0 |               1 |
|  6 | ADI     | ADISSAGE-N.NAVY/N.NAVY/RUNWHT      | N.NAVY     | warna       |           3 |            4 |           1 |           0 |           0 |           0 |           0 |           0 |           0 |           0 |           0 |           0 |           0 |           0 |           0 |           0 |           0 |           0 |           0 |           0 |           0 |           0 |           0 |           0 |           0 |           0 |           0 |           0 |           0 |           0 |           0 |           0 |           0 |           0 |           0 |           0 |           0 |           0 |           0 |               1 |
|  7 | ADI     | ADISSAGE-N.NAVY/N.NAVY/RUNWHT      | RUNWHT     | warna       |           4 |            4 |           1 |           0 |           0 |           0 |           0 |           0 |           0 |           0 |           0 |           0 |           0 |           0 |           0 |           0 |           0 |           0 |           0 |           0 |           0 |           0 |           0 |           0 |           0 |           0 |           0 |           0 |           0 |           0 |           0 |           0 |           0 |           0 |           0 |           0 |           0 |           0 |           0 |               1 |
|  8 | ADI     | 3 STRIPE D 29.5-BASKETBALL NATURAL | 3          | bukan_warna |           1 |            6 |           1 |           0 |           0 |           0 |           0 |           0 |           0 |           0 |           0 |           0 |           0 |           0 |           0 |           0 |           0 |           0 |           0 |           0 |           0 |           0 |           0 |           0 |           0 |           0 |           0 |           0 |           0 |           0 |           0 |           0 |           0 |           0 |           0 |           0 |           0 |           0 |           0 |               0 |
|  9 | ADI     | 3 STRIPE D 29.5-BASKETBALL NATURAL | STRIPE     | bukan_warna |           2 |            6 |           1 |           0 |           0 |           0 |           0 |           0 |           0 |           0 |           0 |           0 |           0 |           0 |           0 |           0 |           0 |           0 |           0 |           0 |           0 |           0 |           0 |           0 |           0 |           0 |           0 |           0 |           0 |           0 |           0 |           0 |           0 |           0 |           0 |           0 |           0 |           0 |           0 |               0 |
| 10 | ADI     | 3 STRIPE D 29.5-BASKETBALL NATURAL | D          | bukan_warna |           3 |            6 |           1 |           0 |           0 |           0 |           0 |           0 |           0 |           0 |           0 |           0 |           0 |           0 |           0 |           0 |           0 |           0 |           0 |           0 |           0 |           0 |           0 |           0 |           0 |           0 |           0 |           0 |           0 |           0 |           0 |           0 |           0 |           0 |           0 |           0 |           0 |           0 |           0 |               0 |
| 11 | ADI     | 3 STRIPE D 29.5-BASKETBALL NATURAL | 29.5       | bukan_warna |           4 |            6 |           1 |           0 |           0 |           0 |           0 |           0 |           0 |           0 |           0 |           0 |           0 |           0 |           0 |           0 |           0 |           0 |           0 |           0 |           0 |           0 |           0 |           0 |           0 |           0 |           0 |           0 |           0 |           0 |           0 |           0 |           0 |           0 |           0 |           0 |           0 |           0 |           0 |               0 |
| 12 | ADI     | 3 STRIPE D 29.5-BASKETBALL NATURAL | BASKETBALL | warna       |           5 |            6 |           1 |           0 |           0 |           0 |           0 |           0 |           0 |           0 |           0 |           0 |           0 |           0 |           0 |           0 |           0 |           0 |           0 |           0 |           0 |           0 |           0 |           0 |           0 |           0 |           0 |           0 |           0 |           0 |           0 |           0 |           0 |           0 |           0 |           0 |           0 |           0 |           0 |               1 |
| 13 | ADI     | 3 STRIPE D 29.5-BASKETBALL NATURAL | NATURAL    | warna       |           6 |            6 |           1 |           0 |           0 |           0 |           0 |           0 |           0 |           0 |           0 |           0 |           0 |           0 |           0 |           0 |           0 |           0 |           0 |           0 |           0 |           0 |           0 |           0 |           0 |           0 |           0 |           0 |           0 |           0 |           0 |           0 |           0 |           0 |           0 |           0 |           0 |           0 |           0 |               1 |
| 14 | ADI     | 3S RUBBER X-BLACK                  | 3S         | bukan_warna |           1 |            4 |           1 |           0 |           0 |           0 |           0 |           0 |           0 |           0 |           0 |           0 |           0 |           0 |           0 |           0 |           0 |           0 |           0 |           0 |           0 |           0 |           0 |           0 |           0 |           0 |           0 |           0 |           0 |           0 |           0 |           0 |           0 |           0 |           0 |           0 |           0 |           0 |           0 |               0 |
| 15 | ADI     | 3S RUBBER X-BLACK                  | RUBBER     | bukan_warna |           2 |            4 |           1 |           0 |           0 |           0 |           0 |           0 |           0 |           0 |           0 |           0 |           0 |           0 |           0 |           0 |           0 |           0 |           0 |           0 |           0 |           0 |           0 |           0 |           0 |           0 |           0 |           0 |           0 |           0 |           0 |           0 |           0 |           0 |           0 |           0 |           0 |           0 |           0 |               0 |
| 16 | ADI     | 3S RUBBER X-BLACK                  | X          | bukan_warna |           3 |            4 |           1 |           0 |           0 |           0 |           0 |           0 |           0 |           0 |           0 |           0 |           0 |           0 |           0 |           0 |           0 |           0 |           0 |           0 |           0 |           0 |           0 |           0 |           0 |           0 |           0 |           0 |           0 |           0 |           0 |           0 |           0 |           0 |           0 |           0 |           0 |           0 |           0 |               0 |
| 17 | ADI     | 3S RUBBER X-BLACK                  | BLACK      | warna       |           4 |            4 |           1 |           0 |           0 |           0 |           0 |           0 |           0 |           0 |           0 |           0 |           0 |           0 |           0 |           0 |           0 |           0 |           0 |           0 |           0 |           0 |           0 |           0 |           0 |           0 |           0 |           0 |           0 |           0 |           0 |           0 |           0 |           0 |           0 |           0 |           0 |           0 |           0 |               1 |
| 18 | ADI     | ADILETTE-BLACK1/WHT/BLACK1         | ADILETTE   | bukan_warna |           1 |            4 |           1 |           0 |           0 |           0 |           0 |           0 |           0 |           0 |           0 |           0 |           0 |           0 |           0 |           0 |           0 |           0 |           0 |           0 |           0 |           0 |           0 |           0 |           0 |           0 |           0 |           0 |           0 |           0 |           0 |           0 |           0 |           0 |           0 |           0 |           0 |           0 |           0 |               0 |
| 19 | ADI     | ADILETTE-BLACK1/WHT/BLACK1         | BLACK1     | warna       |           2 |            4 |           1 |           0 |           0 |           0 |           0 |           0 |           0 |           0 |           0 |           0 |           0 |           0 |           0 |           0 |           0 |           0 |           0 |           0 |           0 |           0 |           0 |           0 |           0 |           0 |           0 |           0 |           0 |           0 |           0 |           0 |           0 |           0 |           0 |           0 |           0 |           0 |           0 |               1 |
| 20 | ADI     | ADILETTE-BLACK1/WHT/BLACK1         | WHT        | warna       |           3 |            4 |           1 |           0 |           0 |           0 |           0 |           0 |           0 |           0 |           0 |           0 |           0 |           0 |           0 |           0 |           0 |           0 |           0 |           0 |           0 |           0 |           0 |           0 |           0 |           0 |           0 |           0 |           0 |           0 |           0 |           0 |           0 |           0 |           0 |           0 |           0 |           0 |           0 |               1 |
| 21 | ADI     | ADILETTE-BLACK1/WHT/BLACK1         | BLACK1     | warna       |           4 |            4 |           1 |           0 |           0 |           0 |           0 |           0 |           0 |           0 |           0 |           0 |           0 |           0 |           0 |           0 |           0 |           0 |           0 |           0 |           0 |           0 |           0 |           0 |           0 |           0 |           0 |           0 |           0 |           0 |           0 |           0 |           0 |           0 |           0 |           0 |           0 |           0 |           0 |               1 |
| 22 | ADI     | ADILETTE-WHITE                     | ADILETTE   | bukan_warna |           1 |            2 |           1 |           0 |           0 |           0 |           0 |           0 |           0 |           0 |           0 |           0 |           0 |           0 |           0 |           0 |           0 |           0 |           0 |           0 |           0 |           0 |           0 |           0 |           0 |           0 |           0 |           0 |           0 |           0 |           0 |           0 |           0 |           0 |           0 |           0 |           0 |           0 |           0 |               0 |
| 23 | ADI     | ADILETTE-WHITE                     | WHITE      | warna       |           2 |            2 |           1 |           0 |           0 |           0 |           0 |           0 |           0 |           0 |           0 |           0 |           0 |           0 |           0 |           0 |           0 |           0 |           0 |           0 |           0 |           0 |           0 |           0 |           0 |           0 |           0 |           0 |           0 |           0 |           0 |           0 |           0 |           0 |           0 |           0 |           0 |           0 |           0 |               1 |
| 24 | ADI     | TANGO ROSARIO-WHITE                | TANGO      | bukan_warna |           1 |            3 |           1 |           0 |           0 |           0 |           0 |           0 |           0 |           0 |           0 |           0 |           0 |           0 |           0 |           0 |           0 |           0 |           0 |           0 |           0 |           0 |           0 |           0 |           0 |           0 |           0 |           0 |           0 |           0 |           0 |           0 |           0 |           0 |           0 |           0 |           0 |           0 |           0 |               0 |
| 25 | ADI     | TANGO ROSARIO-WHITE                | ROSARIO    | bukan_warna |           2 |            3 |           1 |           0 |           0 |           0 |           0 |           0 |           0 |           0 |           0 |           0 |           0 |           0 |           0 |           0 |           0 |           0 |           0 |           0 |           0 |           0 |           0 |           0 |           0 |           0 |           0 |           0 |           0 |           0 |           0 |           0 |           0 |           0 |           0 |           0 |           0 |           0 |           0 |               0 |
| 26 | ADI     | TANGO ROSARIO-WHITE                | WHITE      | warna       |           3 |            3 |           1 |           0 |           0 |           0 |           0 |           0 |           0 |           0 |           0 |           0 |           0 |           0 |           0 |           0 |           0 |           0 |           0 |           0 |           0 |           0 |           0 |           0 |           0 |           0 |           0 |           0 |           0 |           0 |           0 |           0 |           0 |           0 |           0 |           0 |           0 |           0 |           0 |               1 |
| 27 | ADI     | ADISSAGE-N.NAVY/N.NAVY/RUNWHT      | ADISSAGE   | bukan_warna |           1 |            4 |           1 |           0 |           0 |           0 |           0 |           0 |           0 |           0 |           0 |           0 |           0 |           0 |           0 |           0 |           0 |           0 |           0 |           0 |           0 |           0 |           0 |           0 |           0 |           0 |           0 |           0 |           0 |           0 |           0 |           0 |           0 |           0 |           0 |           0 |           0 |           0 |           0 |               0 |
| 28 | ADI     | ADISSAGE-N.NAVY/N.NAVY/RUNWHT      | N.NAVY     | warna       |           2 |            4 |           1 |           0 |           0 |           0 |           0 |           0 |           0 |           0 |           0 |           0 |           0 |           0 |           0 |           0 |           0 |           0 |           0 |           0 |           0 |           0 |           0 |           0 |           0 |           0 |           0 |           0 |           0 |           0 |           0 |           0 |           0 |           0 |           0 |           0 |           0 |           0 |           0 |               1 |
| 29 | ADI     | ADISSAGE-N.NAVY/N.NAVY/RUNWHT      | N.NAVY     | warna       |           3 |            4 |           1 |           0 |           0 |           0 |           0 |           0 |           0 |           0 |           0 |           0 |           0 |           0 |           0 |           0 |           0 |           0 |           0 |           0 |           0 |           0 |           0 |           0 |           0 |           0 |           0 |           0 |           0 |           0 |           0 |           0 |           0 |           0 |           0 |           0 |           0 |           0 |           0 |               1 |
| 30 | ADI     | ADISSAGE-N.NAVY/N.NAVY/RUNWHT      | RUNWHT     | warna       |           4 |            4 |           1 |           0 |           0 |           0 |           0 |           0 |           0 |           0 |           0 |           0 |           0 |           0 |           0 |           0 |           0 |           0 |           0 |           0 |           0 |           0 |           0 |           0 |           0 |           0 |           0 |           0 |           0 |           0 |           0 |           0 |           0 |           0 |           0 |           0 |           0 |           0 |           0 |               1 |
| 31 | ADI     | CFC SCARF-CHEBLU/WHITE             | CFC        | bukan_warna |           1 |            4 |           1 |           0 |           0 |           0 |           0 |           0 |           0 |           0 |           0 |           0 |           0 |           0 |           0 |           0 |           0 |           0 |           0 |           0 |           0 |           0 |           0 |           0 |           0 |           0 |           0 |           0 |           0 |           0 |           0 |           0 |           0 |           0 |           0 |           0 |           0 |           0 |           0 |               0 |
| 32 | ADI     | CFC SCARF-CHEBLU/WHITE             | SCARF      | bukan_warna |           2 |            4 |           1 |           0 |           0 |           0 |           0 |           0 |           0 |           0 |           0 |           0 |           0 |           0 |           0 |           0 |           0 |           0 |           0 |           0 |           0 |           0 |           0 |           0 |           0 |           0 |           0 |           0 |           0 |           0 |           0 |           0 |           0 |           0 |           0 |           0 |           0 |           0 |           0 |               0 |
| 33 | ADI     | CFC SCARF-CHEBLU/WHITE             | CHEBLU     | warna       |           3 |            4 |           1 |           0 |           0 |           0 |           0 |           0 |           0 |           0 |           0 |           0 |           0 |           0 |           0 |           0 |           0 |           0 |           0 |           0 |           0 |           0 |           0 |           0 |           0 |           0 |           0 |           0 |           0 |           0 |           0 |           0 |           0 |           0 |           0 |           0 |           0 |           0 |           0 |               1 |
| 34 | ADI     | CFC SCARF-CHEBLU/WHITE             | WHITE      | warna       |           4 |            4 |           1 |           0 |           0 |           0 |           0 |           0 |           0 |           0 |           0 |           0 |           0 |           0 |           0 |           0 |           0 |           0 |           0 |           0 |           0 |           0 |           0 |           0 |           0 |           0 |           0 |           0 |           0 |           0 |           0 |           0 |           0 |           0 |           0 |           0 |           0 |           0 |           0 |               1 |
| 35 | ADI     | DFB H JSY Y-WHITE/BLACK            | DFB        | bukan_warna |           1 |            6 |           1 |           0 |           0 |           0 |           0 |           0 |           0 |           0 |           0 |           0 |           0 |           0 |           0 |           0 |           0 |           0 |           0 |           0 |           0 |           0 |           0 |           0 |           0 |           0 |           0 |           0 |           0 |           0 |           0 |           0 |           0 |           0 |           0 |           0 |           0 |           0 |           0 |               0 |
| 36 | ADI     | DFB H JSY Y-WHITE/BLACK            | H          | bukan_warna |           2 |            6 |           1 |           0 |           0 |           0 |           0 |           0 |           0 |           0 |           0 |           0 |           0 |           0 |           0 |           0 |           0 |           0 |           0 |           0 |           0 |           0 |           0 |           0 |           0 |           0 |           0 |           0 |           0 |           0 |           0 |           0 |           0 |           0 |           0 |           0 |           0 |           0 |           0 |               0 |
| 37 | ADI     | DFB H JSY Y-WHITE/BLACK            | JSY        | bukan_warna |           3 |            6 |           1 |           0 |           0 |           0 |           0 |           0 |           0 |           0 |           0 |           0 |           0 |           0 |           0 |           0 |           0 |           0 |           0 |           0 |           0 |           0 |           0 |           0 |           0 |           0 |           0 |           0 |           0 |           0 |           0 |           0 |           0 |           0 |           0 |           0 |           0 |           0 |           0 |               0 |
| 38 | ADI     | DFB H JSY Y-WHITE/BLACK            | Y          | bukan_warna |           4 |            6 |           1 |           0 |           0 |           0 |           0 |           0 |           0 |           0 |           0 |           0 |           0 |           0 |           0 |           0 |           0 |           0 |           0 |           0 |           0 |           0 |           0 |           0 |           0 |           0 |           0 |           0 |           0 |           0 |           0 |           0 |           0 |           0 |           0 |           0 |           0 |           0 |           0 |               0 |
| 39 | ADI     | DFB H JSY Y-WHITE/BLACK            | WHITE      | warna       |           5 |            6 |           1 |           0 |           0 |           0 |           0 |           0 |           0 |           0 |           0 |           0 |           0 |           0 |           0 |           0 |           0 |           0 |           0 |           0 |           0 |           0 |           0 |           0 |           0 |           0 |           0 |           0 |           0 |           0 |           0 |           0 |           0 |           0 |           0 |           0 |           0 |           0 |           0 |               1 |
| 40 | ADI     | DFB H JSY Y-WHITE/BLACK            | BLACK      | warna       |           6 |            6 |           1 |           0 |           0 |           0 |           0 |           0 |           0 |           0 |           0 |           0 |           0 |           0 |           0 |           0 |           0 |           0 |           0 |           0 |           0 |           0 |           0 |           0 |           0 |           0 |           0 |           0 |           0 |           0 |           0 |           0 |           0 |           0 |           0 |           0 |           0 |           0 |           0 |               1 |
| 41 | ADI     | 3S PER N-S HC3P-BLACK/BLACK/WHITE  | 3S         | bukan_warna |           1 |            8 |           1 |           0 |           0 |           0 |           0 |           0 |           0 |           0 |           0 |           0 |           0 |           0 |           0 |           0 |           0 |           0 |           0 |           0 |           0 |           0 |           0 |           0 |           0 |           0 |           0 |           0 |           0 |           0 |           0 |           0 |           0 |           0 |           0 |           0 |           0 |           0 |           0 |               0 |
| 42 | ADI     | 3S PER N-S HC3P-BLACK/BLACK/WHITE  | PER        | bukan_warna |           2 |            8 |           1 |           0 |           0 |           0 |           0 |           0 |           0 |           0 |           0 |           0 |           0 |           0 |           0 |           0 |           0 |           0 |           0 |           0 |           0 |           0 |           0 |           0 |           0 |           0 |           0 |           0 |           0 |           0 |           0 |           0 |           0 |           0 |           0 |           0 |           0 |           0 |           0 |               0 |
| 43 | ADI     | 3S PER N-S HC3P-BLACK/BLACK/WHITE  | N          | bukan_warna |           3 |            8 |           1 |           0 |           0 |           0 |           0 |           0 |           0 |           0 |           0 |           0 |           0 |           0 |           0 |           0 |           0 |           0 |           0 |           0 |           0 |           0 |           0 |           0 |           0 |           0 |           0 |           0 |           0 |           0 |           0 |           0 |           0 |           0 |           0 |           0 |           0 |           0 |           0 |               0 |
| 44 | ADI     | 3S PER N-S HC3P-BLACK/BLACK/WHITE  | S          | bukan_warna |           4 |            8 |           1 |           0 |           0 |           0 |           0 |           0 |           0 |           0 |           0 |           0 |           0 |           0 |           0 |           0 |           0 |           0 |           0 |           0 |           0 |           0 |           0 |           0 |           0 |           0 |           0 |           0 |           0 |           0 |           0 |           0 |           0 |           0 |           0 |           0 |           0 |           0 |           0 |               0 |
| 45 | ADI     | 3S PER N-S HC3P-BLACK/BLACK/WHITE  | HC3P       | bukan_warna |           5 |            8 |           1 |           0 |           0 |           0 |           0 |           0 |           0 |           0 |           0 |           0 |           0 |           0 |           0 |           0 |           0 |           0 |           0 |           0 |           0 |           0 |           0 |           0 |           0 |           0 |           0 |           0 |           0 |           0 |           0 |           0 |           0 |           0 |           0 |           0 |           0 |           0 |           0 |               0 |
| 46 | ADI     | 3S PER N-S HC3P-BLACK/BLACK/WHITE  | BLACK      | warna       |           6 |            8 |           1 |           0 |           0 |           0 |           0 |           0 |           0 |           0 |           0 |           0 |           0 |           0 |           0 |           0 |           0 |           0 |           0 |           0 |           0 |           0 |           0 |           0 |           0 |           0 |           0 |           0 |           0 |           0 |           0 |           0 |           0 |           0 |           0 |           0 |           0 |           0 |           0 |               1 |
| 47 | ADI     | 3S PER N-S HC3P-BLACK/BLACK/WHITE  | BLACK      | warna       |           7 |            8 |           1 |           0 |           0 |           0 |           0 |           0 |           0 |           0 |           0 |           0 |           0 |           0 |           0 |           0 |           0 |           0 |           0 |           0 |           0 |           0 |           0 |           0 |           0 |           0 |           0 |           0 |           0 |           0 |           0 |           0 |           0 |           0 |           0 |           0 |           0 |           0 |           0 |               1 |
| 48 | ADI     | 3S PER N-S HC3P-BLACK/BLACK/WHITE  | WHITE      | warna       |           8 |            8 |           1 |           0 |           0 |           0 |           0 |           0 |           0 |           0 |           0 |           0 |           0 |           0 |           0 |           0 |           0 |           0 |           0 |           0 |           0 |           0 |           0 |           0 |           0 |           0 |           0 |           0 |           0 |           0 |           0 |           0 |           0 |           0 |           0 |           0 |           0 |           0 |           0 |               1 |
| 49 | ADI     | 3S PER AN HC 3P-WHITE/WHITE/BLACK  | 3S         | bukan_warna |           1 |            8 |           1 |           0 |           0 |           0 |           0 |           0 |           0 |           0 |           0 |           0 |           0 |           0 |           0 |           0 |           0 |           0 |           0 |           0 |           0 |           0 |           0 |           0 |           0 |           0 |           0 |           0 |           0 |           0 |           0 |           0 |           0 |           0 |           0 |           0 |           0 |           0 |           0 |               0 |
| 50 | ADI     | 3S PER AN HC 3P-WHITE/WHITE/BLACK  | PER        | bukan_warna |           2 |            8 |           1 |           0 |           0 |           0 |           0 |           0 |           0 |           0 |           0 |           0 |           0 |           0 |           0 |           0 |           0 |           0 |           0 |           0 |           0 |           0 |           0 |           0 |           0 |           0 |           0 |           0 |           0 |           0 |           0 |           0 |           0 |           0 |           0 |           0 |           0 |           0 |           0 |               0 |
| 51 | ADI     | 3S PER AN HC 3P-WHITE/WHITE/BLACK  | AN         | bukan_warna |           3 |            8 |           1 |           0 |           0 |           0 |           0 |           0 |           0 |           0 |           0 |           0 |           0 |           0 |           0 |           0 |           0 |           0 |           0 |           0 |           0 |           0 |           0 |           0 |           0 |           0 |           0 |           0 |           0 |           0 |           0 |           0 |           0 |           0 |           0 |           0 |           0 |           0 |           0 |               0 |
| 52 | ADI     | 3S PER AN HC 3P-WHITE/WHITE/BLACK  | HC         | bukan_warna |           4 |            8 |           1 |           0 |           0 |           0 |           0 |           0 |           0 |           0 |           0 |           0 |           0 |           0 |           0 |           0 |           0 |           0 |           0 |           0 |           0 |           0 |           0 |           0 |           0 |           0 |           0 |           0 |           0 |           0 |           0 |           0 |           0 |           0 |           0 |           0 |           0 |           0 |           0 |               0 |
| 53 | ADI     | 3S PER AN HC 3P-WHITE/WHITE/BLACK  | 3P         | bukan_warna |           5 |            8 |           1 |           0 |           0 |           0 |           0 |           0 |           0 |           0 |           0 |           0 |           0 |           0 |           0 |           0 |           0 |           0 |           0 |           0 |           0 |           0 |           0 |           0 |           0 |           0 |           0 |           0 |           0 |           0 |           0 |           0 |           0 |           0 |           0 |           0 |           0 |           0 |           0 |               0 |
| 54 | ADI     | 3S PER AN HC 3P-WHITE/WHITE/BLACK  | WHITE      | warna       |           6 |            8 |           1 |           0 |           0 |           0 |           0 |           0 |           0 |           0 |           0 |           0 |           0 |           0 |           0 |           0 |           0 |           0 |           0 |           0 |           0 |           0 |           0 |           0 |           0 |           0 |           0 |           0 |           0 |           0 |           0 |           0 |           0 |           0 |           0 |           0 |           0 |           0 |           0 |               1 |
| 55 | ADI     | 3S PER AN HC 3P-WHITE/WHITE/BLACK  | WHITE      | warna       |           7 |            8 |           1 |           0 |           0 |           0 |           0 |           0 |           0 |           0 |           0 |           0 |           0 |           0 |           0 |           0 |           0 |           0 |           0 |           0 |           0 |           0 |           0 |           0 |           0 |           0 |           0 |           0 |           0 |           0 |           0 |           0 |           0 |           0 |           0 |           0 |           0 |           0 |           0 |               1 |
| 56 | ADI     | 3S PER AN HC 3P-WHITE/WHITE/BLACK  | BLACK      | warna       |           8 |            8 |           1 |           0 |           0 |           0 |           0 |           0 |           0 |           0 |           0 |           0 |           0 |           0 |           0 |           0 |           0 |           0 |           0 |           0 |           0 |           0 |           0 |           0 |           0 |           0 |           0 |           0 |           0 |           0 |           0 |           0 |           0 |           0 |           0 |           0 |           0 |           0 |           0 |               1 |
| 57 | ADI     | 3S PER AN HC 3P-BLACK/BLACK/BLACK  | 3S         | bukan_warna |           1 |            8 |           1 |           0 |           0 |           0 |           0 |           0 |           0 |           0 |           0 |           0 |           0 |           0 |           0 |           0 |           0 |           0 |           0 |           0 |           0 |           0 |           0 |           0 |           0 |           0 |           0 |           0 |           0 |           0 |           0 |           0 |           0 |           0 |           0 |           0 |           0 |           0 |           0 |               0 |
| 58 | ADI     | 3S PER AN HC 3P-BLACK/BLACK/BLACK  | PER        | bukan_warna |           2 |            8 |           1 |           0 |           0 |           0 |           0 |           0 |           0 |           0 |           0 |           0 |           0 |           0 |           0 |           0 |           0 |           0 |           0 |           0 |           0 |           0 |           0 |           0 |           0 |           0 |           0 |           0 |           0 |           0 |           0 |           0 |           0 |           0 |           0 |           0 |           0 |           0 |           0 |               0 |
| 59 | ADI     | 3S PER AN HC 3P-BLACK/BLACK/BLACK  | AN         | bukan_warna |           3 |            8 |           1 |           0 |           0 |           0 |           0 |           0 |           0 |           0 |           0 |           0 |           0 |           0 |           0 |           0 |           0 |           0 |           0 |           0 |           0 |           0 |           0 |           0 |           0 |           0 |           0 |           0 |           0 |           0 |           0 |           0 |           0 |           0 |           0 |           0 |           0 |           0 |           0 |               0 |
| 60 | ADI     | 3S PER AN HC 3P-BLACK/BLACK/BLACK  | HC         | bukan_warna |           4 |            8 |           1 |           0 |           0 |           0 |           0 |           0 |           0 |           0 |           0 |           0 |           0 |           0 |           0 |           0 |           0 |           0 |           0 |           0 |           0 |           0 |           0 |           0 |           0 |           0 |           0 |           0 |           0 |           0 |           0 |           0 |           0 |           0 |           0 |           0 |           0 |           0 |           0 |               0 |
| 61 | ADI     | 3S PER AN HC 3P-BLACK/BLACK/BLACK  | 3P         | bukan_warna |           5 |            8 |           1 |           0 |           0 |           0 |           0 |           0 |           0 |           0 |           0 |           0 |           0 |           0 |           0 |           0 |           0 |           0 |           0 |           0 |           0 |           0 |           0 |           0 |           0 |           0 |           0 |           0 |           0 |           0 |           0 |           0 |           0 |           0 |           0 |           0 |           0 |           0 |           0 |               0 |
| 62 | ADI     | 3S PER AN HC 3P-BLACK/BLACK/BLACK  | BLACK      | warna       |           6 |            8 |           1 |           0 |           0 |           0 |           0 |           0 |           0 |           0 |           0 |           0 |           0 |           0 |           0 |           0 |           0 |           0 |           0 |           0 |           0 |           0 |           0 |           0 |           0 |           0 |           0 |           0 |           0 |           0 |           0 |           0 |           0 |           0 |           0 |           0 |           0 |           0 |           0 |               1 |
| 63 | ADI     | 3S PER AN HC 3P-BLACK/BLACK/BLACK  | BLACK      | warna       |           7 |            8 |           1 |           0 |           0 |           0 |           0 |           0 |           0 |           0 |           0 |           0 |           0 |           0 |           0 |           0 |           0 |           0 |           0 |           0 |           0 |           0 |           0 |           0 |           0 |           0 |           0 |           0 |           0 |           0 |           0 |           0 |           0 |           0 |           0 |           0 |           0 |           0 |           0 |               1 |
| 64 | ADI     | 3S PER AN HC 3P-BLACK/BLACK/BLACK  | BLACK      | warna       |           8 |            8 |           1 |           0 |           0 |           0 |           0 |           0 |           0 |           0 |           0 |           0 |           0 |           0 |           0 |           0 |           0 |           0 |           0 |           0 |           0 |           0 |           0 |           0 |           0 |           0 |           0 |           0 |           0 |           0 |           0 |           0 |           0 |           0 |           0 |           0 |           0 |           0 |           0 |               1 |
| 65 | ADI     | 3S PER AN HC 1P-WHITE/WHITE/BLACK  | 3S         | bukan_warna |           1 |            8 |           1 |           0 |           0 |           0 |           0 |           0 |           0 |           0 |           0 |           0 |           0 |           0 |           0 |           0 |           0 |           0 |           0 |           0 |           0 |           0 |           0 |           0 |           0 |           0 |           0 |           0 |           0 |           0 |           0 |           0 |           0 |           0 |           0 |           0 |           0 |           0 |           0 |               0 |
| 66 | ADI     | 3S PER AN HC 1P-WHITE/WHITE/BLACK  | PER        | bukan_warna |           2 |            8 |           1 |           0 |           0 |           0 |           0 |           0 |           0 |           0 |           0 |           0 |           0 |           0 |           0 |           0 |           0 |           0 |           0 |           0 |           0 |           0 |           0 |           0 |           0 |           0 |           0 |           0 |           0 |           0 |           0 |           0 |           0 |           0 |           0 |           0 |           0 |           0 |           0 |               0 |
| 67 | ADI     | 3S PER AN HC 1P-WHITE/WHITE/BLACK  | AN         | bukan_warna |           3 |            8 |           1 |           0 |           0 |           0 |           0 |           0 |           0 |           0 |           0 |           0 |           0 |           0 |           0 |           0 |           0 |           0 |           0 |           0 |           0 |           0 |           0 |           0 |           0 |           0 |           0 |           0 |           0 |           0 |           0 |           0 |           0 |           0 |           0 |           0 |           0 |           0 |           0 |               0 |
| 68 | ADI     | 3S PER AN HC 1P-WHITE/WHITE/BLACK  | HC         | bukan_warna |           4 |            8 |           1 |           0 |           0 |           0 |           0 |           0 |           0 |           0 |           0 |           0 |           0 |           0 |           0 |           0 |           0 |           0 |           0 |           0 |           0 |           0 |           0 |           0 |           0 |           0 |           0 |           0 |           0 |           0 |           0 |           0 |           0 |           0 |           0 |           0 |           0 |           0 |           0 |               0 |
| 69 | ADI     | 3S PER AN HC 1P-WHITE/WHITE/BLACK  | 1P         | bukan_warna |           5 |            8 |           1 |           0 |           0 |           0 |           0 |           0 |           0 |           0 |           0 |           0 |           0 |           0 |           0 |           0 |           0 |           0 |           0 |           0 |           0 |           0 |           0 |           0 |           0 |           0 |           0 |           0 |           0 |           0 |           0 |           0 |           0 |           0 |           0 |           0 |           0 |           0 |           0 |               0 |
| 70 | ADI     | 3S PER AN HC 1P-WHITE/WHITE/BLACK  | WHITE      | warna       |           6 |            8 |           1 |           0 |           0 |           0 |           0 |           0 |           0 |           0 |           0 |           0 |           0 |           0 |           0 |           0 |           0 |           0 |           0 |           0 |           0 |           0 |           0 |           0 |           0 |           0 |           0 |           0 |           0 |           0 |           0 |           0 |           0 |           0 |           0 |           0 |           0 |           0 |           0 |               1 |
| 71 | ADI     | 3S PER AN HC 1P-WHITE/WHITE/BLACK  | WHITE      | warna       |           7 |            8 |           1 |           0 |           0 |           0 |           0 |           0 |           0 |           0 |           0 |           0 |           0 |           0 |           0 |           0 |           0 |           0 |           0 |           0 |           0 |           0 |           0 |           0 |           0 |           0 |           0 |           0 |           0 |           0 |           0 |           0 |           0 |           0 |           0 |           0 |           0 |           0 |           0 |               1 |
| 72 | ADI     | 3S PER AN HC 1P-WHITE/WHITE/BLACK  | BLACK      | warna       |           8 |            8 |           1 |           0 |           0 |           0 |           0 |           0 |           0 |           0 |           0 |           0 |           0 |           0 |           0 |           0 |           0 |           0 |           0 |           0 |           0 |           0 |           0 |           0 |           0 |           0 |           0 |           0 |           0 |           0 |           0 |           0 |           0 |           0 |           0 |           0 |           0 |           0 |           0 |               1 |
| 73 | ADI     | 3S PER AN HC 1P-BLACK/BLACK/WHITE  | 3S         | bukan_warna |           1 |            8 |           1 |           0 |           0 |           0 |           0 |           0 |           0 |           0 |           0 |           0 |           0 |           0 |           0 |           0 |           0 |           0 |           0 |           0 |           0 |           0 |           0 |           0 |           0 |           0 |           0 |           0 |           0 |           0 |           0 |           0 |           0 |           0 |           0 |           0 |           0 |           0 |           0 |               0 |
| 74 | ADI     | 3S PER AN HC 1P-BLACK/BLACK/WHITE  | PER        | bukan_warna |           2 |            8 |           1 |           0 |           0 |           0 |           0 |           0 |           0 |           0 |           0 |           0 |           0 |           0 |           0 |           0 |           0 |           0 |           0 |           0 |           0 |           0 |           0 |           0 |           0 |           0 |           0 |           0 |           0 |           0 |           0 |           0 |           0 |           0 |           0 |           0 |           0 |           0 |           0 |               0 |
| 75 | ADI     | 3S PER AN HC 1P-BLACK/BLACK/WHITE  | AN         | bukan_warna |           3 |            8 |           1 |           0 |           0 |           0 |           0 |           0 |           0 |           0 |           0 |           0 |           0 |           0 |           0 |           0 |           0 |           0 |           0 |           0 |           0 |           0 |           0 |           0 |           0 |           0 |           0 |           0 |           0 |           0 |           0 |           0 |           0 |           0 |           0 |           0 |           0 |           0 |           0 |               0 |
| 76 | ADI     | 3S PER AN HC 1P-BLACK/BLACK/WHITE  | HC         | bukan_warna |           4 |            8 |           1 |           0 |           0 |           0 |           0 |           0 |           0 |           0 |           0 |           0 |           0 |           0 |           0 |           0 |           0 |           0 |           0 |           0 |           0 |           0 |           0 |           0 |           0 |           0 |           0 |           0 |           0 |           0 |           0 |           0 |           0 |           0 |           0 |           0 |           0 |           0 |           0 |               0 |
| 77 | ADI     | 3S PER AN HC 1P-BLACK/BLACK/WHITE  | 1P         | bukan_warna |           5 |            8 |           1 |           0 |           0 |           0 |           0 |           0 |           0 |           0 |           0 |           0 |           0 |           0 |           0 |           0 |           0 |           0 |           0 |           0 |           0 |           0 |           0 |           0 |           0 |           0 |           0 |           0 |           0 |           0 |           0 |           0 |           0 |           0 |           0 |           0 |           0 |           0 |           0 |               0 |
| 78 | ADI     | 3S PER AN HC 1P-BLACK/BLACK/WHITE  | BLACK      | warna       |           6 |            8 |           1 |           0 |           0 |           0 |           0 |           0 |           0 |           0 |           0 |           0 |           0 |           0 |           0 |           0 |           0 |           0 |           0 |           0 |           0 |           0 |           0 |           0 |           0 |           0 |           0 |           0 |           0 |           0 |           0 |           0 |           0 |           0 |           0 |           0 |           0 |           0 |           0 |               1 |
| 79 | ADI     | 3S PER AN HC 1P-BLACK/BLACK/WHITE  | BLACK      | warna       |           7 |            8 |           1 |           0 |           0 |           0 |           0 |           0 |           0 |           0 |           0 |           0 |           0 |           0 |           0 |           0 |           0 |           0 |           0 |           0 |           0 |           0 |           0 |           0 |           0 |           0 |           0 |           0 |           0 |           0 |           0 |           0 |           0 |           0 |           0 |           0 |           0 |           0 |           0 |               1 |
| 80 | ADI     | 3S PER AN HC 1P-BLACK/BLACK/WHITE  | WHITE      | warna       |           8 |            8 |           1 |           0 |           0 |           0 |           0 |           0 |           0 |           0 |           0 |           0 |           0 |           0 |           0 |           0 |           0 |           0 |           0 |           0 |           0 |           0 |           0 |           0 |           0 |           0 |           0 |           0 |           0 |           0 |           0 |           0 |           0 |           0 |           0 |           0 |           0 |           0 |           0 |               1 |
| 81 | ADI     | 3S PER CR HC 3P-WHITE/WHITE/WHITE  | 3S         | bukan_warna |           1 |            8 |           1 |           0 |           0 |           0 |           0 |           0 |           0 |           0 |           0 |           0 |           0 |           0 |           0 |           0 |           0 |           0 |           0 |           0 |           0 |           0 |           0 |           0 |           0 |           0 |           0 |           0 |           0 |           0 |           0 |           0 |           0 |           0 |           0 |           0 |           0 |           0 |           0 |               0 |
| 82 | ADI     | 3S PER CR HC 3P-WHITE/WHITE/WHITE  | PER        | bukan_warna |           2 |            8 |           1 |           0 |           0 |           0 |           0 |           0 |           0 |           0 |           0 |           0 |           0 |           0 |           0 |           0 |           0 |           0 |           0 |           0 |           0 |           0 |           0 |           0 |           0 |           0 |           0 |           0 |           0 |           0 |           0 |           0 |           0 |           0 |           0 |           0 |           0 |           0 |           0 |               0 |
| 83 | ADI     | 3S PER CR HC 3P-WHITE/WHITE/WHITE  | CR         | bukan_warna |           3 |            8 |           1 |           0 |           0 |           0 |           0 |           0 |           0 |           0 |           0 |           0 |           0 |           0 |           0 |           0 |           0 |           0 |           0 |           0 |           0 |           0 |           0 |           0 |           0 |           0 |           0 |           0 |           0 |           0 |           0 |           0 |           0 |           0 |           0 |           0 |           0 |           0 |           0 |               0 |
| 84 | ADI     | 3S PER CR HC 3P-WHITE/WHITE/WHITE  | HC         | bukan_warna |           4 |            8 |           1 |           0 |           0 |           0 |           0 |           0 |           0 |           0 |           0 |           0 |           0 |           0 |           0 |           0 |           0 |           0 |           0 |           0 |           0 |           0 |           0 |           0 |           0 |           0 |           0 |           0 |           0 |           0 |           0 |           0 |           0 |           0 |           0 |           0 |           0 |           0 |           0 |               0 |
| 85 | ADI     | 3S PER CR HC 3P-WHITE/WHITE/WHITE  | 3P         | bukan_warna |           5 |            8 |           1 |           0 |           0 |           0 |           0 |           0 |           0 |           0 |           0 |           0 |           0 |           0 |           0 |           0 |           0 |           0 |           0 |           0 |           0 |           0 |           0 |           0 |           0 |           0 |           0 |           0 |           0 |           0 |           0 |           0 |           0 |           0 |           0 |           0 |           0 |           0 |           0 |               0 |
| 86 | ADI     | 3S PER CR HC 3P-WHITE/WHITE/WHITE  | WHITE      | warna       |           6 |            8 |           1 |           0 |           0 |           0 |           0 |           0 |           0 |           0 |           0 |           0 |           0 |           0 |           0 |           0 |           0 |           0 |           0 |           0 |           0 |           0 |           0 |           0 |           0 |           0 |           0 |           0 |           0 |           0 |           0 |           0 |           0 |           0 |           0 |           0 |           0 |           0 |           0 |               1 |
| 87 | ADI     | 3S PER CR HC 3P-WHITE/WHITE/WHITE  | WHITE      | warna       |           7 |            8 |           1 |           0 |           0 |           0 |           0 |           0 |           0 |           0 |           0 |           0 |           0 |           0 |           0 |           0 |           0 |           0 |           0 |           0 |           0 |           0 |           0 |           0 |           0 |           0 |           0 |           0 |           0 |           0 |           0 |           0 |           0 |           0 |           0 |           0 |           0 |           0 |           0 |               1 |
| 88 | ADI     | 3S PER CR HC 3P-WHITE/WHITE/WHITE  | WHITE      | warna       |           8 |            8 |           1 |           0 |           0 |           0 |           0 |           0 |           0 |           0 |           0 |           0 |           0 |           0 |           0 |           0 |           0 |           0 |           0 |           0 |           0 |           0 |           0 |           0 |           0 |           0 |           0 |           0 |           0 |           0 |           0 |           0 |           0 |           0 |           0 |           0 |           0 |           0 |           0 |               1 |
| 89 | ADI     | 3S PER CR HC 3P-BLACK/BLACK/WHITE  | 3S         | bukan_warna |           1 |            8 |           1 |           0 |           0 |           0 |           0 |           0 |           0 |           0 |           0 |           0 |           0 |           0 |           0 |           0 |           0 |           0 |           0 |           0 |           0 |           0 |           0 |           0 |           0 |           0 |           0 |           0 |           0 |           0 |           0 |           0 |           0 |           0 |           0 |           0 |           0 |           0 |           0 |               0 |
| 90 | ADI     | 3S PER CR HC 3P-BLACK/BLACK/WHITE  | PER        | bukan_warna |           2 |            8 |           1 |           0 |           0 |           0 |           0 |           0 |           0 |           0 |           0 |           0 |           0 |           0 |           0 |           0 |           0 |           0 |           0 |           0 |           0 |           0 |           0 |           0 |           0 |           0 |           0 |           0 |           0 |           0 |           0 |           0 |           0 |           0 |           0 |           0 |           0 |           0 |           0 |               0 |
| 91 | ADI     | 3S PER CR HC 3P-BLACK/BLACK/WHITE  | CR         | bukan_warna |           3 |            8 |           1 |           0 |           0 |           0 |           0 |           0 |           0 |           0 |           0 |           0 |           0 |           0 |           0 |           0 |           0 |           0 |           0 |           0 |           0 |           0 |           0 |           0 |           0 |           0 |           0 |           0 |           0 |           0 |           0 |           0 |           0 |           0 |           0 |           0 |           0 |           0 |           0 |               0 |
| 92 | ADI     | 3S PER CR HC 3P-BLACK/BLACK/WHITE  | HC         | bukan_warna |           4 |            8 |           1 |           0 |           0 |           0 |           0 |           0 |           0 |           0 |           0 |           0 |           0 |           0 |           0 |           0 |           0 |           0 |           0 |           0 |           0 |           0 |           0 |           0 |           0 |           0 |           0 |           0 |           0 |           0 |           0 |           0 |           0 |           0 |           0 |           0 |           0 |           0 |           0 |               0 |
| 93 | ADI     | 3S PER CR HC 3P-BLACK/BLACK/WHITE  | 3P         | bukan_warna |           5 |            8 |           1 |           0 |           0 |           0 |           0 |           0 |           0 |           0 |           0 |           0 |           0 |           0 |           0 |           0 |           0 |           0 |           0 |           0 |           0 |           0 |           0 |           0 |           0 |           0 |           0 |           0 |           0 |           0 |           0 |           0 |           0 |           0 |           0 |           0 |           0 |           0 |           0 |               0 |
| 94 | ADI     | 3S PER CR HC 3P-BLACK/BLACK/WHITE  | BLACK      | warna       |           6 |            8 |           1 |           0 |           0 |           0 |           0 |           0 |           0 |           0 |           0 |           0 |           0 |           0 |           0 |           0 |           0 |           0 |           0 |           0 |           0 |           0 |           0 |           0 |           0 |           0 |           0 |           0 |           0 |           0 |           0 |           0 |           0 |           0 |           0 |           0 |           0 |           0 |           0 |               1 |
| 95 | ADI     | 3S PER CR HC 3P-BLACK/BLACK/WHITE  | BLACK      | warna       |           7 |            8 |           1 |           0 |           0 |           0 |           0 |           0 |           0 |           0 |           0 |           0 |           0 |           0 |           0 |           0 |           0 |           0 |           0 |           0 |           0 |           0 |           0 |           0 |           0 |           0 |           0 |           0 |           0 |           0 |           0 |           0 |           0 |           0 |           0 |           0 |           0 |           0 |           0 |               1 |
| 96 | ADI     | 3S PER CR HC 3P-BLACK/BLACK/WHITE  | WHITE      | warna       |           8 |            8 |           1 |           0 |           0 |           0 |           0 |           0 |           0 |           0 |           0 |           0 |           0 |           0 |           0 |           0 |           0 |           0 |           0 |           0 |           0 |           0 |           0 |           0 |           0 |           0 |           0 |           0 |           0 |           0 |           0 |           0 |           0 |           0 |           0 |           0 |           0 |           0 |           0 |               1 |
| 97 | ADI     | 3S PER CR HC 1P-WHITE/WHITE/BLACK  | 3S         | bukan_warna |           1 |            8 |           1 |           0 |           0 |           0 |           0 |           0 |           0 |           0 |           0 |           0 |           0 |           0 |           0 |           0 |           0 |           0 |           0 |           0 |           0 |           0 |           0 |           0 |           0 |           0 |           0 |           0 |           0 |           0 |           0 |           0 |           0 |           0 |           0 |           0 |           0 |           0 |           0 |               0 |
| 98 | ADI     | 3S PER CR HC 1P-WHITE/WHITE/BLACK  | PER        | bukan_warna |           2 |            8 |           1 |           0 |           0 |           0 |           0 |           0 |           0 |           0 |           0 |           0 |           0 |           0 |           0 |           0 |           0 |           0 |           0 |           0 |           0 |           0 |           0 |           0 |           0 |           0 |           0 |           0 |           0 |           0 |           0 |           0 |           0 |           0 |           0 |           0 |           0 |           0 |           0 |               0 |
| 99 | ADI     | 3S PER CR HC 1P-WHITE/WHITE/BLACK  | CR         | bukan_warna |           3 |            8 |           1 |           0 |           0 |           0 |           0 |           0 |           0 |           0 |           0 |           0 |           0 |           0 |           0 |           0 |           0 |           0 |           0 |           0 |           0 |           0 |           0 |           0 |           0 |           0 |           0 |           0 |           0 |           0 |           0 |           0 |           0 |           0 |           0 |           0 |           0 |           0 |           0 |               0 |


### Konversi Data ke dalam Train dan Test untuk Model 0

Data akan dibagi ke dalam train dan test data menggunakan metode `train_test_split` dari modul *sklearn.model_selection* dengan menggunakan rasio dan keacakan yang telah ditentukan di variabel global (*RASIO_TEST_TRAIN* dan *RANDOM_STATE*).


```python
# Menyimpan header data
data_header = data_encoded[['kata', 'brand', 'urut_kata', 'total_kata', 'label']].columns

"""
Model 0 adalah MultinomialNB yang akan menggunakan feature_extraction TfIdfVectorizer
dimana TfIdfVectorizer hanya dapat menerima satu kolom data yang akan diubah menjadi vector
(angka), kecuali kita dapat menggabungkan kembali brand kata dan kolom kolom lainnya ke dalam
satu kolom seperti['NIK GREEN 1 0 0 0 1'] alih - alih [['NIK', 'GREEN', '1', '0', '0', '0', '1']]
Maka untuk Model 0 kita tetap akan hanya menggunakan kolom 'kata' sebagai fitur.
kolom 'nama_artikel', 'brand', 'urut_kata' 'total_kata' dan 'label' sebenarnya tidak akan 
digunakan untuk training, namun pada train_test_split ini kita akan menyimpan brand untuk 
display hasil prediksi berbanding dengan target label (ground truth)
"""
train_data_mnb, test_data_mnb, train_target_mnb, test_target_mnb = train_test_split(data_encoded[['kata', 'nama_artikel', 'brand', 'urut_kata', 'total_kata', 'label']],
                                                                                    data_encoded['label_encoded'],
                                                                                    test_size=RASIO_TEST_TRAIN,
                                                                                    random_state=RANDOM_STATE)

# Untuk model lainnya kita akan menggunakan semua fitur minus 'brand', 'nama_artikel', 'label' dan 'label_encoded' .drop
train_data, test_data, train_target, test_target = train_test_split(data_encoded.drop(['brand', 'nama_artikel', 'label', 'label_encoded'], axis=1),
                                                                    data_encoded['label_encoded'],
                                                                    test_size=RASIO_TEST_TRAIN,
                                                                    random_state=RANDOM_STATE)
```


```python
# Shape dari train_data_mnb, test_data_mnb, train_target_mnb dan test_target_mnb
train_data_mnb.shape, test_data_mnb.shape, train_target_mnb.shape, test_target_mnb.shape
```




    ((45400, 6), (11351, 6), (45400,), (11351,))




```python
# Shape dari train_data, test_data, train_target dan test_target
train_data.shape, test_data.shape, train_target.shape, test_target.shape
```




    ((45400, 40), (11351, 40), (45400,), (11351,))




```python
# Eksplorasi contoh hasil split train dan test
train_target_unik, train_target_hitung = np.unique(train_target_mnb, return_counts=True)
test_target_unik, test_target_hitung = np.unique(test_target_mnb, return_counts=True)
print('2 data pertama di train_data_mnb:')
with pd.option_context('display.max_columns', None):
    print(train_data_mnb.iloc[:2].to_markdown())
print('2 data pertama di train_data:')
with pd.option_context('display.max_columns', None):
    print(train_data[:2].to_markdown())
print('\n2 data pertama di train_target (mnb & non-mnb, sama):')
with pd.option_context('display.max_columns', None):
    print(train_target[:2].to_markdown()) 
print('2 data pertama di test_data_mnb:')
with pd.option_context('display.max_columns', None):
    print(test_data_mnb.iloc[:2].to_markdown())
print('2 data pertama di test_data:')
with pd.option_context('display.max_columns', None):
    print(test_data[:2].to_markdown())
print('2 data pertama di test_target (mnb & non-mnb, sama):')
with pd.option_context('display.max_columns', None):
    print(test_target[:2].to_markdown())
train_target_distribusi = np.column_stack((train_target_unik, train_target_hitung))
test_target_distribusi = np.column_stack((test_target_unik, test_target_hitung))
print(f'Distribusi label (target) di train: \n{train_target_distribusi}\n')
print(f'Distribusi label (target) di test: \n{test_target_distribusi}\n')
print('Dimana label 0 = bukan warna dan label 1 = warna')
```

    2 data pertama di train_data_mnb:
|       | kata   | nama_artikel                                          | brand   |   urut_kata |   total_kata | label   |
|------:|:-------|:------------------------------------------------------|:--------|------------:|-------------:|:--------|
| 43886 | GREY   | AS W NK DRY TANK DFC YOGA FOIL-BARELY ROSE/SMOKE GREY | NIK     |          12 |           12 | warna   |
| 14859 | BLACK  | PRED SG CLB-BLACK                                     | ADI     |           4 |            4 | warna   |

    2 data pertama di train_data:
|       | kata   |   urut_kata |   total_kata |   brand_ADI |   brand_ADS |   brand_AGL |   brand_AND |   brand_ASC |   brand_BAL |   brand_BBC |   brand_BEA |   brand_CAO |   brand_CIT |   brand_CRP |   brand_DOM |   brand_FIS |   brand_GUE |   brand_HER |   brand_JAS |   brand_KIP |   brand_NEW |   brand_NFA |   brand_NFC |   brand_NFL |   brand_NIB |   brand_NIC |   brand_NIK |   brand_NPS |   brand_ODD |   brand_PBY |   brand_PSB |   brand_PTG |   brand_PUM |   brand_REL |   brand_SAU |   brand_SOC |   brand_STN |   brand_UME |   brand_VAP |   brand_WAR |
|------:|:-------|------------:|-------------:|------------:|------------:|------------:|------------:|------------:|------------:|------------:|------------:|------------:|------------:|------------:|------------:|------------:|------------:|------------:|------------:|------------:|------------:|------------:|------------:|------------:|------------:|------------:|------------:|------------:|------------:|------------:|------------:|------------:|------------:|------------:|------------:|------------:|------------:|------------:|------------:|------------:|
| 43886 | GREY   |          12 |           12 |           0 |           0 |           0 |           0 |           0 |           0 |           0 |           0 |           0 |           0 |           0 |           0 |           0 |           0 |           0 |           0 |           0 |           0 |           0 |           0 |           0 |           0 |           0 |           1 |           0 |           0 |           0 |           0 |           0 |           0 |           0 |           0 |           0 |           0 |           0 |           0 |           0 |
| 14859 | BLACK  |           4 |            4 |           1 |           0 |           0 |           0 |           0 |           0 |           0 |           0 |           0 |           0 |           0 |           0 |           0 |           0 |           0 |           0 |           0 |           0 |           0 |           0 |           0 |           0 |           0 |           0 |           0 |           0 |           0 |           0 |           0 |           0 |           0 |           0 |           0 |           0 |           0 |           0 |           0 |

    2 data pertama di train_target (mnb & non-mnb, sama):
|       |   label_encoded |
|------:|----------------:|
| 43886 |               1 |
| 14859 |               1 |

    2 data pertama di test_data_mnb:
|       | kata   | nama_artikel                        | brand   |   urut_kata |   total_kata | label       |
|------:|:-------|:------------------------------------|:--------|------------:|-------------:|:------------|
| 16829 | SESOYE | FORTAPLAY AC I-CORBLU/SESOYE/CONAVY | ADI     |           5 |            6 | warna       |
|  5081 | GHOST  | GHOST LESTO-GREY/CBLACK             | ADI     |           1 |            4 | bukan_warna |

    2 data pertama di test_data:
|       | kata   |   urut_kata |   total_kata |   brand_ADI |   brand_ADS |   brand_AGL |   brand_AND |   brand_ASC |   brand_BAL |   brand_BBC |   brand_BEA |   brand_CAO |   brand_CIT |   brand_CRP |   brand_DOM |   brand_FIS |   brand_GUE |   brand_HER |   brand_JAS |   brand_KIP |   brand_NEW |   brand_NFA |   brand_NFC |   brand_NFL |   brand_NIB |   brand_NIC |   brand_NIK |   brand_NPS |   brand_ODD |   brand_PBY |   brand_PSB |   brand_PTG |   brand_PUM |   brand_REL |   brand_SAU |   brand_SOC |   brand_STN |   brand_UME |   brand_VAP |   brand_WAR |
|------:|:-------|------------:|-------------:|------------:|------------:|------------:|------------:|------------:|------------:|------------:|------------:|------------:|------------:|------------:|------------:|------------:|------------:|------------:|------------:|------------:|------------:|------------:|------------:|------------:|------------:|------------:|------------:|------------:|------------:|------------:|------------:|------------:|------------:|------------:|------------:|------------:|------------:|------------:|------------:|------------:|
| 16829 | SESOYE |           5 |            6 |           1 |           0 |           0 |           0 |           0 |           0 |           0 |           0 |           0 |           0 |           0 |           0 |           0 |           0 |           0 |           0 |           0 |           0 |           0 |           0 |           0 |           0 |           0 |           0 |           0 |           0 |           0 |           0 |           0 |           0 |           0 |           0 |           0 |           0 |           0 |           0 |           0 |
|  5081 | GHOST  |           1 |            4 |           1 |           0 |           0 |           0 |           0 |           0 |           0 |           0 |           0 |           0 |           0 |           0 |           0 |           0 |           0 |           0 |           0 |           0 |           0 |           0 |           0 |           0 |           0 |           0 |           0 |           0 |           0 |           0 |           0 |           0 |           0 |           0 |           0 |           0 |           0 |           0 |           0 |

    2 data pertama di test_target (mnb & non-mnb, sama):
|       |   label_encoded |
|------:|----------------:|
| 16829 |               1 |
|  5081 |               0 |

    Distribusi label (target) di train: 
    [[    0 27355]
     [    1 18045]]
    
    Distribusi label (target) di test: 
    [[   0 6819]
     [   1 4532]]
    
    Dimana label 0 = bukan warna dan label 1 = warna
    

## Model 0: Model Dasar

Model pertama yang akan kita buat adalah model *Multinomial Naive-Bayes* yang akan mengkategorisasikan *input* ke dalam kategori *output*. *Multinomial Naive-Bayes* adalah sebuah algoritma dengan metode *supervised learning* yang paling umum digunakan dalam pengkategorisasian data tekstual.
Pada dasarnya *Naive-Bayes* merupakan algoritma yang menghitung probabilitas dari sebuah event (*output*) berdasarkan probabilitas akumulatif kejadian dari event sebelumnya. Secara singkat algoritma ini akan mempelajari berapa probabilitas dari sebuah kata, misalkan 'ADISSAGE' adalah sebuah label `bukan_warna` berdasarkan probabilitas kejadian 'ADISSAGE' adalah `bukan_warna` pada event - event sebelumnya.

Formula dari probabilitias algoritma *Naive-Bayes*:

$P(A|B) = \frac{P(A) * P(B|A)}{P(B)}$

Sebelum melakukan training menggunakan algoritma *Multinomial Naive-Bayes* kita perlu untuk merubah data kata menjadi bentuk numerik yang kali ini akan dikonversi menggunakan metode TF-IDF (*Term Frequency-Inverse Document Frequency*). TF-IDF sendiri merupakan metode yang akan berusaha memvaluasi nilai relevansi dan frekuensi dari sebuah kata dalam sekumpulan dokumen. *Term Frequency* merujuk pada seberapa sering sebuah kata muncul dalam 1 dokumen, sedangkan *Inverse Document Frequency* adalah perhitungan logaritma dari jumlah seluruh dokumen dibagi dengan jumlah dokumen dengan kata yang dimaksud terdapat di dalamnya. Hasil perhitungan dari TF dan IDF ini akan dikalikan untuk mendapatkan nilai dari seberapa sering dan seberapa relevan nilai dari sebuah kata. Misalkan 'ADISSAGE' sering muncul dalam 1 dokumen tapi tidak terlalu banyak muncul di dokumen - dokumen lainnya, maka hal ini dapat mengindikasikan bahwa kata 'ADISSAGE' mungkin memiliki relevansi yang tinggi dalam kategorisasi sebuah dokumen, sebaliknya jika kata 'WHITE' sering muncul di 1 dokumen dan juga sering muncul di dokumen - dokumen lainnya, maka kata 'WHITE' ini mungkin merupakan sebuah kata yang umum dan memiliki nilai relevansi yang rendah dalam pengkategorisasian sebuah dokumen.

Untuk lebih lengkapnya mengenai *Naive-Bayes* dan TF-IDF dapat merujuk pada sumber berikut:

* https://towardsdatascience.com/naive-bayes-classifier-81d512f50a7c
* https://monkeylearn.com/blog/what-is-tf-idf/



```python
# Membuat pipeline untuk mengubah kata ke dalam tf-idf
model_0 = Pipeline([
    ("tf-idf", TfidfVectorizer()),
    ("clf", MultinomialNB())
])

# Fit pipeline dengan data training
model_0.fit(X=np.squeeze(train_data_mnb.iloc[:, 0]), y=train_target_mnb)
```




<style>#sk-container-id-2 {color: black;background-color: white;}#sk-container-id-2 pre{padding: 0;}#sk-container-id-2 div.sk-toggleable {background-color: white;}#sk-container-id-2 label.sk-toggleable__label {cursor: pointer;display: block;width: 100%;margin-bottom: 0;padding: 0.3em;box-sizing: border-box;text-align: center;}#sk-container-id-2 label.sk-toggleable__label-arrow:before {content: "▸";float: left;margin-right: 0.25em;color: #696969;}#sk-container-id-2 label.sk-toggleable__label-arrow:hover:before {color: black;}#sk-container-id-2 div.sk-estimator:hover label.sk-toggleable__label-arrow:before {color: black;}#sk-container-id-2 div.sk-toggleable__content {max-height: 0;max-width: 0;overflow: hidden;text-align: left;background-color: #f0f8ff;}#sk-container-id-2 div.sk-toggleable__content pre {margin: 0.2em;color: black;border-radius: 0.25em;background-color: #f0f8ff;}#sk-container-id-2 input.sk-toggleable__control:checked~div.sk-toggleable__content {max-height: 200px;max-width: 100%;overflow: auto;}#sk-container-id-2 input.sk-toggleable__control:checked~label.sk-toggleable__label-arrow:before {content: "▾";}#sk-container-id-2 div.sk-estimator input.sk-toggleable__control:checked~label.sk-toggleable__label {background-color: #d4ebff;}#sk-container-id-2 div.sk-label input.sk-toggleable__control:checked~label.sk-toggleable__label {background-color: #d4ebff;}#sk-container-id-2 input.sk-hidden--visually {border: 0;clip: rect(1px 1px 1px 1px);clip: rect(1px, 1px, 1px, 1px);height: 1px;margin: -1px;overflow: hidden;padding: 0;position: absolute;width: 1px;}#sk-container-id-2 div.sk-estimator {font-family: monospace;background-color: #f0f8ff;border: 1px dotted black;border-radius: 0.25em;box-sizing: border-box;margin-bottom: 0.5em;}#sk-container-id-2 div.sk-estimator:hover {background-color: #d4ebff;}#sk-container-id-2 div.sk-parallel-item::after {content: "";width: 100%;border-bottom: 1px solid gray;flex-grow: 1;}#sk-container-id-2 div.sk-label:hover label.sk-toggleable__label {background-color: #d4ebff;}#sk-container-id-2 div.sk-serial::before {content: "";position: absolute;border-left: 1px solid gray;box-sizing: border-box;top: 0;bottom: 0;left: 50%;z-index: 0;}#sk-container-id-2 div.sk-serial {display: flex;flex-direction: column;align-items: center;background-color: white;padding-right: 0.2em;padding-left: 0.2em;position: relative;}#sk-container-id-2 div.sk-item {position: relative;z-index: 1;}#sk-container-id-2 div.sk-parallel {display: flex;align-items: stretch;justify-content: center;background-color: white;position: relative;}#sk-container-id-2 div.sk-item::before, #sk-container-id-2 div.sk-parallel-item::before {content: "";position: absolute;border-left: 1px solid gray;box-sizing: border-box;top: 0;bottom: 0;left: 50%;z-index: -1;}#sk-container-id-2 div.sk-parallel-item {display: flex;flex-direction: column;z-index: 1;position: relative;background-color: white;}#sk-container-id-2 div.sk-parallel-item:first-child::after {align-self: flex-end;width: 50%;}#sk-container-id-2 div.sk-parallel-item:last-child::after {align-self: flex-start;width: 50%;}#sk-container-id-2 div.sk-parallel-item:only-child::after {width: 0;}#sk-container-id-2 div.sk-dashed-wrapped {border: 1px dashed gray;margin: 0 0.4em 0.5em 0.4em;box-sizing: border-box;padding-bottom: 0.4em;background-color: white;}#sk-container-id-2 div.sk-label label {font-family: monospace;font-weight: bold;display: inline-block;line-height: 1.2em;}#sk-container-id-2 div.sk-label-container {text-align: center;}#sk-container-id-2 div.sk-container {/* jupyter's `normalize.less` sets `[hidden] { display: none; }` but bootstrap.min.css set `[hidden] { display: none !important; }` so we also need the `!important` here to be able to override the default hidden behavior on the sphinx rendered scikit-learn.org. See: https://github.com/scikit-learn/scikit-learn/issues/21755 */display: inline-block !important;position: relative;}#sk-container-id-2 div.sk-text-repr-fallback {display: none;}</style><div id="sk-container-id-2" class="sk-top-container"><div class="sk-text-repr-fallback"><pre>Pipeline(steps=[(&#x27;tf-idf&#x27;, TfidfVectorizer()), (&#x27;clf&#x27;, MultinomialNB())])</pre><b>In a Jupyter environment, please rerun this cell to show the HTML representation or trust the notebook. <br />On GitHub, the HTML representation is unable to render, please try loading this page with nbviewer.org.</b></div><div class="sk-container" hidden><div class="sk-item sk-dashed-wrapped"><div class="sk-label-container"><div class="sk-label sk-toggleable"><input class="sk-toggleable__control sk-hidden--visually" id="sk-estimator-id-4" type="checkbox" ><label for="sk-estimator-id-4" class="sk-toggleable__label sk-toggleable__label-arrow">Pipeline</label><div class="sk-toggleable__content"><pre>Pipeline(steps=[(&#x27;tf-idf&#x27;, TfidfVectorizer()), (&#x27;clf&#x27;, MultinomialNB())])</pre></div></div></div><div class="sk-serial"><div class="sk-item"><div class="sk-estimator sk-toggleable"><input class="sk-toggleable__control sk-hidden--visually" id="sk-estimator-id-5" type="checkbox" ><label for="sk-estimator-id-5" class="sk-toggleable__label sk-toggleable__label-arrow">TfidfVectorizer</label><div class="sk-toggleable__content"><pre>TfidfVectorizer()</pre></div></div></div><div class="sk-item"><div class="sk-estimator sk-toggleable"><input class="sk-toggleable__control sk-hidden--visually" id="sk-estimator-id-6" type="checkbox" ><label for="sk-estimator-id-6" class="sk-toggleable__label sk-toggleable__label-arrow">MultinomialNB</label><div class="sk-toggleable__content"><pre>MultinomialNB()</pre></div></div></div></div></div></div></div>




```python
# Evaluasi model_0 pada data test
skor_model_0 = model_0.score(X=np.squeeze(test_data_mnb.iloc[:, 0]), y=test_target_mnb)
skor_model_0
```




    0.9921592811206061



### Eksplorasi Hasil Model 0
Pada hasil training dengan menggunakan model algoritma *Multinomial Naive-Bayes* kita mendapatkan akurasi sebesar ~99.22%

Secara sekilas model yang pertama ini (model_0) memberikan akurasi yang sangat tinggi dalam membedakan kata `warna` dan `bukan_warna`. Namun secara brand spesifik, akurasi ini mungkin akan lebih buruk karena di beberapa brand terutama 'PUM' kita dapat menjumpai artikel dengan nama misalkan 'PUMA XTG WOVEN PANTS PUMA BLACK-PUMA WHITE' dimana kata PUMA pertama adalah `bukan_warna` namun kata PUMA kedua dan ketiga adalah bagian dari `warna`.

Dengan demikian, nanti kita mungkin akan mengulas lebih mendalam model pertama ini menggunakan dataset yang dipisahkan berdasar brand. Untuk sementara kita akan melanjutkan mengembangkan model - model alternatif untuk pemisahan `bukan_warna` dan `warna` dari nama artikel.


```python
# Membuat prediksi menggunakan data test
model_0_pred = model_0.predict(np.squeeze(test_data_mnb.iloc[:, 0]))
model_0_pred
```




    array([1, 0, 1, ..., 0, 0, 0])




```python
# Membuat fungsi dasar untuk menghitung accuray, precision, recall, f1-score
def hitung_metrik(target, prediksi):
    """
    Menghitung akurasi, presisi, recall dan f1-score dari model klasifikasi biner
    
    Args:
        target: label yang sebenarnya dalam bentuk 1D array
        prediksi: label yang diprediksi dalam bentuk 1D array
        
    Returns:
        nilai accuracy, precision, recall dan f1-score dalam bentuk dictionary
    """
    # Menghitung akurasi model
    model_akurasi = accuracy_score(target, prediksi)
    # Menghitung precision, recall, f1-score dan support dari model
    model_presisi, model_recall, model_f1, _ = precision_recall_fscore_support(target, prediksi, average='weighted')
    
    hasil_model = {'akurasi': model_akurasi,
                   'presisi': model_presisi,
                   'recall': model_recall,
                   'f1-score': model_f1}
    
    return hasil_model
```


```python
# Menghitung metrik dari model_0
model_0_metrik = hitung_metrik(target=test_target_mnb, 
                               prediksi=model_0_pred)
model_0_metrik
```




    {'akurasi': 0.9921592811206061,
     'presisi': 0.9921602131872556,
     'recall': 0.9921592811206061,
     'f1-score': 0.9921562044603152}



**Akurasi** merupakan metrik yang menghitung jumlah prediksi yang benar dibanding total jumlah label yang dijadikan evaluasi (test data, bukan training data).

$\frac{\text{prediksi benar}}{\text{total prediksi}}$

**Presisi** merupakan metrik yang menghitung *true positive* berbanding dengan *true positive* dan *false positive*

$\frac{\text{true positive}}{\text{true positive } + \text{ false positive}}$

**Recall** merupakan metrik yang menghitung *true positive* berbanding dengan *true positive* dan *false negative*

$\frac{\text{true positive}}{\text{true positive } + \text{ false negative}}$

**f1-score** merupakan metrik yang mengabungkan presisi dan recall

$2 * \frac{\text{presisi } * \text{ recall}}{\text{presisi } + \text{ recall}}$

Dimana:

* True Positive (TP): Prediksi `warna` pada target label `warna`
* False Positive (FP): Prediksi `warna` pada target label `bukan_warna`
* True Negative (TN): Prediksi `bukan_warna` pada target label `bukan_warna`
* False Negative (FN): Prediksi `bukan_warna` pada target label `warna`


```python
# Membuat fungsi untuk menampilkan confusion matrix
def plot_conf_matrix(target_label, 
                     prediksi_label, 
                     nama_model,
                     akurasi, 
                     label_titik_x, 
                     label_titik_y):
    """
    Fungsi ini akan menampilkan matrix confusion untuk perbandingan
    target label dan prediksi label dan memahami seberapa kesulitan
    sebuah model melakukan prediksi
    
    Args:
        target_label (list atau 1D-array): label yang sebenarnya dalam bentuk 1D array
        prediksi_label (list atau 1D-array): label yang diprediksi dalam bentuk 1D array
        akurasi (float): akurasi model dalam bentuk float
        label_titik_x (list str): label untuk x-axis dalam bentuk list
        label_titik_y (list str): label untuk y-axis dalam bentuk list
        
        label_titik_x dan label_titik_y, keduanya merupakan list dari sekumpulan
        string dan harus dalam bentuk vektor yang sama
        
    Returns:
        plot_confusion_matrix
    """
    # Membuat confusion matrix
    cf_matrix = confusion_matrix(target_label,
                                 prediksi_label)
    # Pengaturan confusion_matrix menggunakan seaborn
    plot_confusion_matrix = sns.heatmap(cf_matrix, annot=True, fmt='d', cmap='Blues')
    plot_confusion_matrix.set_title(f'Confusion Matrix\n{nama_model}\nAkurasi {akurasi:.2%}', fontsize=18)
    plot_confusion_matrix.set_xlabel('Prediksi Label')
    plot_confusion_matrix.set_ylabel('Target Label')
    plot_confusion_matrix.xaxis.set_ticklabels(label_titik_x)
    plot_confusion_matrix.yaxis.set_ticklabels(label_titik_y)
    
    return plot_confusion_matrix
```


```python
# Menampilkan confusion matrix untuk model_0
plot_conf_matrix(target_label=test_target_mnb,
                 prediksi_label=model_0_pred,
                 nama_model='Model 0 Multinomial Naive Bayes',
                 akurasi=model_0_metrik['akurasi'],
                 label_titik_x=['bukan_warna', 'warna'],
                 label_titik_y=['bukan_warna', 'warna'])
plt.figure(facecolor='w')
plt.show()
```


    
![png](ColorSkim_AI_files/ColorSkim_AI_28_0.png)
    





Pada tabel *Confusion Matrix* di atas kita dapat melihat bahwa Model 0 berhasil memprediksi secara tepat 6,785 kata dengan label `bukan_warna` dan 4,477 kata dengan label `warna`.

Terdapat setidaknya 55 kata yang merupakan `warna` namun diprediksi oleh Model 0 sebagai `bukan_warna` dan 34 kata yang merupakan `bukan_warna` namun diprediksi oleh Model 0 sebagai `warna`


```python
# Membuat fungsi untuk menampilkan kesalahan model dalam dataframe
def df_kesalahan_prediksi(label_encoder, 
                          test_data, 
                          prediksi, 
                          probabilitas_prediksi=None, 
                          order_ulang_header=None):
    """
    Fungsi ini akan menerima objek label encoder sklearn, set test_data
    sebelum modifikasi encoding fitur dan label, prediksi dari model
    serta urutan order_ulang_header jika diperlukan
    
    Args:
        label_encoder (obyek LabelEncoder sklear.preprocessing): obyek label encoder dari sklearn.preprocessing
        test_data (pd.DataFrame): dataframe lengkap sebelum modifikasi fitur dan label
        prediksi (tf.Tensor): tensor dengan shape 1 dimensi yang memuat prediksi model
        order_ulang_header (list): list dengan urutan header yang diinginkan
        
    Returns:
        pd.DataFrame yang diprint dengan format markdown
    """
    inverse_label_encoder = list(label_encoder.inverse_transform([0, 1]))
    
    if order_ulang_header is None:
        data_final = pd.DataFrame(test_data)
    elif type(order_ulang_header) is list:
        data_final = pd.DataFrame(test_data)[order_ulang_header]
    else:
        raise TypeError('order_ulang_header harus berupa list')
    
    kolom_pred = pd.DataFrame(np.int8(prediksi), columns=['prediksi'])
    kolom_prob_pred = pd.DataFrame(probabilitas_prediksi, columns=['probabilitas']) * 100
    data_final['prediksi'] = kolom_pred.iloc[:, 0].tolist()
    
    if probabilitas_prediksi is not None:
        data_final['probabilitas'] = kolom_prob_pred.iloc[:, 0].tolist()
        
    data_final['prediksi'] = data_final['prediksi'].astype(int).map(lambda x: inverse_label_encoder[x])
    
    if probabilitas_prediksi is not None:
        data_final = data_final.loc[data_final['label'] != data_final['prediksi']].sort_values(by='probabilitas', ascending=False)
        data_final['probabilitas'] = data_final['probabilitas'].round(2).astype(str) + '%'
    else:
        data_final = data_final.loc[data_final['label'] != data_final['prediksi']].sort_values(by='prediksi', ascending=False)
        
    with pd.option_context('display.max_rows', None):
        print(data_final.to_markdown())
```


```python
# Menampilkan kesalahan prediksi 
df_kesalahan_prediksi(label_encoder=label_encoder,
                      test_data=test_data_mnb,
                      prediksi=model_0_pred,
                      order_ulang_header=['brand', 
                                          'nama_artikel',
                                          'kata', 
                                          'urut_kata', 
                                          'total_kata', 
                                          'label'])
```

|       | brand   | nama_artikel                                                | kata        |   urut_kata |   total_kata | label       | prediksi    |
|------:|:--------|:------------------------------------------------------------|:------------|------------:|-------------:|:------------|:------------|
| 19265 | BBC     | BB DARK STAR LS KNIT-BLACK                                  | DARK        |           2 |            6 | bukan_warna | warna       |
|  8962 | ADI     | LIN CORE ORG-BLACK                                          | CORE        |           2 |            4 | bukan_warna | warna       |
| 56086 | WAR     | FLAT GLOW IN THE DARK-WHITE                                 | GLOW        |           2 |            6 | bukan_warna | warna       |
| 16288 | ADI     | CLR BLK CRW 2PP-BLACK                                       | BLK         |           2 |            5 | bukan_warna | warna       |
| 52109 | PUM     | CELL STELLAR GLOW WNS PUMA WHITE-PURPLE                     | GLOW        |           3 |            7 | bukan_warna | warna       |
| 19643 | BEA     | SERIES 35-MULTI                                             | 35          |           2 |            3 | bukan_warna | warna       |
| 48153 | PTG     | MISTY DOVE-GREY                                             | DOVE        |           2 |            3 | bukan_warna | warna       |
| 56112 | WAR     | RED BLACK FLAT NON REFLECTIVE-RED/BLACK                     | RED         |           1 |            7 | bukan_warna | warna       |
| 17520 | AGL     | BROWN MOUNTAIN 008 - PEACH                                  | BROWN       |           1 |            4 | bukan_warna | warna       |
| 30654 | NIK     | NIKE AIR ZOOM PEGASUS 35-BRIGHT CRIMSON/ICE BLUE-SAIL       | 35          |           5 |           10 | bukan_warna | warna       |
| 21091 | HER     | POP QUIZ-600D POLY NAVY/ZIP                                 | 600D        |           3 |            6 | bukan_warna | warna       |
| 52114 | PUM     | CELL STELLAR GLOW WN"S PUMA BLACK-PURPLE HEATHER            | GLOW        |           3 |            8 | bukan_warna | warna       |
| 56226 | WAR     | SHOELACES ORANGE OVAL LACES-ORANGE                          | ORANGE      |           2 |            5 | bukan_warna | warna       |
|  8965 | ADI     | LIN CORE CROSSB-BLACK                                       | CORE        |           2 |            4 | bukan_warna | warna       |
| 48805 | PUM     | WMN CORE ROUND BACKPACK LILAC SNOW-VALENTINE                | CORE        |           2 |            7 | bukan_warna | warna       |
| 18208 | BBC     | BB CLEAR SKY L/S T-SHIRT-BLACK                              | CLEAR       |           2 |            8 | bukan_warna | warna       |
| 56116 | WAR     | FULL BLACK FLAT NON REFLECTIVE-BLACK                        | FULL        |           1 |            6 | bukan_warna | warna       |
| 21174 | HER     | HER-HERITAGE-BOSTON RED SOX-(21L)-BAG-US                    | RED         |           4 |            8 | bukan_warna | warna       |
| 48650 | PUM     | WMN CORE SEASONAL ARCHIVE BACKPACK PEACO                    | CORE        |           2 |            6 | bukan_warna | warna       |
| 18940 | BBC     | FULL SCALE CRASH L/S T-SHIRT-ORANGE                         | FULL        |           1 |            8 | bukan_warna | warna       |
| 24154 | NIK     | TS CORE POLO-ROYAL BLUE/WHITE                               | CORE        |           2 |            6 | bukan_warna | warna       |
| 52841 | PUM     | CORE-RUN S/S TEE-LAPIS BLUE                                 | CORE        |           1 |            7 | bukan_warna | warna       |
| 48075 | PTG     | POLKA ORANGE-ORANGE                                         | ORANGE      |           2 |            3 | bukan_warna | warna       |
| 54972 | SAU     | JAZZ VINTAGE-GREY/BLUE/WHITE                                | VINTAGE     |           2 |            5 | bukan_warna | warna       |
|  8735 | ADI     | FULL ZIP-CWHITE                                             | FULL        |           1 |            3 | bukan_warna | warna       |
|  8968 | ADI     | LIN CORE BP-BLACK                                           | CORE        |           2 |            4 | bukan_warna | warna       |
| 51267 | PUM     | PLATFORM TRACE STRAP WN S WHISPER WHITE-                    | TRACE       |           2 |            7 | bukan_warna | warna       |
| 36008 | NIK     | NIKE SIGNAL D/MS/X-GUAVA ICE/LIGHT AQUA-HYPER CRIMSON       | SIGNAL      |           2 |           11 | bukan_warna | warna       |
| 19560 | BBC     | WOODLAND CAMO CURVE T-SHIRT-GREY                            | WOODLAND    |           1 |            6 | bukan_warna | warna       |
| 56083 | WAR     | ROPE GLOW IN THE DARK-WHITE                                 | GLOW        |           2 |            6 | bukan_warna | warna       |
| 18933 | BBC     | FULL SCALE CRASH L/S T-SHIRT-BLACK                          | FULL        |           1 |            8 | bukan_warna | warna       |
| 53459 | PUM     | GLOW PACK CREW PUMA WHITE                                   | GLOW        |           1 |            5 | bukan_warna | warna       |
| 17198 | AGL     | ESAGLXY YELLOW CRICKET LIGHTER -YELLOW                      | YELLOW      |           2 |            5 | bukan_warna | warna       |
| 30639 | NIK     | NIKE AIR ZOOM PEGASUS 35-BLUE ORBIT/BRIGHT CITRON-BLUE VOID | 35          |           5 |           11 | bukan_warna | warna       |
| 14727 | ADI     | SHOPPER-LEGEND INK                                          | LEGEND      |           2 |            3 | warna       | bukan_warna |
|  1407 | ADI     | NMD_TS1 PK-NIGHT CARGO                                      | CARGO       |           4 |            4 | warna       | bukan_warna |
|  7274 | ADI     | NMD_R1-SESAME/TRACAR/BASGRN                                 | SESAME      |           2 |            4 | warna       | bukan_warna |
|  3490 | ADI     | FUTUREPACER-SHOCK RED                                       | SHOCK       |           2 |            3 | warna       | bukan_warna |
| 21685 | HER     | FOURTEEN-NIGHT CAMO                                         | NIGHT       |           2 |            3 | warna       | bukan_warna |
| 55259 | STN     | FAMILY FORCE-AQUA                                           | AQUA        |           3 |            3 | warna       | bukan_warna |
| 33814 | NIK     | NIKE EXPZ07WHITE/BLACK                                      | EXPZ07WHITE |           2 |            3 | warna       | bukan_warna |
| 21386 | HER     | SEVENTEEN-BRBDSCHRY/BKCRSHTCH                               | BRBDSCHRY   |           2 |            3 | warna       | bukan_warna |
| 16112 | ADI     | FLUIDSTREET-VAPOUR PINK                                     | VAPOUR      |           2 |            3 | warna       | bukan_warna |
| 11545 | ADI     | DURAMO 9-ACTIVE RED                                         | ACTIVE      |           3 |            4 | warna       | bukan_warna |
|  4659 | ADI     | CAMPUS-BOAQUA/FTWWHT/CWHITE                                 | BOAQUA      |           2 |            4 | warna       | bukan_warna |
| 21982 | HER     | HANSON-FLORAL BLR                                           | FLORAL      |           2 |            3 | warna       | bukan_warna |
|  4222 | ADI     | TUBULAR DOOM SOCK PK-SESAME/SESAME/CRYWHT                   | SESAME      |           5 |            7 | warna       | bukan_warna |
| 29098 | NIK     | NIKE DOWNSHIFTER 8ASHEN SLATE/OBSIDIANDIFFUSED BLUEBLACK    | 8ASHEN      |           3 |            6 | warna       | bukan_warna |
| 55759 | STN     | VIARTA-RASTA                                                | RASTA       |           2 |            2 | warna       | bukan_warna |
|   656 | ADI     | NMD R1 STLT PK-CBLACK/NOBGRN/BGREEN                         | BGREEN      |           7 |            7 | warna       | bukan_warna |
|  6532 | ADI     | TUBULAR DOOM SOCK PK-BASGRN/SESAME/CWHITE                   | SESAME      |           6 |            7 | warna       | bukan_warna |
| 25371 | NIK     | JORDAN AIR JUMPMAN-BLACK/INFRARED 23                        | 23          |           6 |            6 | warna       | bukan_warna |
| 10328 | ADI     | RUN60S-ACTIVE MAROON                                        | ACTIVE      |           2 |            3 | warna       | bukan_warna |
| 22780 | KIP     | FS72-SHADOW BROWN-140                                       | SHADOW      |           2 |            4 | warna       | bukan_warna |
| 56661 | WAR     | 125CM THE BLUES FLAT LACES                                  | THE         |           2 |            5 | warna       | bukan_warna |
| 55981 | STN     | XYZ-OATMEAL                                                 | OATMEAL     |           2 |            2 | warna       | bukan_warna |
| 55804 | STN     | RAILWAY-VOLT                                                | VOLT        |           2 |            2 | warna       | bukan_warna |
| 26752 | NIK     | WMNS KAWA SLIDEPINK PRIME/ORANGE PEELORANGE PEEL            | PEELORANGE  |           6 |            7 | warna       | bukan_warna |
| 17275 | AGL     | ITALIC 5 PANEL MAROON 005-MAROON                            | 5           |           5 |            6 | warna       | bukan_warna |
|  1405 | ADI     | NMD_TS1 PK-NIGHT CARGO                                      | PK          |           2 |            4 | warna       | bukan_warna |
| 56746 | WAR     | 125CM PAISLEY WHITE FLAT                                    | PAISLEY     |           2 |            4 | warna       | bukan_warna |
| 33831 | NIK     | NIKE EXPX14WHITE/WOLF GREYBLACK                             | EXPX14WHITE |           2 |            4 | warna       | bukan_warna |
|   808 | ADI     | POD-S3.1 C-CBLACK/CBLACK/LEGIVY                             | LEGIVY      |           6 |            6 | warna       | bukan_warna |
|  1039 | ADI     | ARKYN PK W-CBLACK/CBLACK/TESIME                             | TESIME      |           6 |            6 | warna       | bukan_warna |
|  5964 | ADI     | FUTUREPACER-CLOUD WHITE                                     | CLOUD       |           2 |            3 | warna       | bukan_warna |
| 31091 | NIK     | NIKE VIALEBLACK/VOLTSOLAR REDANTHRACITE                     | VIALEBLACK  |           2 |            4 | warna       | bukan_warna |
| 13918 | ADI     | NMD_R1.V2-CARDBOARD                                         | CARDBOARD   |           2 |            2 | warna       | bukan_warna |
| 46960 | NIK     | NK FTR10PURE PLATINUM/BRIGHT CRIMSON/DARK GREY              | FTR10PURE   |           2 |            7 | warna       | bukan_warna |
| 56444 | WAR     | 90CM OREO ROPE                                              | OREO        |           2 |            3 | warna       | bukan_warna |
| 23355 | NIC     | NIKE KD FULL COURT 8P-AMBER/BLACK/METALLIC SILVER/BLACK 07  | 7           |          11 |           11 | warna       | bukan_warna |
| 12023 | ADI     | ASWEERUN-LEGEND INK                                         | LEGEND      |           2 |            3 | warna       | bukan_warna |
|  8759 | ADI     | X LESTO-ACTIVE RED/BLACK/OFF WHITE                          | ACTIVE      |           3 |            7 | warna       | bukan_warna |
| 54953 | SAU     | COURAGEOUS-BRN/YEL                                          | BRN         |           2 |            3 | warna       | bukan_warna |
|  7372 | ADI     | PROPHERE-SGREEN/CGREEN/CBLACK                               | SGREEN      |           2 |            4 | warna       | bukan_warna |
| 32998 | NIK     | PSG M NK BRT STAD JSY SS AW-INFRARED 23/BLACK               | 23          |          10 |           11 | warna       | bukan_warna |
| 50395 | PUM     | RESOLVE PUMA BLACK-PUMA SILVER                              | PUMA        |           2 |            5 | warna       | bukan_warna |
|    12 | ADI     | 3 STRIPE D 29.5-BASKETBALL NATURAL                          | BASKETBALL  |           5 |            6 | warna       | bukan_warna |
| 54951 | SAU     | COURAGEOUS-TAN/PNK                                          | TAN         |           2 |            3 | warna       | bukan_warna |
| 15466 | ADI     | OZELIA-SAVANNAH                                             | SAVANNAH    |           2 |            2 | warna       | bukan_warna |
| 10336 | ADI     | RUN60S-MAROON                                               | MAROON      |           2 |            2 | warna       | bukan_warna |
|  2592 | ADI     | GAZELLE-ICEPUR/WHITE/GOLDMT                                 | ICEPUR      |           2 |            4 | warna       | bukan_warna |
| 13740 | ADI     | ULTRA4D-MAROON                                              | MAROON      |           2 |            2 | warna       | bukan_warna |
|  1403 | ADI     | NMD_R1-GREY TWO F17                                         | F17         |           4 |            4 | warna       | bukan_warna |
|  2197 | ADI     | EQT SUPPORT RF PK-FROGRN/CBLACK/EASGRN                      | EASGRN      |           7 |            7 | warna       | bukan_warna |
| 15761 | ADI     | FLUIDFLOW 2.0-ALUMINA                                       | ALUMINA     |           3 |            3 | warna       | bukan_warna |
| 46940 | NIK     | NK REACTBRIGHT CRIMSON/DARK GREY/PURE PLATINUM              | REACTBRIGHT |           2 |            7 | warna       | bukan_warna |
| 56484 | WAR     | 125CM NEON REFLECTIVE ROPE LACES                            | NEON        |           2 |            5 | warna       | bukan_warna |
| 10573 | ADI     | NMD_R1-METAL GREY                                           | METAL       |           2 |            3 | warna       | bukan_warna |
| 31572 | NIK     | WMNS NIKE QUEST-LIGHTCARBON/BLACKLASER ORANGE               | LIGHTCARBON |           4 |            6 | warna       | bukan_warna |
    

## Model 1: Conv1D dengan Embedding

`Conv1D` atau konvolusi 1 dimensi merupakan satu jenis layer dari layer convolution yang umumnya digunakan untuk mengekstrak fitur penting dari input data.

Meskipun umumnya jaringan saraf tiruan *convolution* digunakan untuk klasifikasi gambar (`Conv2D`) pada pembelajaran *image recognition*, tidak jarang juga `Conv1D` dipergunakan dalam *natural language processing* atau *time series forecasting*.

Layer ini pada prinsipnya menggunakan *kernel_size*, *padding* dan juga *stride* untuk menciptakan sebuah jendela yang akan men-*scan* input matrix atau vektor secara perlahan dan melakukan *pooling* (*min*, *max* atau *average pooling*) untuk mengekstrak nilai yang menjadi fitur penting dari input data.

Oleh karena itu layer *convolutional* ini sering dipergunakan dalam data yang sifatnya *sequence-to-sequence* atau *seq2seq* seperti dalam kasus *natural language processing*, *image classification/recognition*, *audio/video recognition* dan *time series forecasting*.

![convlayer](images/convlayer.gif)

*contoh `Conv2D` pada jaringan saraf tiruan untuk klasifikasi biner/multiclass dari input gambar*

Lebih lanjut mengenai jaringan saraf tiruan *convolution* (convolutional neural network) dapat merujuk pada [CNN Explainer](https://poloclub.github.io/cnn-explainer/)

### Vektorisasi dan Embedding Kata

#### Membuat Lapisan Vektorisasi Kata

Vektorisasi sebenarnya merupakan proses yang cukup sederhana yang merubah kata menjadi representasi numerik berdasarkan total jumlah kata dalam *vocabulary* dari input data.

Di lapisan vektorisasi ini sebenarnya kita melakukan beberapa proses pengolahan terhadap teks yang bersifat opsional, diantaranya:

* Standarisasi kata, merubah semua kata menjadi *lowercase* dan menghilangkan tanda baca (*punctuation*)
* Split setiap input teks menjadi per kata (untuk input yang berupa kalimat)
* Pembentukan *ngrams* pada *corpus*. Apa itu [*ngrams*](https://en.wikipedia.org/wiki/N-gram) dan [*text corpus*](https://en.wikipedia.org/wiki/Text_corpus).
* Indeksasi token (kata)
* Transformasi setiap input menggunakan indeksasi token untuk menghasilkan vektor integer atau vektor angka *float*

Sedangkan *embedding* adalah proses lebih lanjut setelah vektorisasi kata ke dalam representasi numerik. Pada dasarnya embedding adalah sebuah lapisan yang akan memberikan kemampuan untuk menyimpan bobot awal (*initial weight*) dan juga bobot yang nilainya akan di*update* selama proses *training* untuk kata dalam input data.

Sebenarnya tujuan dari proses *embedding* adalah untuk merubah kata per kata dalam sebuah kalimat dalam satu representasi vektor dengan panjang yang sama (dalam kasus *universal sentence embedding* adalah vektor dengan panjang 512) dan merata - ratakan nilai dari kesemua vektor per kata dalam kalimat menjadi satu vektor yang digunakan sebagai acuan klasifikasi, pengelompokan (clustring) atau deteksi.

Meskipun dalam kasus ColorSkim ini yang coba kita lakukan adalah melakukan klasifikasi per kata dan bukan merupakan klasifikasi per kalimat, proses *embedding* masih dapat menjadi satu faktor yang penting dalam melakukan update bobot (*weights*) untuk setiap *neuron* di dalam lapisan model yang dipergunakan melalui proses *backpropagation*.

![embedding](images/embedding.png)

Pada akhir proses training, bobot dari suatu kata sudah melalui beberapa ratus putaran *training* (*epoch*) dari jaringan saraf tiruan dan diharapkan sudah memiliki nilai yang lebih akurat untuk merepresentasikan keadaan (*state*) dari suatu kata terhadap kategori kata atau kalimat yang menjadi target dari proses *training*.

Lebih lengkapnya dapat merujuk pada link berikut:

- [Lapisan Vektorisasi Teks](https://www.tensorflow.org/api_docs/python/tf/keras/layers/TextVectorization)
- [Lapisan Embedding Teks](https://www.tensorflow.org/text/guide/word_embeddings)


```python
# jumlah data (kata) dalam train_data
print(f'Jumlah data: {len(train_data.kata)}')
print('3 kata pertama dalam train_data:')
print(train_data.kata[:3].to_markdown())
```

    Jumlah data: 45400
    3 kata pertama dalam train_data:
|       | kata   |
|------:|:-------|
| 43886 | GREY   |
| 14859 | BLACK  |
| 47729 | U      |
    


```python
# jumlah data unik (kata unik) dalam train_data[:, 0]
jumlah_kata_train = len(np.unique(train_data.kata))
jumlah_kata_train
```




    2957




```python
# Membuat lapisan vektorisasi kata
lapisan_vektorisasi = TextVectorization(max_tokens=jumlah_kata_train,
                                        output_sequence_length=1,
                                        standardize='lower_and_strip_punctuation',
                                        name='lapisan_vektorisasi')
```


```python
# Mengadaptasikan lapisan vektorisasi ke dalam train_kata
lapisan_vektorisasi.adapt(train_data.kata.tolist())
```


```python
# Uji vektorisasi kata
target_kata = random.choice(train_data.kata.tolist())
print(f'Kata:\n{target_kata}\n')
print(f'Kata setelah vektorisasi:\n{lapisan_vektorisasi([target_kata])}')
```

    Kata:
    ULTRABOOST
    
    Kata setelah vektorisasi:
    [[22]]
    


```python
# Konfigurasi lapisan vektorisasi
lapisan_vektorisasi.get_config()
```




    {'name': 'lapisan_vektorisasi',
     'trainable': True,
     'batch_input_shape': (None,),
     'dtype': 'string',
     'max_tokens': 2957,
     'standardize': 'lower_and_strip_punctuation',
     'split': 'whitespace',
     'ngrams': None,
     'output_mode': 'int',
     'output_sequence_length': 1,
     'pad_to_max_tokens': False,
     'sparse': False,
     'ragged': False,
     'vocabulary': None,
     'idf_weights': None}




```python
# Jumlah vocabulary dalam lapisan_vektorisasi
jumlah_vocab = lapisan_vektorisasi.get_vocabulary()
len(jumlah_vocab)
```




    2906



#### Membuat Lapisan Text Embedding


```python
# Membuat lapisan embedding kata
lapisan_embedding = Embedding(input_dim=len(jumlah_vocab),
                              output_dim=UKURAN_BATCH,
                              mask_zero=True,
                              name='lapisan_embedding')
```


```python
# Contoh vektorisasi dan embedding
print(f'Kata sebelum vektorisasi:\n{target_kata}\n')
kata_tervektor = lapisan_vektorisasi([target_kata])
print(f'\nKata sesudah vektorisasi (sebelum embedding):\n{kata_tervektor}\n')
kata_terembed = lapisan_embedding(kata_tervektor)
print(f'\nKata setelah embedding:\n{kata_terembed}\n')
print(f'Shape dari kata setelah embedding:\n{kata_terembed.shape}')
```

    Kata sebelum vektorisasi:
    ULTRABOOST
    
    
    Kata sesudah vektorisasi (sebelum embedding):
    [[22]]
    
    
    Kata setelah embedding:
    [[[ 0.04344383 -0.01464387 -0.03505018  0.03013081 -0.03754324
        0.0161945  -0.00386264 -0.02871505 -0.02963296 -0.00619398
       -0.03534311  0.03194788 -0.04005265 -0.00023266  0.01971561
       -0.02440481  0.0317731   0.00433347  0.00037297  0.03806284
        0.01001209 -0.01998474  0.0183844  -0.02738135  0.04759083
       -0.02447642 -0.03849269  0.00627618 -0.00946458 -0.02909201
       -0.03733308  0.00344805]]]
    
    Shape dari kata setelah embedding:
    (1, 1, 32)
    

### Membuat TensorFlow Dataset, Batching dan Prefetching

Pada bagian ini kita akan merubah data menjadi *dataset* dan menerapkan *batching* serta *prefetching* pada dataset untuk mempercepat performa *training* model.

![prefetched](images/prefetched.jpg)

Lebih lengkap mengenai peningkatan performa training bisa dilihat di [Better Performance with the tf.data API](https://www.tensorflow.org/guide/data_performance)


```python
# Membuat TensorFlow dataset
train_kata_dataset = from_tensor_slices((train_data.iloc[:, 0], train_target))
test_kata_dataset = from_tensor_slices((test_data.iloc[:, 0], test_target))

train_kata_dataset, test_kata_dataset
```




    (<TensorSliceDataset element_spec=(TensorSpec(shape=(), dtype=tf.string, name=None), TensorSpec(shape=(), dtype=tf.int32, name=None))>,
     <TensorSliceDataset element_spec=(TensorSpec(shape=(), dtype=tf.string, name=None), TensorSpec(shape=(), dtype=tf.int32, name=None))>)




```python
# Membuat TensorSliceDataset menjadi prefetched dataset
train_kata_dataset = train_kata_dataset.batch(UKURAN_BATCH).prefetch(tf.data.AUTOTUNE)
test_kata_dataset = test_kata_dataset.batch(UKURAN_BATCH).prefetch(tf.data.AUTOTUNE)

train_kata_dataset, test_kata_dataset
```




    (<BatchDataset element_spec=(TensorSpec(shape=(None,), dtype=tf.string, name=None), TensorSpec(shape=(None,), dtype=tf.int32, name=None))>,
     <BatchDataset element_spec=(TensorSpec(shape=(None,), dtype=tf.string, name=None), TensorSpec(shape=(None,), dtype=tf.int32, name=None))>)



### Membangun dan Menjalankan Training Model 1


```python
# Jika folder dengan path 'colorskim_checkpoint/{model.name}' sudah ada, maka skip fit model 
# untuk menghemat waktu pengembangan dan hanya load model yang sudah ada dalam folder tersebut saja
if not os.path.isdir(f'colorskim_checkpoint/{MODEL[1]}'):
        # set random.set_seed untuk konsistensi keacakan
        tf.random.set_seed(RANDOM_STATE)

        # * Membuat model_1 dengan layer Conv1D dan lapisan vektorisasi serta embedding input kata
        inputs = Input(shape=(1,), 
                       dtype=tf.string, 
                       name='lapisan_input')
        lapisan_vektor = lapisan_vektorisasi(inputs)
        lapisan_embed = lapisan_embedding(lapisan_vektor)
        x = Conv1D(filters=UKURAN_BATCH, 
                   kernel_size=5, 
                   padding='same', 
                   activation='relu',
                   name='lapisan_konvolusional_1_dimensi')(lapisan_embed)
        x = GlobalMaxPooling1D(name='lapisan_max_pool')(x)
        outputs = Dense(units=1, 
                        activation='sigmoid', 
                        name='lapisan_output')(x)
        model_1 = Model(inputs=inputs, 
                        outputs=outputs, 
                        name=MODEL[1])

        # Compile
        model_1.compile(loss=BinaryCrossentropy(),
                        optimizer=Adam(),
                        metrics=['accuracy'])

        # Setup wandb init dan config
        wb.init(project=wandb['proyek'],
                entity=wandb['user'],
                name=model_1.name,
                config={'epochs': EPOCHS,
                        'n_layers': len(model_1.layers)})

        
        # Fit model_1
        model_1.fit(train_kata_dataset,
                    epochs=wb.config.epochs,
                    validation_data=test_kata_dataset,
                    callbacks=[wandb_callback(train_kata_dataset),
                               model_checkpoint(model_1.name),
                               early_stopping(),
                               reduce_lr_on_plateau()])
        
        # tutup logging wandb
        wb.finish()
        
        # load model_1
        model_1 = load_model(f'colorskim_checkpoint/{MODEL[1]}')
else:
        # load model_1
        model_1 = load_model(f'colorskim_checkpoint/{MODEL[1]}')
```


```python
# Ringkasan dari model_1
model_1.summary()
```

    Model: "model_1_Conv1D_vektorisasi_embedding"
    _________________________________________________________________
     Layer (type)                Output Shape              Param #   
    =================================================================
     lapisan_input (InputLayer)  [(None, 1)]               0         
                                                                     
     lapisan_vektorisasi (TextVe  (None, 1)                0         
     ctorization)                                                    
                                                                     
     lapisan_embedding (Embeddin  (None, 1, 32)            92992     
     g)                                                              
                                                                     
     lapisan_konvolusional_1_dim  (None, 1, 32)            5152      
     ensi (Conv1D)                                                   
                                                                     
     lapisan_max_pool (GlobalMax  (None, 32)               0         
     Pooling1D)                                                      
                                                                     
     lapisan_output (Dense)      (None, 1)                 33        
                                                                     
    =================================================================
    Total params: 98,177
    Trainable params: 98,177
    Non-trainable params: 0
    _________________________________________________________________
    


```python
# Plot model_1
plot_model(model_1, show_shapes=True)
```




    
![png](ColorSkim_AI_files/ColorSkim_AI_51_0.png)
    



### Eksplorasi Hasil Model 1

Setelah proses *training* pada model_1 yang terhenti di epoch 14 setelah melalui beberapa kali reduksi *learning_rate* namun dengan *val_accuracy* yang tidak meningkat setelah melalui sejumlah toleransi epoch dari `EarlyStopping` *callbacks*, kita mendapatkan *val_accuracy* terakhir di 99.21%.
Di bagian bawah kita akan melakukan beberapa evaluasi dari hasil *training* model_1:

1. Evaluasi *val_loss* dan *val_accuracy* model_1
2. Memuat model dengan *val_accuracy* terbaik selama *training* model_1 dan lakukan evaluasi
3. Membuat contoh prediksi dengan model terbaik selama *training* model_1
4. Hitung metrik dari model terbaik selama *training* model_1
5. *Plot confusion matrix* dari model terbaik selama *training* model_1
6. Tampilkan *False Negative* dan *False Positive* dari model terbaik selama *training* model_1 dalam dataframe


```python
# Evaluasi model_1
model_1.evaluate(test_kata_dataset)
```

    355/355 [==============================] - 20s 38ms/step - loss: 0.0289 - accuracy: 0.9921
    [0.02888941951096058, 0.9920712113380432]




```python
# Membuat probabilitas prodeksi menggunakan model_1
model_1_pred_prob = tf.squeeze(model_1.predict(test_kata_dataset))
model_1_pred_prob
```




    <tf.Tensor: shape=(11351,), dtype=float32, numpy=
    array([9.8689961e-01, 1.5283754e-03, 9.9962139e-01, ..., 1.1869609e-05,
           7.7320910e-06, 4.8428519e-05], dtype=float32)>




```python
# Membuat prediksi untuk model_1
model_1_pred = tf.round(model_1_pred_prob)
model_1_pred
```




    <tf.Tensor: shape=(11351,), dtype=float32, numpy=array([1., 0., 1., ..., 0., 0., 0.], dtype=float32)>




```python
# Menghitung metriks dari model_1
model_1_metrik = hitung_metrik(target=test_target,
                               prediksi=model_1_pred)
model_1_metrik
```




    {'akurasi': 0.9920711831556691,
     'presisi': 0.9920716853479361,
     'recall': 0.9920711831556691,
     'f1-score': 0.9920682214744327}




```python

# ⁡⁣⁢Membuat fungsi untuk plot residual dari model regresi logistik⁡
def residual_plot_logr(test_target, 
                       nama_model,
                       model_akurasi, 
                       probabilitas_prediksi_model, 
                       jumlah_bin=100, 
                       rentang=[0, 1]):
    """
    Fungsi ini akan menciptakan residual plot untuk logistik regresi dari permodelan
    
    Args:
        test_target (np.ndarray): target dari test data dalama bentuk 𝟭D numpy array
        nama_model (str): nama model dalam string untuk ditampilkan di judul plot
        model_akurasi (float): akurasi model
        probabilitas_prediksi_model (np.ndarray): probabilitas prediksi model dalam bentuk 𝟭D numpy array
        jumlah_bin (int): jumlah bin yang akan digunakan untuk plot sepanjang axis x
        rentang (list): rentang yang akan digunakan di axis x
        
    Returns:
        residual_plot (matplotlib.pyplot.scatter): plot residual dari model regresi logistik⁡⁡
    """
    
    # fungsi internal untuk menjumlahkan residu dalam kelompok bin tertentu
    def func(residu):
        y = np.sum(residu)
        return y

    axis_x = [langkah_x/jumlah_bin for langkah_x in range(jumlah_bin+1)]

    residual = test_target - probabilitas_prediksi_model
    bin_residual = binned_statistic(residual, residual, statistic=func, bins=jumlah_bin+1, range=rentang)[0]
    plt.scatter(axis_x, bin_residual, c='r')
    plt.title(f'Residual Regresi Logistik\n{nama_model}\nAkurasi: {model_akurasi:.2%}',
              fontsize=14)
    plt.xlabel('Target Label dalam Bin')
    plt.ylabel('Residual')
    plt.figure(facecolor='w')
```


```python
# Plot residual dari model 1
residual_plot_logr(test_target, MODEL[1], model_1_metrik['akurasi'], model_1_pred_prob)
plt.show()
```


    
![png](ColorSkim_AI_files/ColorSkim_AI_58_0.png)
    



```python
# Menampilkan confusion matrix dari model_1
plot_conf_matrix(target_label=test_target,
                 prediksi_label=model_1_pred,
                 nama_model=MODEL[1],
                 akurasi=model_1_metrik['akurasi'],
                 label_titik_x=['bukan_warna', 'warna'],
                 label_titik_y=['bukan_warna', 'warna'])
plt.figure(facecolor='w')
plt.show()
```


    
![png](ColorSkim_AI_files/ColorSkim_AI_59_0.png)
    



```python
# Menampilkan kesalahan prediksi model_1
df_kesalahan_prediksi(label_encoder=label_encoder,
                      test_data=test_data_mnb,
                      prediksi=model_1_pred,
                      probabilitas_prediksi=model_1_pred_prob,
                      order_ulang_header=['brand', 
                                          'kata',
                                          'nama_artikel', 
                                          'urut_kata', 
                                          'total_kata', 
                                          'label'])
```

|       | brand   | kata        | nama_artikel                                                |   urut_kata |   total_kata | label       | prediksi    | probabilitas   |
|------:|:--------|:------------|:------------------------------------------------------------|------------:|-------------:|:------------|:------------|:---------------|
| 17520 | AGL     | BROWN       | BROWN MOUNTAIN 008 - PEACH                                  |           1 |            4 | bukan_warna | warna       | 99.45%         |
| 21174 | HER     | RED         | HER-HERITAGE-BOSTON RED SOX-(21L)-BAG-US                    |           4 |            8 | bukan_warna | warna       | 99.34%         |
| 56112 | WAR     | RED         | RED BLACK FLAT NON REFLECTIVE-RED/BLACK                     |           1 |            7 | bukan_warna | warna       | 99.34%         |
| 48075 | PTG     | ORANGE      | POLKA ORANGE-ORANGE                                         |           2 |            3 | bukan_warna | warna       | 99.27%         |
| 56226 | WAR     | ORANGE      | SHOELACES ORANGE OVAL LACES-ORANGE                          |           2 |            5 | bukan_warna | warna       | 99.27%         |
| 21091 | HER     | 600D        | POP QUIZ-600D POLY NAVY/ZIP                                 |           3 |            6 | bukan_warna | warna       | 99.05%         |
| 16288 | ADI     | BLK         | CLR BLK CRW 2PP-BLACK                                       |           2 |            5 | bukan_warna | warna       | 98.82%         |
| 17198 | AGL     | YELLOW      | ESAGLXY YELLOW CRICKET LIGHTER -YELLOW                      |           2 |            5 | bukan_warna | warna       | 97.57%         |
| 48153 | PTG     | DOVE        | MISTY DOVE-GREY                                             |           2 |            3 | bukan_warna | warna       | 95.45%         |
| 51267 | PUM     | TRACE       | PLATFORM TRACE STRAP WN S WHISPER WHITE-                    |           2 |            7 | bukan_warna | warna       | 91.96%         |
| 18933 | BBC     | FULL        | FULL SCALE CRASH L/S T-SHIRT-BLACK                          |           1 |            8 | bukan_warna | warna       | 90.96%         |
| 18940 | BBC     | FULL        | FULL SCALE CRASH L/S T-SHIRT-ORANGE                         |           1 |            8 | bukan_warna | warna       | 90.96%         |
| 56116 | WAR     | FULL        | FULL BLACK FLAT NON REFLECTIVE-BLACK                        |           1 |            6 | bukan_warna | warna       | 90.96%         |
|  8735 | ADI     | FULL        | FULL ZIP-CWHITE                                             |           1 |            3 | bukan_warna | warna       | 90.96%         |
| 52109 | PUM     | GLOW        | CELL STELLAR GLOW WNS PUMA WHITE-PURPLE                     |           3 |            7 | bukan_warna | warna       | 89.1%          |
| 53459 | PUM     | GLOW        | GLOW PACK CREW PUMA WHITE                                   |           1 |            5 | bukan_warna | warna       | 89.1%          |
| 56083 | WAR     | GLOW        | ROPE GLOW IN THE DARK-WHITE                                 |           2 |            6 | bukan_warna | warna       | 89.1%          |
| 56086 | WAR     | GLOW        | FLAT GLOW IN THE DARK-WHITE                                 |           2 |            6 | bukan_warna | warna       | 89.1%          |
| 52114 | PUM     | GLOW        | CELL STELLAR GLOW WN"S PUMA BLACK-PURPLE HEATHER            |           3 |            8 | bukan_warna | warna       | 89.1%          |
| 48650 | PUM     | CORE        | WMN CORE SEASONAL ARCHIVE BACKPACK PEACO                    |           2 |            6 | bukan_warna | warna       | 80.28%         |
| 52841 | PUM     | CORE        | CORE-RUN S/S TEE-LAPIS BLUE                                 |           1 |            7 | bukan_warna | warna       | 80.28%         |
|  8968 | ADI     | CORE        | LIN CORE BP-BLACK                                           |           2 |            4 | bukan_warna | warna       | 80.28%         |
| 24154 | NIK     | CORE        | TS CORE POLO-ROYAL BLUE/WHITE                               |           2 |            6 | bukan_warna | warna       | 80.28%         |
|  8965 | ADI     | CORE        | LIN CORE CROSSB-BLACK                                       |           2 |            4 | bukan_warna | warna       | 80.28%         |
|  8962 | ADI     | CORE        | LIN CORE ORG-BLACK                                          |           2 |            4 | bukan_warna | warna       | 80.28%         |
| 48805 | PUM     | CORE        | WMN CORE ROUND BACKPACK LILAC SNOW-VALENTINE                |           2 |            7 | bukan_warna | warna       | 80.28%         |
| 19643 | BEA     | 35          | SERIES 35-MULTI                                             |           2 |            3 | bukan_warna | warna       | 79.45%         |
| 30639 | NIK     | 35          | NIKE AIR ZOOM PEGASUS 35-BLUE ORBIT/BRIGHT CITRON-BLUE VOID |           5 |           11 | bukan_warna | warna       | 79.45%         |
| 30654 | NIK     | 35          | NIKE AIR ZOOM PEGASUS 35-BRIGHT CRIMSON/ICE BLUE-SAIL       |           5 |           10 | bukan_warna | warna       | 79.45%         |
| 54972 | SAU     | VINTAGE     | JAZZ VINTAGE-GREY/BLUE/WHITE                                |           2 |            5 | bukan_warna | warna       | 78.0%          |
| 19560 | BBC     | WOODLAND    | WOODLAND CAMO CURVE T-SHIRT-GREY                            |           1 |            6 | bukan_warna | warna       | 74.04%         |
| 19265 | BBC     | DARK        | BB DARK STAR LS KNIT-BLACK                                  |           2 |            6 | bukan_warna | warna       | 67.7%          |
| 19622 | BBC     | CREAM       | ICE CREAM MAN TEE-WHITE                                     |           2 |            5 | bukan_warna | warna       | 67.23%         |
| 18208 | BBC     | CLEAR       | BB CLEAR SKY L/S T-SHIRT-BLACK                              |           2 |            8 | bukan_warna | warna       | 67.08%         |
| 36008 | NIK     | SIGNAL      | NIKE SIGNAL D/MS/X-GUAVA ICE/LIGHT AQUA-HYPER CRIMSON       |           2 |           11 | bukan_warna | warna       | 60.5%          |
|  8759 | ADI     | ACTIVE      | X LESTO-ACTIVE RED/BLACK/OFF WHITE                          |           3 |            7 | warna       | bukan_warna | 43.27%         |
| 10328 | ADI     | ACTIVE      | RUN60S-ACTIVE MAROON                                        |           2 |            3 | warna       | bukan_warna | 43.27%         |
| 11545 | ADI     | ACTIVE      | DURAMO 9-ACTIVE RED                                         |           3 |            4 | warna       | bukan_warna | 43.27%         |
| 21685 | HER     | NIGHT       | FOURTEEN-NIGHT CAMO                                         |           2 |            3 | warna       | bukan_warna | 40.44%         |
|  7274 | ADI     | SESAME      | NMD_R1-SESAME/TRACAR/BASGRN                                 |           2 |            4 | warna       | bukan_warna | 38.07%         |
|  6532 | ADI     | SESAME      | TUBULAR DOOM SOCK PK-BASGRN/SESAME/CWHITE                   |           6 |            7 | warna       | bukan_warna | 38.07%         |
|  4222 | ADI     | SESAME      | TUBULAR DOOM SOCK PK-SESAME/SESAME/CRYWHT                   |           5 |            7 | warna       | bukan_warna | 38.07%         |
| 32998 | NIK     | 23          | PSG M NK BRT STAD JSY SS AW-INFRARED 23/BLACK               |          10 |           11 | warna       | bukan_warna | 34.74%         |
| 25371 | NIK     | 23          | JORDAN AIR JUMPMAN-BLACK/INFRARED 23                        |           6 |            6 | warna       | bukan_warna | 34.74%         |
|  5964 | ADI     | CLOUD       | FUTUREPACER-CLOUD WHITE                                     |           2 |            3 | warna       | bukan_warna | 25.57%         |
| 56661 | WAR     | THE         | 125CM THE BLUES FLAT LACES                                  |           2 |            5 | warna       | bukan_warna | 15.38%         |
| 21386 | HER     | BRBDSCHRY   | SEVENTEEN-BRBDSCHRY/BKCRSHTCH                               |           2 |            3 | warna       | bukan_warna | 7.35%          |
| 54953 | SAU     | BRN         | COURAGEOUS-BRN/YEL                                          |           2 |            3 | warna       | bukan_warna | 7.35%          |
| 16112 | ADI     | VAPOUR      | FLUIDSTREET-VAPOUR PINK                                     |           2 |            3 | warna       | bukan_warna | 7.35%          |
|  4659 | ADI     | BOAQUA      | CAMPUS-BOAQUA/FTWWHT/CWHITE                                 |           2 |            4 | warna       | bukan_warna | 7.35%          |
| 33814 | NIK     | EXPZ07WHITE | NIKE EXPZ07WHITE/BLACK                                      |           2 |            3 | warna       | bukan_warna | 7.35%          |
| 31572 | NIK     | LIGHTCARBON | WMNS NIKE QUEST-LIGHTCARBON/BLACKLASER ORANGE               |           4 |            6 | warna       | bukan_warna | 7.35%          |
|    12 | ADI     | BASKETBALL  | 3 STRIPE D 29.5-BASKETBALL NATURAL                          |           5 |            6 | warna       | bukan_warna | 7.35%          |
| 55804 | STN     | VOLT        | RAILWAY-VOLT                                                |           2 |            2 | warna       | bukan_warna | 7.35%          |
| 46960 | NIK     | FTR10PURE   | NK FTR10PURE PLATINUM/BRIGHT CRIMSON/DARK GREY              |           2 |            7 | warna       | bukan_warna | 7.35%          |
| 13918 | ADI     | CARDBOARD   | NMD_R1.V2-CARDBOARD                                         |           2 |            2 | warna       | bukan_warna | 7.35%          |
| 31091 | NIK     | VIALEBLACK  | NIKE VIALEBLACK/VOLTSOLAR REDANTHRACITE                     |           2 |            4 | warna       | bukan_warna | 7.35%          |
|   808 | ADI     | LEGIVY      | POD-S3.1 C-CBLACK/CBLACK/LEGIVY                             |           6 |            6 | warna       | bukan_warna | 7.35%          |
| 55981 | STN     | OATMEAL     | XYZ-OATMEAL                                                 |           2 |            2 | warna       | bukan_warna | 7.35%          |
| 33831 | NIK     | EXPX14WHITE | NIKE EXPX14WHITE/WOLF GREYBLACK                             |           2 |            4 | warna       | bukan_warna | 7.35%          |
|   656 | ADI     | BGREEN      | NMD R1 STLT PK-CBLACK/NOBGRN/BGREEN                         |           7 |            7 | warna       | bukan_warna | 7.35%          |
| 26752 | NIK     | PEELORANGE  | WMNS KAWA SLIDEPINK PRIME/ORANGE PEELORANGE PEEL            |           6 |            7 | warna       | bukan_warna | 7.35%          |
|  1039 | ADI     | TESIME      | ARKYN PK W-CBLACK/CBLACK/TESIME                             |           6 |            6 | warna       | bukan_warna | 7.35%          |
| 15466 | ADI     | SAVANNAH    | OZELIA-SAVANNAH                                             |           2 |            2 | warna       | bukan_warna | 7.35%          |
| 13740 | ADI     | MAROON      | ULTRA4D-MAROON                                              |           2 |            2 | warna       | bukan_warna | 7.35%          |
| 55759 | STN     | RASTA       | VIARTA-RASTA                                                |           2 |            2 | warna       | bukan_warna | 7.35%          |
| 46940 | NIK     | REACTBRIGHT | NK REACTBRIGHT CRIMSON/DARK GREY/PURE PLATINUM              |           2 |            7 | warna       | bukan_warna | 7.35%          |
|  2197 | ADI     | EASGRN      | EQT SUPPORT RF PK-FROGRN/CBLACK/EASGRN                      |           7 |            7 | warna       | bukan_warna | 7.35%          |
| 29098 | NIK     | 8ASHEN      | NIKE DOWNSHIFTER 8ASHEN SLATE/OBSIDIANDIFFUSED BLUEBLACK    |           3 |            6 | warna       | bukan_warna | 7.35%          |
|  2592 | ADI     | ICEPUR      | GAZELLE-ICEPUR/WHITE/GOLDMT                                 |           2 |            4 | warna       | bukan_warna | 7.35%          |
|  7372 | ADI     | SGREEN      | PROPHERE-SGREEN/CGREEN/CBLACK                               |           2 |            4 | warna       | bukan_warna | 7.35%          |
| 10336 | ADI     | MAROON      | RUN60S-MAROON                                               |           2 |            2 | warna       | bukan_warna | 7.35%          |
| 15761 | ADI     | ALUMINA     | FLUIDFLOW 2.0-ALUMINA                                       |           3 |            3 | warna       | bukan_warna | 7.35%          |
| 22780 | KIP     | SHADOW      | FS72-SHADOW BROWN-140                                       |           2 |            4 | warna       | bukan_warna | 2.94%          |
| 14727 | ADI     | LEGEND      | SHOPPER-LEGEND INK                                          |           2 |            3 | warna       | bukan_warna | 0.95%          |
| 12023 | ADI     | LEGEND      | ASWEERUN-LEGEND INK                                         |           2 |            3 | warna       | bukan_warna | 0.95%          |
| 17275 | AGL     | 5           | ITALIC 5 PANEL MAROON 005-MAROON                            |           5 |            6 | warna       | bukan_warna | 0.63%          |
| 21982 | HER     | FLORAL      | HANSON-FLORAL BLR                                           |           2 |            3 | warna       | bukan_warna | 0.48%          |
|  1403 | ADI     | F17         | NMD_R1-GREY TWO F17                                         |           4 |            4 | warna       | bukan_warna | 0.47%          |
| 23355 | NIC     | 7           | NIKE KD FULL COURT 8P-AMBER/BLACK/METALLIC SILVER/BLACK 07  |          11 |           11 | warna       | bukan_warna | 0.4%           |
| 56444 | WAR     | OREO        | 90CM OREO ROPE                                              |           2 |            3 | warna       | bukan_warna | 0.33%          |
| 56746 | WAR     | PAISLEY     | 125CM PAISLEY WHITE FLAT                                    |           2 |            4 | warna       | bukan_warna | 0.28%          |
|  1407 | ADI     | CARGO       | NMD_TS1 PK-NIGHT CARGO                                      |           4 |            4 | warna       | bukan_warna | 0.23%          |
| 50395 | PUM     | PUMA        | RESOLVE PUMA BLACK-PUMA SILVER                              |           2 |            5 | warna       | bukan_warna | 0.17%          |
| 56484 | WAR     | NEON        | 125CM NEON REFLECTIVE ROPE LACES                            |           2 |            5 | warna       | bukan_warna | 0.1%           |
|  3490 | ADI     | SHOCK       | FUTUREPACER-SHOCK RED                                       |           2 |            3 | warna       | bukan_warna | 0.09%          |
| 10573 | ADI     | METAL       | NMD_R1-METAL GREY                                           |           2 |            3 | warna       | bukan_warna | 0.06%          |
| 55259 | STN     | AQUA        | FAMILY FORCE-AQUA                                           |           3 |            3 | warna       | bukan_warna | 0.05%          |
| 54951 | SAU     | TAN         | COURAGEOUS-TAN/PNK                                          |           2 |            3 | warna       | bukan_warna | 0.02%          |
|  1405 | ADI     | PK          | NMD_TS1 PK-NIGHT CARGO                                      |           2 |            4 | warna       | bukan_warna | 0.02%          |
    


```python
# selesai dengan model 1, bersihkan memori di GPU terkait model_1
del model_1
gc.collect()
```




    10475



## Model 2: Transfer Learning pretrained feature exraction menggunakan Universal Sentence Encoder (USE)

Pada bagian ini kita akan mencoba untuk melakukan training pada data menggunakan lapisan *feature extraction* yang sudah ada dan sudah dilatih pada dataset tertentu. 
Proses embedding secara umum memiliki beberapa kelemahan diantaranya:

1. Kehilangan informasi, dimana dalam kasus kalimat "Produk ini bagus" dan kalimat "Ini" menggunakan rerata vektor memiliki tingkat kemiripan yg cukup tinggi meskipun keduanya merupakan kalimat yang memiliki makna cukup berbeda
2. Tidak memandang urutan, dimana kalimat "Makan ikan menggunakan sendok" dan kalimat "Makan sendok menggunakan ikan" akan memiliki kemiripan vektor 100%

Kita dapat menghindari permasalahan ini misalkan dengan menerapkan *feature engineering* untuk membuat input menjadi semakin kompleks dan menghindari masalah yang mungkin timbul dari proses *embedding*, namun hal ini dapat melibatkan beberapa proses seperti menghilangkan *stop-words*, pembobotan menggunakan TF-IDF, menambahkan *ngrams* untuk mendapatkan posisi kata dalam kalimat, penumpukan lapisan *MaxPooling* dan lain sebagainya.

Universal Sentence Encoder merupakan suatu lapisan yang sudah melakukan hampir kesemua proses ini dalam proses *embedding* input data.


```python
# Mengunduh pretrained USE dari tensorflow hub atau model USE yang sudah didownload
tf_hub_embedding = hub.KerasLayer('colorskim_checkpoint/use.v4/',
                                  trainable=False,
                                  name='lapisan_embedding_USE')
```


```python
# Melakukan tes pretrained embedding pada contoh kata
kata_acak = random.choice(train_data_mnb['kata'].tolist())
print(f'Kata acak:\n {kata_acak}')
kata_embed_pretrain = tf_hub_embedding([kata_acak])
print(f'\nKata setelah embed dengan USE:\n{kata_embed_pretrain[0]}\n')
print(f'Panjang dari kata setelah embedding: {len(kata_embed_pretrain[0])}')
```

    Kata acak:
     NIKE
    
    Kata setelah embed dengan USE:
    [ 0.03551936 -0.01092234 -0.04452999 -0.011615    0.02078822  0.03616779
      0.02028491 -0.04521689 -0.06267793 -0.01205434 -0.06147045 -0.04277659
      0.044472   -0.03846516 -0.01646156  0.04231635 -0.04280001 -0.04259221
      0.02336692 -0.0149713   0.00785487  0.0131372   0.06377136  0.00091608
     -0.01144155 -0.00248365 -0.0456779  -0.046661   -0.01129454 -0.02044706
      0.07824446 -0.02754597  0.00210647  0.07455748  0.03685687  0.04225997
      0.03182033 -0.02679156  0.06342416  0.04529719 -0.06783359 -0.0248361
      0.07103027 -0.00629707  0.00433935  0.05146051 -0.01485019  0.03876241
      0.07278479  0.0009041   0.05664367 -0.06668704  0.0241777   0.07243448
     -0.04070075 -0.06603528  0.06918954 -0.04095845  0.06053145  0.07463565
      0.01343846 -0.02636237  0.0616648   0.05107952  0.02252498  0.02215132
     -0.06586358  0.00607882  0.05185373  0.05264953 -0.07568023 -0.04224531
     -0.00805937 -0.04342831  0.00167232 -0.06503586  0.004361   -0.01371297
     -0.03355193  0.02891815  0.06603467  0.01700427 -0.0279174   0.02507633
     -0.02567692 -0.04835125 -0.03684086 -0.04631411  0.05025171  0.04808898
     -0.02833271 -0.01959861  0.02793185  0.0543673   0.03376793 -0.05010658
      0.00592942 -0.06519038  0.01859676 -0.0353713   0.00766027 -0.00334424
     -0.03353013  0.0313806  -0.02838316  0.03093965  0.05986924  0.0233705
     -0.06270017  0.05457193 -0.02671948 -0.02759826 -0.01483797 -0.04928165
      0.02404594 -0.02501503  0.05870967  0.0207445  -0.05472719  0.05273714
     -0.07252224 -0.06794119 -0.01538243  0.01332556 -0.05433645  0.0688628
     -0.03460832 -0.01138156 -0.00949107 -0.01125747 -0.06732188 -0.04552472
      0.07460805  0.00889968 -0.00280288  0.06425659  0.0613145   0.00834193
     -0.03950722 -0.06944551 -0.01862261 -0.05291961  0.05189019 -0.0704136
     -0.01533997  0.03102191 -0.06291    -0.02337333  0.06298447  0.05690104
      0.04923342  0.03985649 -0.06449309 -0.00915918 -0.02750308 -0.02121983
     -0.03716319 -0.05120215 -0.02033108 -0.0376828   0.05732057  0.0508699
     -0.05313801 -0.02852584 -0.07193327  0.04402891 -0.043501    0.01040053
     -0.05722377 -0.02140645  0.0280024   0.06227337 -0.00149252 -0.04377782
      0.04100104  0.05172922  0.01621546  0.00419973 -0.03920238  0.00992098
     -0.05693381 -0.05796782  0.04262339 -0.05103716  0.01172633 -0.01269967
      0.07080697 -0.06015277 -0.02805375  0.03648995 -0.0404921  -0.0074265
      0.07094666  0.04367533  0.06983639 -0.07221088  0.00054545 -0.07213555
     -0.07322811 -0.05495409 -0.05772717 -0.07736363 -0.04488755  0.00677462
      0.01069611  0.00771156  0.02260842  0.00909366  0.07486101  0.06389294
     -0.06780038 -0.07150001  0.00668007 -0.03589934  0.03037837  0.02435404
     -0.07544938 -0.04152847  0.06835841  0.04962672 -0.02535345 -0.02914595
     -0.03527066  0.0560013   0.01352146  0.01594795 -0.06152666  0.06713767
      0.01546754 -0.04652558  0.0087596  -0.05915464  0.04778609  0.07824762
     -0.01258852 -0.02061768  0.05956362  0.04689779 -0.06026362 -0.02124402
      0.00915563  0.068636   -0.04575855  0.01184536 -0.01048048 -0.0718644
     -0.05675729  0.01992276 -0.05778389 -0.07409275  0.04049169 -0.02760815
      0.0036087  -0.02334823 -0.0483748  -0.02317634 -0.05466368 -0.00628347
     -0.06936521 -0.06271482  0.00500453 -0.04129516  0.01160167 -0.04566887
      0.02773606 -0.04745081 -0.06555958  0.00411041 -0.02365758  0.04201781
     -0.06258721 -0.00577203 -0.01704673  0.01889912  0.04837367 -0.00619665
      0.02614878  0.01605906  0.07285248  0.0379419  -0.0283509   0.02107766
     -0.01478508  0.05798338  0.05910319  0.03409204 -0.05160711 -0.04472215
     -0.07365095  0.04046992  0.01024232 -0.04927742  0.07710335 -0.07088924
      0.01531921 -0.05522642  0.05740457 -0.01346224 -0.01329934 -0.07674375
      0.02388921 -0.040433   -0.05591584 -0.03715642  0.02376054  0.02959199
     -0.04881466 -0.01823488 -0.01824399  0.04357    -0.07166452  0.00840077
      0.00098459 -0.00247648 -0.04364586 -0.04615242 -0.03177207  0.05448828
      0.01465963 -0.03656979 -0.06917886 -0.05253936  0.02594306  0.07820571
      0.06673191 -0.00256606  0.04285944  0.06004287 -0.02713689 -0.03663595
     -0.0310873   0.06888737  0.0343462   0.0313268  -0.049203    0.06942404
     -0.07673703  0.0602483   0.01684608 -0.06147083 -0.01759119 -0.06714571
      0.05348786 -0.05480853  0.05088906  0.05817876  0.01446326  0.04168681
      0.0077525  -0.04761557 -0.04279744 -0.05798563  0.06405424 -0.05190267
     -0.06436147 -0.06057622 -0.07461155 -0.04222599 -0.07217864 -0.03697427
     -0.02096592  0.04226105  0.06492008 -0.06692387  0.00926001  0.0138763
      0.07231332  0.02603806  0.02273759 -0.00857977 -0.00488686 -0.03683521
      0.01489123  0.04045784 -0.05820299 -0.06682371 -0.02228186 -0.06135305
     -0.01000855 -0.05706303  0.0782282  -0.05021447  0.02071947  0.06907023
     -0.03429902 -0.00119103  0.05301087 -0.02616852 -0.024908   -0.02819091
      0.06957974 -0.02587058 -0.04717125  0.04558295  0.04095564  0.00593362
     -0.04767918 -0.03579942 -0.03175095  0.06806615  0.07167613  0.07341167
      0.04470375  0.02985043 -0.01142076 -0.07169252  0.0419479  -0.00477881
      0.04641821 -0.04670449  0.00157692 -0.02617471 -0.02453358  0.01629492
     -0.07304903 -0.03545947 -0.02758045  0.04102558 -0.03854948 -0.00176723
     -0.05074076 -0.05868213  0.02295039 -0.0779191   0.00753643  0.04240848
     -0.0449448  -0.01627814 -0.03864083  0.04569025 -0.0215541  -0.00385933
      0.0687096  -0.00038805  0.06731253  0.00203905 -0.02522428  0.03640829
     -0.06817984  0.02590258  0.06238135  0.02559368 -0.00472497 -0.00768843
     -0.02542115 -0.0396123   0.04834259 -0.03672076 -0.00461542 -0.04150703
      0.04825861  0.03033854 -0.05819327 -0.02407195  0.04944583  0.02262879
      0.00537716 -0.01742457  0.01463452  0.06018772  0.01764123  0.04064748
     -0.03416735  0.03101878 -0.05000495 -0.03750736 -0.04026246  0.05893917
     -0.03406354 -0.04039322  0.0068501  -0.03224698  0.0422332  -0.00174747
      0.03877247  0.03151488 -0.04211442  0.06551541  0.01918465 -0.06001789
     -0.00637987 -0.07045291  0.03122615 -0.00347115  0.0119053   0.06561846
     -0.03955285  0.0098134   0.03993066  0.06143269 -0.05930713  0.04833847
      0.03384694  0.00298179 -0.00459251  0.02913976  0.05558946  0.00083254
     -0.04760565 -0.07094654  0.05547972  0.07688119 -0.01018946 -0.07702006
     -0.00581692  0.01322428 -0.06336655  0.03551421 -0.03503459  0.02975735
     -0.02781365 -0.04372307]
    
    Panjang dari kata setelah embedding: 512
    

### Membangun dan Menjalankan Training Model 2


```python
# Jika folder dengan path 'colorskim_checkpoint/{model.name}' sudah ada, maka skip fit model 
# untuk menghemat waktu pengembangan dan hanya load model yang sudah ada dalam folder tersebut 
# saja
# Terutama untuk model_2 yang sangat resource intensif, baik data yang didownload dari tfhub.dev
# maupun output model dari training yang cukup besar (~1GB) berbanding model_1 yang hanya menghasilkan
# model dengan ukuran 2MB, maka untuk output model dari model_2 akan disimpan di remote data version
# control dengan modul dvc atau dapat dipindahtangankan secara fisik melalui media penyimpanan
if not os.path.isdir(f'colorskim_checkpoint/{MODEL[2]}'):
        # set random seed
        tf.random.set_seed(RANDOM_STATE)
        
        # Membuat model_2 menggunakan USE
        inputs = Input(shape=[], 
                dtype=tf.string, 
                name='lapisan_input')
        lapisan_embed_pretrained = tf_hub_embedding(inputs)
        x = Conv1D(filters=UKURAN_BATCH, 
                kernel_size=5, 
                padding='same', 
                activation='relu',
                name='lapisan_konvolusional_1_dimensi')(tf.expand_dims(lapisan_embed_pretrained, axis=-1))
        x = GlobalMaxPooling1D(name='lapisan_max_pooling')(x)
        outputs = Dense(units=1, 
                        activation='sigmoid', 
                        name='lapisan_output')(x)
        model_2 = tf.keras.Model(inputs=inputs, 
                                outputs=outputs, 
                                name=MODEL[2])

        # Compile model_2
        model_2.compile(loss=BinaryCrossentropy(),
                        optimizer=Adam(),
                        metrics=['accuracy'])


        # Setup wandb init dan config
        wb.init(project=wandb['proyek'],
                entity=wandb['user'],
                name=model_2.name,
                config={'epochs': EPOCHS,
                        'n_layers': len(model_2.layers)})

        # Fit model_2
        model_2.fit(train_kata_dataset,
                    epochs=EPOCHS,
                    validation_data=test_kata_dataset,
                    callbacks=[wandb_callback(train_kata_dataset),
                               model_checkpoint(model_2.name),
                               reduce_lr_on_plateau(),
                               early_stopping()])
        
        # tutup logging wandb
        wb.finish()
        
        # load model_2
        model_2 = load_model(f'colorskim_checkpoint/{MODEL[2]}')
else:
        # hapus tf_hub_embedding
        del tf_hub_embedding
        gc.collect()
        # load model_2
        model_2 = load_model(f'colorskim_checkpoint/{MODEL[2]}')
```


```python
# Ringkasan dari model_2
model_2.summary()
```

    Model: "model_2_Conv1D_USE_embed"
    _________________________________________________________________
     Layer (type)                Output Shape              Param #   
    =================================================================
     lapisan_input (InputLayer)  [(None,)]                 0         
                                                                     
     lapisan_embedding_USE (Kera  (None, 512)              256797824 
     sLayer)                                                         
                                                                     
     tf.expand_dims (TFOpLambda)  (None, 512, 1)           0         
                                                                     
     lapisan_konvolusional_1_dim  (None, 512, 32)          192       
     ensi (Conv1D)                                                   
                                                                     
     lapisan_max_pooling (Global  (None, 32)               0         
     MaxPooling1D)                                                   
                                                                     
     lapisan_output (Dense)      (None, 1)                 33        
                                                                     
    =================================================================
    Total params: 256,798,049
    Trainable params: 225
    Non-trainable params: 256,797,824
    _________________________________________________________________
    


```python
# Plot model_2
plot_model(model_2, show_shapes=True)
```




    
![png](ColorSkim_AI_files/ColorSkim_AI_68_0.png)
    



### Eksplorasi Hasil Model 2


```python
# Evaluasi model_2
model_2.evaluate(test_kata_dataset)
```

    355/355 [==============================] - 19s 45ms/step - loss: 0.1921 - accuracy: 0.9389
    [0.19213275611400604, 0.9388599991798401]




```python
# Membuat probabilitas prediksi model_2
model_2_pred_prob = tf.squeeze(model_2.predict(test_kata_dataset))
model_2_pred_prob
```




    <tf.Tensor: shape=(11351,), dtype=float32, numpy=
    array([0.46290663, 0.06519245, 0.9924333 , ..., 0.1533481 , 0.01215318,
           0.0921701 ], dtype=float32)>




```python
# Membuat prediksi dengan model_2
model_2_pred = tf.round(model_2_pred_prob)
model_2_pred[:10]
```




    <tf.Tensor: shape=(10,), dtype=float32, numpy=array([0., 0., 1., 0., 1., 0., 0., 0., 1., 0.], dtype=float32)>




```python
# Menghitung metriks dari model_2
model_2_metrik = hitung_metrik(target=test_target,
                               prediksi=model_2_pred)
model_2_metrik
```




    {'akurasi': 0.9388600123337151,
     'presisi': 0.9390214154816886,
     'recall': 0.9388600123337151,
     'f1-score': 0.9385958102215999}




```python
# Plot residual dari model_2
residual_plot_logr(test_target, MODEL[2], model_2_metrik['akurasi'], model_2_pred_prob)
```


    
![png](ColorSkim_AI_files/ColorSkim_AI_74_0.png)
    



```python
# Confusion matrix dari model_2
plot_conf_matrix(target_label=test_target,
                 prediksi_label=model_2_pred,
                 nama_model=MODEL[2],
                 akurasi=model_2_metrik['akurasi'],
                 label_titik_x=['bukan_warna', 'warna'],
                 label_titik_y=['bukan_warna', 'warna'])
plt.figure(facecolor='w')
plt.show()
```


    
![png](ColorSkim_AI_files/ColorSkim_AI_75_0.png)
    



```python
# Menampilkan kesalahan prediksi dalam dataframe
df_kesalahan_prediksi(label_encoder=label_encoder,
                      test_data=test_data_mnb,
                      prediksi=model_2_pred,
                      probabilitas_prediksi=model_2_pred_prob,
                      order_ulang_header=['brand',
                                          'kata',
                                          'nama_artikel',
                                          'urut_kata',
                                          'total_kata',
                                          'label'])
```

|       | brand   | kata          | nama_artikel                                                                             |   urut_kata |   total_kata | label       | prediksi    | probabilitas   |
|------:|:--------|:--------------|:-----------------------------------------------------------------------------------------|------------:|-------------:|:------------|:------------|:---------------|
| 20530 | CAO     | 700SK         | GA-700SK-1ADR                                                                            |           2 |            3 | bukan_warna | warna       | 97.71%         |
| 56112 | WAR     | RED           | RED BLACK FLAT NON REFLECTIVE-RED/BLACK                                                  |           1 |            7 | bukan_warna | warna       | 97.51%         |
| 21174 | HER     | RED           | HER-HERITAGE-BOSTON RED SOX-(21L)-BAG-US                                                 |           4 |            8 | bukan_warna | warna       | 97.51%         |
| 54184 | PUM     | FIGC          | FIGC AWAY SHIRT REPLICA-PUMA WHITE-PEACOAT                                               |           1 |            7 | bukan_warna | warna       | 95.07%         |
| 54157 | PUM     | FIGC          | FIGC AWAY SHIRT REPLICA PUMA WHITE-PEACOAT                                               |           1 |            7 | bukan_warna | warna       | 95.07%         |
| 54017 | PUM     | FIGC          | FIGC ITALIA STADIUM JACKET-WHITE-PEACOAT                                                 |           1 |            6 | bukan_warna | warna       | 95.07%         |
| 27571 | NIK     | HYPERVENOM    | HYPERVENOM PHATAL III DF FG-LASER ORANGE/WHITE-BLACK-VOLT                                |           1 |           10 | bukan_warna | warna       | 94.85%         |
| 32944 | NIK     | HYPERVENOM    | HYPERVENOM 3 CLUB FGWHITE/MTLC COOL GREYVOLTMTLC COOL GREY                               |           1 |            9 | bukan_warna | warna       | 94.85%         |
| 32913 | NIK     | HYPERVENOM    | HYPERVENOM 3 ACADEMY IC-BLACK/MTLC VIVID GOLD                                            |           1 |            8 | bukan_warna | warna       | 94.85%         |
| 25205 | NIK     | HYPERVENOM    | HYPERVENOM PHATAL II DF FG-BLACK/BLACK-MTLC HEMATITE                                     |           1 |            9 | bukan_warna | warna       | 94.85%         |
| 32901 | NIK     | HYPERVENOM    | HYPERVENOM 3 PRO DF FGLT CRIMSON/MTLC DARK GREYWOLF GREY                                 |           1 |           10 | bukan_warna | warna       | 94.85%         |
| 27576 | NIK     | HYPERVENOM    | HYPERVENOM PHELON III FG-LASER ORANGE/WHITE-BLACK-VOLT                                   |           1 |            9 | bukan_warna | warna       | 94.85%         |
| 32951 | NIK     | HYPERVENOM    | JR HYPERVENOM 3 CLUB FGLT CRIMSON/MTLC DARK GREYWOLF GREY                                |           2 |           10 | bukan_warna | warna       | 94.85%         |
| 27594 | NIK     | HYPERVENOM    | HYPERVENOM PHANTOM III FG-ELECTRIC GREEN/BLACK-HYPER ORANGE-                             |           1 |            9 | bukan_warna | warna       | 94.85%         |
| 32893 | NIK     | HYPERVENOM    | HYPERVENOM 3 PRO DF FG-WHITE/MTLC COOL GREY-VOLT-MTLC COOL G                             |           1 |           13 | bukan_warna | warna       | 94.85%         |
| 27568 | NIK     | HYPERVENOM    | HYPERVENOM PHATAL III DF FG-ELECTRIC GREEN/BLACK-HYPER ORANG                             |           1 |           10 | bukan_warna | warna       | 94.85%         |
| 15616 | ADI     | EDGE.3        | PREDATOR EDGE.3 IN-CORE BLACK                                                            |           2 |            5 | bukan_warna | warna       | 94.49%         |
| 48162 | PTG     | GREEN         | TREA GREEN-GREEN                                                                         |           2 |            3 | bukan_warna | warna       | 94.05%         |
| 34040 | NIK     | SUPRFLY       | JR SUPRFLY 6 ACADEMY GS NJR MGAMARILLO/WHITEBLACK                                        |           2 |            8 | bukan_warna | warna       | 93.28%         |
| 11286 | ADI     | RETRORUN      | RETRORUN-CORE BLACK                                                                      |           1 |            3 | bukan_warna | warna       | 92.55%         |
| 14406 | ADI     | MICROPACER_R1 | MICROPACER_R1-CORE BLACK                                                                 |           1 |            3 | bukan_warna | warna       | 90.84%         |
|  9197 | ADI     | ANK           | TREFOIL ANK STR-MAROON                                                                   |           2 |            4 | bukan_warna | warna       | 90.51%         |
|  9480 | ADI     | ANK           | LIGHT ANK 1PP-WHITE                                                                      |           2 |            4 | bukan_warna | warna       | 90.51%         |
|  9502 | ADI     | ANK           | LIGHT ANK 3PP-WHITE                                                                      |           2 |            4 | bukan_warna | warna       | 90.51%         |
|  9776 | ADI     | ANK           | TREF ANK SCK HC-BLACK                                                                    |           2 |            5 | bukan_warna | warna       | 90.51%         |
| 16288 | ADI     | BLK           | CLR BLK CRW 2PP-BLACK                                                                    |           2 |            5 | bukan_warna | warna       | 90.16%         |
| 15600 | ADI     | SPEEDFLOW.3   | X SPEEDFLOW.3 IN-SKY RUSH                                                                |           2 |            5 | bukan_warna | warna       | 89.42%         |
| 13706 | ADI     | SPEEDFLOW.3   | X SPEEDFLOW.3 FG J-RED                                                                   |           2 |            5 | bukan_warna | warna       | 89.42%         |
| 13709 | ADI     | SPEEDFLOW.3   | X SPEEDFLOW.3 IN J-RED                                                                   |           2 |            5 | bukan_warna | warna       | 89.42%         |
| 24214 | NIK     | OKWAHN        | OKWAHN II-GOLDEN BEIGE/DEEP ROYAL BLUE-TRUE BERRY                                        |           1 |            9 | bukan_warna | warna       | 89.27%         |
| 55642 | STN     | MOUSEKETEER   | STN-MOUSEKETEER-MULTI-(M)-ACC-MN                                                         |           2 |            6 | bukan_warna | warna       | 89.19%         |
| 17198 | AGL     | YELLOW        | ESAGLXY YELLOW CRICKET LIGHTER -YELLOW                                                   |           2 |            5 | bukan_warna | warna       | 89.07%         |
| 20168 | CAO     | 8DR           | DW-5600CA-8DR                                                                            |           3 |            3 | bukan_warna | warna       | 88.99%         |
| 20256 | CAO     | 8DR           | DW-6900LU-8DR-GRAY                                                                       |           3 |            4 | bukan_warna | warna       | 88.99%         |
|  5757 | ADI     | ALTARUN       | ALTARUN CF I-REAL MAGENTA                                                                |           1 |            5 | bukan_warna | warna       | 88.32%         |
|  5718 | ADI     | ALTARUN       | ALTARUN CF K-BLUE/FTWR WHITE/BLUE                                                        |           1 |            7 | bukan_warna | warna       | 88.32%         |
| 30229 | NIK     | LUNARCHARGE   | NIKE LUNARCHARGE ESSENTIAL-BLACK/TEAM RED-TEAM RED                                       |           2 |            8 | bukan_warna | warna       | 88.03%         |
| 55080 | SOC     | SNIPPLE       | SNIPPLE LOW ECLIPSE-BLACK/MULTI                                                          |           1 |            5 | bukan_warna | warna       | 86.55%         |
| 55505 | STN     | JOVEN         | STN-JOVEN-GREY-(M)-ACC-US                                                                |           2 |            6 | bukan_warna | warna       | 86.25%         |
| 55487 | STN     | JOVEN         | STN-JOVEN-BLACK-(M)-ACC-US                                                               |           2 |            6 | bukan_warna | warna       | 86.25%         |
| 55463 | STN     | JOVEN         | STANCE JOVEN FOUNDATION GREY                                                             |           2 |            4 | bukan_warna | warna       | 86.25%         |
|  9201 | ADI     | GYMSACK       | GYMSACK TREFOIL-BLACK/NGTCAR                                                             |           1 |            4 | bukan_warna | warna       | 84.56%         |
|  9135 | ADI     | GYMSACK       | GYMSACK 3D-BLACK                                                                         |           1 |            3 | bukan_warna | warna       | 84.56%         |
|  3544 | ADI     | GYMSACK       | GYMSACK 3D-WHITE                                                                         |           1 |            3 | bukan_warna | warna       | 84.56%         |
| 32507 | NIK     | METCON        | NIKE METCON 4-BLACK/WHITE                                                                |           2 |            5 | bukan_warna | warna       | 83.89%         |
| 27626 | NIK     | METCON        | NIKE METCON 3-BLACK/WHITE-METALLIC SILVER                                                |           2 |            7 | bukan_warna | warna       | 83.89%         |
| 18880 | BBC     | OVERDYED      | OVERDYED BUTTON-UP POPOVER HOODIE-OLIVE                                                  |           1 |            6 | bukan_warna | warna       | 83.76%         |
| 52023 | PUM     | ANZARUN       | ANZARUN LITE PUMA BLACK-PUMA WHITE                                                       |           1 |            6 | bukan_warna | warna       | 83.68%         |
| 13176 | ADI     | MUTATOR       | PREDATOR MUTATOR 20.1 L FG-SKY TINT                                                      |           2 |            7 | bukan_warna | warna       | 82.25%         |
| 11314 | ADI     | MUTATOR       | PREDATOR MUTATOR 20+ FG-SIGNAL GREEN                                                     |           2 |            6 | bukan_warna | warna       | 82.25%         |
| 10910 | ADI     | MUTATOR       | PREDATOR MUTATOR 20.1 FG-TEAM ROYAL BLUE                                                 |           2 |            7 | bukan_warna | warna       | 82.25%         |
|   895 | ADI     | CONFED        | CONFED GLIDNOLO-WHITE/BRIRED/RED/BLACK                                                   |           1 |            6 | bukan_warna | warna       | 81.99%         |
| 56226 | WAR     | ORANGE        | SHOELACES ORANGE OVAL LACES-ORANGE                                                       |           2 |            5 | bukan_warna | warna       | 81.34%         |
| 48075 | PTG     | ORANGE        | POLKA ORANGE-ORANGE                                                                      |           2 |            3 | bukan_warna | warna       | 81.34%         |
| 17081 | ADI     | ADISOCK       | ADISOCK 12-WHT/BLACK                                                                     |           1 |            4 | bukan_warna | warna       | 81.34%         |
| 12189 | ADI     | PARKHOOD      | PARKHOOD WB-BLACK                                                                        |           1 |            3 | bukan_warna | warna       | 80.63%         |
| 32109 | NIK     | OBRAX         | OBRAX 2 ACADEMY DF IC-WHITE/MTLC COOL GREY-LT CRIMSON                                    |           1 |           11 | bukan_warna | warna       | 80.22%         |
| 32120 | NIK     | OBRAX         | OBRAX 2 CLUB IC-WHITE/MTLC COOL GREY-LT CRIMSON                                          |           1 |           10 | bukan_warna | warna       | 80.22%         |
| 32114 | NIK     | OBRAX         | OBRAX 2 CLUB IC-DARK GREY/BLACK-TOTAL ORANGE-WHITE                                       |           1 |           10 | bukan_warna | warna       | 80.22%         |
| 22725 | KIP     | FS68          | FS68-BEIGE BRAID-115                                                                     |           1 |            4 | bukan_warna | warna       | 79.76%         |
| 49108 | PUM     | LALIGA        | LALIGA 1 FIFA QUALITY PRO PINK ALERT-YELLOW ALERT                                        |           1 |            9 | bukan_warna | warna       | 78.92%         |
| 48125 | PTG     | TANNE         | TANNE GREY-GREY                                                                          |           1 |            3 | bukan_warna | warna       | 78.4%          |
| 22680 | KIP     | FS64          | FS64-BLUE YELOW ZIGZAG-90                                                                |           1 |            5 | bukan_warna | warna       | 77.84%         |
| 20708 | CAO     | 5600B         | GM-5600B-3DR                                                                             |           2 |            3 | bukan_warna | warna       | 77.72%         |
| 47988 | PSB     | PERSIB        | SCARF PERSIB LOGO - BLUE                                                                 |           2 |            4 | bukan_warna | warna       | 77.61%         |
|  1706 | ADI     | DEERUPT       | DEERUPT W-GREONE/GREONE/AERBLU                                                           |           1 |            5 | bukan_warna | warna       | 77.13%         |
|  6861 | ADI     | DEERUPT       | DEERUPT RUNNER PARLEY-CBLACK/CBLACK/BLUSPI                                               |           1 |            6 | bukan_warna | warna       | 77.13%         |
|  6886 | ADI     | DEERUPT       | DEERUPT RUNNER-GRETHR/LGSOGR/GUM1                                                        |           1 |            5 | bukan_warna | warna       | 77.13%         |
|  1716 | ADI     | DEERUPT       | DEERUPT RUNNER-CBLACK/CBLACK/ASHPEA                                                      |           1 |            5 | bukan_warna | warna       | 77.13%         |
|  6952 | ADI     | DEERUPT       | DEERUPT RUNNER W-CBLACK/CBLACK/CHAPNK                                                    |           1 |            6 | bukan_warna | warna       | 77.13%         |
|  6881 | ADI     | DEERUPT       | DEERUPT RUNNER-GRETHR/GREFOU/FTWWHT                                                      |           1 |            5 | bukan_warna | warna       | 77.13%         |
|  1746 | ADI     | DEERUPT       | DEERUPT RUNNER-DKBLUE/DKBLUE/ASHBLU                                                      |           1 |            5 | bukan_warna | warna       | 77.13%         |
|  1741 | ADI     | DEERUPT       | DEERUPT RUNNER-BASGRN/BASGRN/ORANGE                                                      |           1 |            5 | bukan_warna | warna       | 77.13%         |
|  6891 | ADI     | DEERUPT       | DEERUPT RUNNER-CWHITE/CBLACK/CBLACK                                                      |           1 |            5 | bukan_warna | warna       | 77.13%         |
|  2828 | ADI     | SOBAKOV       | SOBAKOV-CBLACK/CBLACK/FLARED                                                             |           1 |            4 | bukan_warna | warna       | 76.76%         |
|  1799 | ADI     | SOBAKOV       | SOBAKOV-FTWWHT/CRYWHT/CRYWHT                                                             |           1 |            4 | bukan_warna | warna       | 76.76%         |
|  2899 | ADI     | SOBAKOV       | SOBAKOV FTWWHT/FTWWHT/GUM3                                                               |           1 |            4 | bukan_warna | warna       | 76.76%         |
| 37425 | NIK     | ELMNTL        | NK ELMNTL BKPK - 20-BLACK/BLACK/WHITE                                                    |           2 |            7 | bukan_warna | warna       | 76.67%         |
| 45336 | NIK     | ELMNTL        | NK ELMNTL BKPK ? HBR-BLACK/BLACK/WHITE                                                   |           2 |            8 | bukan_warna | warna       | 76.67%         |
| 37442 | NIK     | ELMNTL        | NK ELMNTL BKPK-20-ECHO PINK/ECHO PINK/METALLIC GOLD                                      |           2 |           10 | bukan_warna | warna       | 76.67%         |
| 37431 | NIK     | ELMNTL        | NK ELMNTL BKPK - 2.0-EARTH/EARTH/PALE IVORY                                              |           2 |            8 | bukan_warna | warna       | 76.67%         |
| 45342 | NIK     | ELMNTL        | NK ELMNTL BKPK ? HBR-IRON GREY/IRON GREY/BLACK                                           |           2 |           10 | bukan_warna | warna       | 76.67%         |
| 37005 | NIK     | ELMNTL        | NK ELMNTL BKPK-NOBLE RED/BLACK/BORDEAUX                                                  |           2 |            7 | bukan_warna | warna       | 76.67%         |
| 20660 | CAO     | 200RD         | GBD-200RD-4DR                                                                            |           2 |            3 | bukan_warna | warna       | 76.47%         |
| 21248 | HER     | LILAMER       | HER-LILAMER LT-BLACK-(OS)-BAG-US                                                         |           2 |            7 | bukan_warna | warna       | 76.38%         |
|  7371 | ADI     | PROPHERE      | PROPHERE-SGREEN/CGREEN/CBLACK                                                            |           1 |            4 | bukan_warna | warna       | 75.27%         |
|  6852 | ADI     | PROPHERE      | PROPHERE W-FTWWHT/FTWWHT/SUPCOL                                                          |           1 |            5 | bukan_warna | warna       | 75.27%         |
|  1386 | ADI     | PROPHERE      | PROPHERE-SBROWN/CBLACK/CBROWN                                                            |           1 |            4 | bukan_warna | warna       | 75.27%         |
|  5715 | ADI     | PROPHERE      | PROPHERE W-FTWR WHITE                                                                    |           1 |            4 | bukan_warna | warna       | 75.27%         |
|  1257 | ADI     | PROPHERE      | PROPHERE-FTWWHT/FTWWHT/CRYWHT                                                            |           1 |            4 | bukan_warna | warna       | 75.27%         |
| 12263 | ADI     | CLSC          | CLSC S BP-BLACK                                                                          |           1 |            4 | bukan_warna | warna       | 75.23%         |
| 16229 | ADI     | CLSC          | CLSC BP M-BLACK                                                                          |           1 |            4 | bukan_warna | warna       | 75.23%         |
| 12260 | ADI     | CLSC          | CLSC M T4H-BLACK                                                                         |           1 |            4 | bukan_warna | warna       | 75.23%         |
| 28741 | NIK     | ARROWZ        | NIKE ARROWZ-BLACK/WHITE-DARK GREY                                                        |           2 |            6 | bukan_warna | warna       | 75.17%         |
| 28470 | NIK     | TIEMPO        | TIEMPO LEGEND VII FG-BLACK/WHITE-BLACK                                                   |           1 |            7 | bukan_warna | warna       | 74.9%          |
| 28490 | NIK     | TIEMPO        | TIEMPO RIO IV FG-UNIVERSITY RED/WHITE-BLACK                                              |           1 |            8 | bukan_warna | warna       | 74.9%          |
| 28452 | NIK     | TIEMPO        | JR TIEMPO LIGERA IV FG-GAMMA BLUE/WHITE-OBSIDIAN-GLACIER BLU                             |           2 |           11 | bukan_warna | warna       | 74.9%          |
| 20310 | CAO     | 100BT         | GA-100BT-1ADR-BLACK                                                                      |           2 |            4 | bukan_warna | warna       | 74.83%         |
| 37383 | NIK     | ACDMY         | NK ACDMY SHOEBAG-BLACK/LASER CRIMSON/METALLIC BLACK                                      |           2 |            8 | bukan_warna | warna       | 74.72%         |
| 26544 | NIK     | ACDMY         | Y NK ACDMY SHORT JAQ K-BLACK/WHITE/WHITE/WHITE                                           |           3 |           10 | bukan_warna | warna       | 74.72%         |
| 37378 | NIK     | ACDMY         | NK ACDMY SHOEBAG-BLACK/BLACK/WHITE                                                       |           2 |            6 | bukan_warna | warna       | 74.72%         |
| 13811 | ADI     | OWNTHEGAME    | OWNTHEGAME-CORE BLACK                                                                    |           1 |            3 | bukan_warna | warna       | 74.23%         |
| 10316 | ADI     | OWNTHEGAME    | OWNTHEGAME-FTWR WHITE                                                                    |           1 |            3 | bukan_warna | warna       | 74.23%         |
| 16035 | ADI     | OWNTHEGAME    | OWNTHEGAME 2.0-CORE BLACK                                                                |           1 |            4 | bukan_warna | warna       | 74.23%         |
| 19926 | CAO     | 2B1DR         | BGA-150PG-2B1DR                                                                          |           3 |            3 | bukan_warna | warna       | 73.21%         |
| 50284 | PUM     | TISHATSU      | TISHATSU RUNNER MESH PUMA BLACK-SAFETY YELLOW                                            |           1 |            7 | bukan_warna | warna       | 73.17%         |
| 27055 | NIK     | ONDA          | MAGISTA ONDA II FG-LASER ORANGE/BLACK-WHITE-VOLT                                         |           2 |            9 | bukan_warna | warna       | 73.03%         |
| 27060 | NIK     | ONDA          | MAGISTAX ONDA II IC-BLACK/WHITE-UNIVERSITY RED                                           |           2 |            8 | bukan_warna | warna       | 73.03%         |
| 27050 | NIK     | ONDA          | MAGISTA ONDA II FG-BLACK/WHITE-UNIVERSITY RED                                            |           2 |            8 | bukan_warna | warna       | 73.03%         |
| 16247 | ADI     | MULTICOLOR    | WAISTBAG-MULTICOLOR                                                                      |           2 |            2 | bukan_warna | warna       | 72.86%         |
| 11107 | ADI     | MULTICOLOR    | ULTRABOOST MULTICOLOR-COLLEGIATE NAVY                                                    |           2 |            4 | bukan_warna | warna       | 72.86%         |
| 11185 | ADI     | X9000L3       | X9000L3 M-CORE BLACK                                                                     |           1 |            4 | bukan_warna | warna       | 72.55%         |
| 11176 | ADI     | X9000L3       | X9000L3 W-CORE BLACK                                                                     |           1 |            4 | bukan_warna | warna       | 72.55%         |
|  1079 | ADI     | SAMBAROSE     | SAMBAROSE W-CBLACK/CBLACK/CBLACK                                                         |           1 |            5 | bukan_warna | warna       | 72.5%          |
| 14069 | ADI     | SAMBAROSE     | SAMBAROSE W-FTWR WHITE                                                                   |           1 |            4 | bukan_warna | warna       | 72.5%          |
|  1063 | ADI     | SAMBAROSE     | SAMBAROSE W-CBLACK/FTWWHT/GUM5                                                           |           1 |            5 | bukan_warna | warna       | 72.5%          |
| 12807 | ADI     | SAMBAROSE     | SAMBAROSE W-SILVER MET.                                                                  |           1 |            4 | bukan_warna | warna       | 72.5%          |
| 14131 | ADI     | X9000L4       | X9000L4 CYBERPUNK 2077-CORE BLACK                                                        |           1 |            5 | bukan_warna | warna       | 72.17%         |
| 16588 | ADI     | X9000L4       | X9000L4 M-FTWR WHITE                                                                     |           1 |            4 | bukan_warna | warna       | 72.17%         |
| 13687 | ADI     | X9000L4       | X9000L4-FTWR WHITE                                                                       |           1 |            3 | bukan_warna | warna       | 72.17%         |
| 13304 | ADI     | X9000L4       | X9000L4 W-CRYSTAL WHITE                                                                  |           1 |            4 | bukan_warna | warna       | 72.17%         |
| 13296 | ADI     | X9000L4       | X9000L4-GREY THREE                                                                       |           1 |            3 | bukan_warna | warna       | 72.17%         |
| 16594 | ADI     | X9000L4       | X9000L4 W-CORE BLACK                                                                     |           1 |            4 | bukan_warna | warna       | 72.17%         |
| 22457 | KIP     | RS46          | RS46-BLACK WHITE WEB-140                                                                 |           1 |            5 | bukan_warna | warna       | 71.93%         |
| 22453 | KIP     | RS46          | RS46-BLACK WHITE WEB-115                                                                 |           1 |            5 | bukan_warna | warna       | 71.93%         |
| 19524 | BBC     | BLEACHED      | BLEACHED LOGO T-SHIRT-BLACK BLEACHED                                                     |           1 |            6 | bukan_warna | warna       | 71.67%         |
|  7638 | ADI     | QTFLEX        | CF QTFLEX W-CONAVY/SILVMT/FTWWHT                                                         |           2 |            6 | bukan_warna | warna       | 71.42%         |
|  1919 | ADI     | QTFLEX        | QTFLEX-ASHGRN/RAWGRN/CLEMIN                                                              |           1 |            4 | bukan_warna | warna       | 71.42%         |
| 20513 | CAO     | 3ADR          | GA-700CM-3ADR                                                                            |           3 |            3 | bukan_warna | warna       | 71.37%         |
| 20470 | CAO     | 3ADR          | GA-2110SU-3ADR                                                                           |           3 |            3 | bukan_warna | warna       | 71.37%         |
| 55555 | STN     | LOGOMAN       | NBA LOGOMAN CREW II-BLACK                                                                |           2 |            5 | bukan_warna | warna       | 71.27%         |
| 46663 | NIK     | ASUNK         | ASUNK SOLO SWSH SATIN BMBRJKT-BLACK/DOLL/WHITE                                           |           1 |            8 | bukan_warna | warna       | 71.18%         |
| 20466 | CAO     | 2110ET        | GA-2110ET-2ADR                                                                           |           2 |            3 | bukan_warna | warna       | 70.88%         |
| 17926 | AGL     | TACOLOCAL     | TACOLOCAL SINCE-BLACK                                                                    |           1 |            3 | bukan_warna | warna       | 70.23%         |
| 20664 | CAO     | GBD8002DRNS   | GBD8002DRNS                                                                              |           1 |            1 | bukan_warna | warna       | 70.19%         |
| 20182 | CAO     | 5600HR        | DW-5600HR-1DR                                                                            |           2 |            3 | bukan_warna | warna       | 70.09%         |
| 42357 | NIK     | ONDECK        | NIKE ONDECK FLIP FLOP-BLACK/WHITE-BLACK                                                  |           2 |            7 | bukan_warna | warna       | 69.73%         |
| 54031 | PUM     | BORUSSE       | BVB BORUSSE TEE-CYBER YELLOW-PUMA BLACK                                                  |           2 |            7 | bukan_warna | warna       | 69.45%         |
| 24919 | NIK     | YTH           | ENT YTH SS HM STADIUM JSY-WHITE/BLUE GREY/SPORT ROYAL                                    |           2 |           11 | bukan_warna | warna       | 69.09%         |
| 24924 | NIK     | YTH           | FFF YTH SS HM STADIUM JSY-HYPER COBALT/WHITE                                             |           2 |            9 | bukan_warna | warna       | 69.09%         |
|  8818 | ADI     | CONEXT19      | CONEXT19 SALTRN-WHITE                                                                    |           1 |            3 | bukan_warna | warna       | 68.85%         |
|  8771 | ADI     | CONEXT19      | CONEXT19 TCPT-WHITE                                                                      |           1 |            3 | bukan_warna | warna       | 68.85%         |
|  8776 | ADI     | CONEXT19      | CONEXT19 CPT-RAW WHITE                                                                   |           1 |            4 | bukan_warna | warna       | 68.85%         |
| 34704 | NIK     | EBERNON       | NIKE EBERNON LOW-WOLF GREY/BLACK-WHITE                                                   |           2 |            7 | bukan_warna | warna       | 68.67%         |
| 34700 | NIK     | EBERNON       | NIKE EBERNON LOW-BLACK/WHITE                                                             |           2 |            5 | bukan_warna | warna       | 68.67%         |
| 52163 | PUM     | FTR           | LEADCAT FTR PUMA BLACK-PUMA TEAM GOLD-PUMA WHITE                                         |           2 |            9 | bukan_warna | warna       | 67.71%         |
| 12464 | ADI     | BOTTL         | PERF BOTTL 0,75-WHITE                                                                    |           2 |            4 | bukan_warna | warna       | 67.36%         |
| 22701 | KIP     | FS66          | FS66-GREY WHITE BRAID-90                                                                 |           1 |            5 | bukan_warna | warna       | 67.33%         |
| 20233 | CAO     | 5900RS        | DW-5900RS-1DR                                                                            |           2 |            3 | bukan_warna | warna       | 67.28%         |
| 12291 | ADI     | STARLANCER    | STARLANCER CLB-LIGHT AQUA                                                                |           1 |            4 | bukan_warna | warna       | 66.83%         |
|  4889 | ADI     | STARLANCER    | STARLANCER V-SOLRED/WHITE/BLACK                                                          |           1 |            5 | bukan_warna | warna       | 66.83%         |
|  5570 | ADI     | ALPHAEDGE     | ALPHAEDGE 4D M-FTWR WHITE                                                                |           1 |            5 | bukan_warna | warna       | 66.83%         |
| 39749 | NIK     | ATSUMA        | NIKE ATSUMA-OFF NOIR/IRON GREY-BLACK-WHITE                                               |           2 |            8 | bukan_warna | warna       | 66.4%          |
|   777 | ADI     | INF           | RACER TR INF-SCARLE/CBLACK/FTWWHT                                                        |           3 |            6 | bukan_warna | warna       | 65.23%         |
|  9636 | ADI     | INF           | BP INF-MULTICOLOR                                                                        |           2 |            3 | bukan_warna | warna       | 65.23%         |
| 20480 | CAO     | CSWGYOSAD     | CSWGYOSAD GA400GB1A9DR-BLACK                                                             |           1 |            3 | bukan_warna | warna       | 64.98%         |
| 29951 | NIK     | MERCURIALX    | MERCURIALX VORTEX III NJR IC-RACER BLUE/BLACK-CHROME-VOLT                                |           1 |           10 | bukan_warna | warna       | 64.81%         |
| 26239 | NIK     | MERCURIALX    | JR MERCURIALX VICTORY VI IC-LASER ORANGE/BLACK-WHITE-VOLT                                |           2 |           10 | bukan_warna | warna       | 64.81%         |
| 26324 | NIK     | MERCURIALX    | MERCURIALX VICTORY VI IC-TOTAL CRIMSON/VOLT-BLACK-PINK BLAST                             |           1 |           10 | bukan_warna | warna       | 64.81%         |
| 24983 | NIK     | MERCURIALX    | NIKE MERCURIALX PRO (IC)-MID NVY/MID NVY-PNK BLST-RCR B                                  |           2 |           12 | bukan_warna | warna       | 64.81%         |
| 26315 | NIK     | MERCURIALX    | MERCURIALX VICTORY VI IC-UNIVERSITY RED/BLACK-BRIGHT CRIMSON                             |           1 |            9 | bukan_warna | warna       | 64.81%         |
| 25550 | NIK     | MERCURIALX    | MERCURIALX PROXIMO CR IC-DP RYL BL/MTLLC SLVR-RCR BL-BL                                  |           1 |           12 | bukan_warna | warna       | 64.81%         |
| 27543 | NIK     | MERCURIALX    | MERCURIALX VICTORY VI CR7 IC-BLUE TINT/BLACK-WHITE-BLUE TINT                             |           1 |           11 | bukan_warna | warna       | 64.81%         |
| 13382 | ADI     | ULTIMASHOW    | ULTIMASHOW-CORE BLACK                                                                    |           1 |            3 | bukan_warna | warna       | 64.8%          |
| 15182 | ADI     | FBIRD         | FBIRD TT-BLACK                                                                           |           1 |            3 | bukan_warna | warna       | 64.54%         |
| 55478 | STN     | UMPQUA        | STANCE UMPQUA HIKE-ADVENTURE GREY L-GREY-L                                               |           2 |            8 | bukan_warna | warna       | 64.32%         |
| 17300 | AGL     | L750          | L750 SWEATER 007 - BURGUNDY                                                              |           1 |            4 | bukan_warna | warna       | 64.14%         |
| 20775 | CIT     | AF            | T-SHIRT AF 1-BLACK                                                                       |           3 |            5 | bukan_warna | warna       | 64.09%         |
| 48475 | PUM     | EVOSPEED      | EVOSPEED BACKPACK-PUMA BLACK-GREEN GECKO-SAFETY YELLOW                                   |           1 |            8 | bukan_warna | warna       | 64.06%         |
| 12340 | ADI     | TOPLOADER     | PE TOPLOADER BP-GREY FIVE                                                                |           2 |            5 | bukan_warna | warna       | 64.02%         |
| 20727 | CAO     | S6900MC       | GMD-S6900MC-3DR                                                                          |           2 |            3 | bukan_warna | warna       | 63.97%         |
| 10768 | ADI     | FORTARUN      | FORTARUN X SUMMER.RDY CF K-DASH GREEN                                                    |           1 |            7 | bukan_warna | warna       | 63.57%         |
|  7336 | ADI     | FORTARUN      | FORTARUN MICKEY AC I-CORE BLACK                                                          |           1 |            6 | bukan_warna | warna       | 63.57%         |
| 10359 | ADI     | FORTARUN      | FORTARUN AC K-CORE BLACK                                                                 |           1 |            5 | bukan_warna | warna       | 63.57%         |
|  1751 | ADI     | FORTARUN      | FORTARUN X CF K-DKBLUE/CLELIL/REALIL                                                     |           1 |            7 | bukan_warna | warna       | 63.57%         |
| 13895 | ADI     | FORTARUN      | FORTARUN LEGO NINJAGO EL K-SHOCK BLUE                                                    |           1 |            7 | bukan_warna | warna       | 63.57%         |
| 20547 | CAO     | 1A2DR         | GA-710-1A2DR                                                                             |           3 |            3 | bukan_warna | warna       | 63.21%         |
| 20491 | CAO     | 1BDR          | GA-700-1BDR                                                                              |           3 |            3 | bukan_warna | warna       | 62.83%         |
| 20762 | CAO     | 1BDR          | GW-B5600BC-1BDR                                                                          |           3 |            3 | bukan_warna | warna       | 62.83%         |
| 29699 | NIK     | DUALTONE      | NIKE DUALTONE RACER (GS)-BLACK/WHITE-PALE GREY                                           |           2 |            8 | bukan_warna | warna       | 62.25%         |
| 29763 | NIK     | DUALTONE      | NIKE DUALTONE RACER-BLACK/WHITE-DARK GREY                                                |           2 |            7 | bukan_warna | warna       | 62.25%         |
|   652 | ADI     | STLT          | NMD R1 STLT PK-CBLACK/NOBGRN/BGREEN                                                      |           3 |            7 | bukan_warna | warna       | 62.02%         |
|  9936 | ADI     | MAGMUR        | MAGMUR RUNNER W-ST DESERT SAND                                                           |           1 |            6 | bukan_warna | warna       | 61.4%          |
| 56043 | VAP     | VAPUR         | VAPUR KIDS-FUSE                                                                          |           1 |            3 | bukan_warna | warna       | 60.88%         |
| 56041 | VAP     | VAPUR         | VAPUR KIDS-BO                                                                            |           1 |            3 | bukan_warna | warna       | 60.88%         |
|  6439 | ADI     | ALTASWIM      | ALTASWIM C-REATEA/HIREOR/ASHGRE                                                          |           1 |            5 | bukan_warna | warna       | 60.68%         |
|  2304 | ADI     | ALTASWIM      | ALTASWIM C-BLUE/FTWWHT/FTWWHT                                                            |           1 |            5 | bukan_warna | warna       | 60.68%         |
| 11634 | ADI     | ALTASWIM      | ALTASWIM C-AERO BLUE S18/FTWR WHITE/FTWR WHITE                                           |           1 |            9 | bukan_warna | warna       | 60.68%         |
| 10938 | ADI     | STEPBACK      | HARDEN STEPBACK-CORE BLACK                                                               |           2 |            4 | bukan_warna | warna       | 59.97%         |
| 20418 | CAO     | 2000SU        | GA-2000SU-1ADR                                                                           |           2 |            3 | bukan_warna | warna       | 59.95%         |
| 11359 | ADI     | C40           | AFC C40 CAP-SCARLET                                                                      |           2 |            4 | bukan_warna | warna       | 59.72%         |
|  5236 | ADI     | C40           | C40 5P CLMLT CA-COLLEGIATE NAVY/WHITE/WHITE                                              |           1 |            8 | bukan_warna | warna       | 59.72%         |
|  3750 | ADI     | AFA           | AFA H SHO-BLACK/CLBLUE/WHITE                                                             |           1 |            6 | bukan_warna | warna       | 59.08%         |
| 12714 | ADI     | CTY           | ULTRABOOST CTY-CORE BLACK                                                                |           2 |            4 | bukan_warna | warna       | 59.04%         |
|  2014 | ADI     | TIRO          | TIRO SB-BLACK/DKGREY/WHITE                                                               |           1 |            5 | bukan_warna | warna       | 58.89%         |
|  3962 | ADI     | TIRO          | TIRO SB-BLUE/CONAVY/WHITE                                                                |           1 |            5 | bukan_warna | warna       | 58.89%         |
| 15367 | ADI     | TIRO          | TIRO GS-BLACK                                                                            |           1 |            3 | bukan_warna | warna       | 58.89%         |
| 14773 | ADI     | TIRO          | TIRO SB-BLACK                                                                            |           1 |            3 | bukan_warna | warna       | 58.89%         |
| 20326 | CAO     | 5ADR          | GA-100CM-5ADR-BROWN                                                                      |           3 |            4 | bukan_warna | warna       | 58.48%         |
| 20473 | CAO     | 5ADR          | GA-2200MFR-5ADR                                                                          |           3 |            3 | bukan_warna | warna       | 58.48%         |
| 41355 | NIK     | NJR           | NJR NK MERC LT-SU20-WHITE/VOLT/RED ORBIT/BLACK                                           |           1 |           10 | bukan_warna | warna       | 58.31%         |
| 53690 | PUM     | NJR           | NJR 2.0 TRACK JACKET PUMA BLACK                                                          |           1 |            6 | bukan_warna | warna       | 58.31%         |
| 19829 | CAO     | AW5914ADR     | CSWGYOSA AW5914ADR-BLACK                                                                 |           2 |            3 | bukan_warna | warna       | 58.17%         |
| 10067 | ADI     | BYW           | CRAZY BYW III-CORE BLACK                                                                 |           2 |            5 | bukan_warna | warna       | 57.86%         |
| 10233 | ADI     | BYW           | CRAZY BYW III-TECH INK                                                                   |           2 |            5 | bukan_warna | warna       | 57.86%         |
|  5159 | ADI     | SHOEBAG       | DFB SHOEBAG-BLACK/WHITE                                                                  |           2 |            4 | bukan_warna | warna       | 57.34%         |
| 48089 | PTG     | VERMI         | VERMI BLACK-BLACK                                                                        |           1 |            3 | bukan_warna | warna       | 56.81%         |
| 25943 | NIK     | ESSENTIALIST  | NIKE ESSENTIALIST-RACER BLUE/LOYAL BLUE-WHITE                                            |           2 |            7 | bukan_warna | warna       | 56.6%          |
|  9246 | ADI     | 2PP           | SOLID CREW 2PP-DARK BLUE                                                                 |           3 |            5 | bukan_warna | warna       | 55.75%         |
| 11241 | ADI     | S.RDY         | ULTRABOOST S.RDY W-FTWR WHITE                                                            |           2 |            5 | bukan_warna | warna       | 55.58%         |
| 12386 | ADI     | TRF           | TRF DRESS-NIGHT MARINE                                                                   |           1 |            4 | bukan_warna | warna       | 55.14%         |
| 13173 | ADI     | SENSE.4       | COPA SENSE.4 IN-FTWR WHITE                                                               |           2 |            5 | bukan_warna | warna       | 54.74%         |
| 10340 | ADI     | VALASION      | 90S VALASION-CORE BLACK                                                                  |           2 |            4 | bukan_warna | warna       | 54.7%          |
| 54901 | SAU     | LOWPRO        | JAZZ LOWPRO-GRAY/WHITE                                                                   |           2 |            4 | bukan_warna | warna       | 54.57%         |
| 47559 | NIK     | 3PPK          | NIKE 3PPK DRI-FIT LGHTWT HI-LO-BLACK/(FLINT GREY)                                        |           2 |           10 | bukan_warna | warna       | 53.58%         |
| 47562 | NIK     | 3PPK          | NIKE 3PPK DRI-FIT LGHTWT HI-LO-WHITE/(FLINT GREY)                                        |           2 |           10 | bukan_warna | warna       | 53.58%         |
| 47519 | NIK     | 3PPK          | NIKE 3PPK D-F LIGHTWEIGHT QTR-WHITE/FLINT GREY                                           |           2 |            9 | bukan_warna | warna       | 53.58%         |
| 47555 | NIK     | 3PPK          | NIKE SB 3PPK CREW SOCKS-BLACK/WHITE                                                      |           3 |            7 | bukan_warna | warna       | 53.58%         |
| 47705 | NIK     | 3PPK          | U NSW 3PPK FOOTIE-CARBON HEATHER/BLACK                                                   |           3 |            7 | bukan_warna | warna       | 53.58%         |
|  9908 | ADI     | FUTUREPACER   | FUTUREPACER-SILVER MET.                                                                  |           1 |            3 | bukan_warna | warna       | 53.18%         |
|  3487 | ADI     | FUTUREPACER   | FUTUREPACER-ST PALE NUDE                                                                 |           1 |            4 | bukan_warna | warna       | 53.18%         |
| 19919 | CAO     | 150FL         | BGA-150FL-1ADR                                                                           |           2 |            3 | bukan_warna | warna       | 53.1%          |
|  4010 | ADI     | MANAZERO      | MANAZERO M-UTIBLK/CBLACK/FTWWHT                                                          |           1 |            5 | bukan_warna | warna       | 52.65%         |
| 36024 | NIK     | N110          | NIKE N110 D/MS/X-BLACK/DARK GREY-RED ORBIT-RUSH VIOLET                                   |           2 |           12 | bukan_warna | warna       | 52.47%         |
|   457 | ADI     | TC1P          | TT ID LIGH TC1P-SESOSL/DGREYH/BLACK                                                      |           4 |            7 | bukan_warna | warna       | 52.46%         |
| 24527 | NIK     | GENICCO       | NIKE GENICCO-LOYAL BLUE/WHITE-RCR BLUE-SNST                                              |           2 |            8 | bukan_warna | warna       | 52.37%         |
|  1478 | ADI     | LACELESS      | ULTRABOOST LACELESS-CORE BLACK                                                           |           2 |            4 | bukan_warna | warna       | 52.25%         |
| 12737 | ADI     | LACELESS      | SUPERSTAR LACELESS-CORE BLACK                                                            |           2 |            4 | bukan_warna | warna       | 52.25%         |
| 48060 | PTG     | VINKA         | VINKA BLACK-BLACK                                                                        |           1 |            3 | bukan_warna | warna       | 51.67%         |
| 32222 | NIK     | VAPORX        | JR VAPORX 12 CLUB GS IC-WOLF GREY/LT CRIMSONBLACK                                        |           2 |           10 | bukan_warna | warna       | 51.13%         |
| 24674 | NIK     | FUTSLIDE      | FUTSLIDE SLIP-RIVER ROCK/PORT WINE-WHITE-VOLT                                            |           1 |            8 | bukan_warna | warna       | 50.7%          |
| 24677 | NIK     | FUTSLIDE      | FUTSLIDE SLIP-BLACK/THUNDER BLUE-BLUE TINT-VOLT                                          |           1 |            8 | bukan_warna | warna       | 50.7%          |
|  5170 | ADI     | FEF           | FEF BACKPACK-RED/POWRED/BOGOLD                                                           |           1 |            5 | bukan_warna | warna       | 50.67%         |
|  5166 | ADI     | FEF           | FEF BOTTLE-RED/POWRED/BOGOLD                                                             |           1 |            5 | bukan_warna | warna       | 50.67%         |
| 20191 | CAO     | 5600MS        | DW-5600MS-1DR                                                                            |           2 |            3 | bukan_warna | warna       | 50.62%         |
|  7006 | ADI     | VIRAL2        | TUBULAR VIRAL2 W-CBROWN/CBROWN/FTWWHT                                                    |           2 |            6 | bukan_warna | warna       | 50.53%         |
| 18208 | BBC     | CLEAR         | BB CLEAR SKY L/S T-SHIRT-BLACK                                                           |           2 |            8 | bukan_warna | warna       | 50.35%         |
| 56037 | VAP     | 0.7L          | 0.7L ELEMENT-FIRE                                                                        |           1 |            3 | bukan_warna | warna       | 50.0%          |
| 56014 | VAP     | 0.7L          | 0.7L MOSSY OAK-INFINITY                                                                  |           1 |            5 | bukan_warna | warna       | 50.0%          |
|       |         |               | BOTTLE                                                                                   |             |              |             |             |                |
| 56008 | VAP     | 0.7L          | 0.7L ELEMENT-RED                                                                         |           1 |            3 | bukan_warna | warna       | 50.0%          |
| 56022 | VAP     | 0.7L          | 0.7L TWO TONE-TEAL                                                                       |           1 |            4 | bukan_warna | warna       | 50.0%          |
|   774 | ADI     | TRAPNK        | CF QTFLEX W-CBLACK/CBLACK/TRAPNK                                                         |           6 |            6 | warna       | bukan_warna | 48.93%         |
|  5204 | ADI     | TRAPNK        | MUFC A JSY-ICEPNK/TRAPNK/BLACK                                                           |           5 |            6 | warna       | bukan_warna | 48.93%         |
|  4054 | ADI     | TRAPNK        | ULTRABOOST X ALL TERRAIN-MYSRUB/CBLACK/TRAPNK                                            |           7 |            7 | warna       | bukan_warna | 48.93%         |
|  8054 | ADI     | TRAPNK        | PREDATOR TANGO 18.1 TR-CLEORA/CLEORA/TRAPNK                                              |           7 |            7 | warna       | bukan_warna | 48.93%         |
|  2268 | ADI     | MYSINK        | ENERGY CLOUD WTC M-MYSINK/MYSINK/CROYAL                                                  |           6 |            7 | warna       | bukan_warna | 48.24%         |
|  3183 | ADI     | MYSINK        | ADVANTAGE CL QT W-FTWWHT/FTWWHT/MYSINK                                                   |           7 |            7 | warna       | bukan_warna | 48.24%         |
|  2227 | ADI     | MYSINK        | DURAMO 8 M-MYSINK/BLUE/SYELLO                                                            |           4 |            6 | warna       | bukan_warna | 48.24%         |
|  2267 | ADI     | MYSINK        | ENERGY CLOUD WTC M-MYSINK/MYSINK/CROYAL                                                  |           5 |            7 | warna       | bukan_warna | 48.24%         |
|  4155 | ADI     | MYSINK        | PUREBOOST XPOSE-NOBINK/MYSINK/TACBLU                                                     |           4 |            5 | warna       | bukan_warna | 48.24%         |
| 16829 | ADI     | SESOYE        | FORTAPLAY AC I-CORBLU/SESOYE/CONAVY                                                      |           5 |            6 | warna       | bukan_warna | 46.29%         |
|  4065 | ADI     | SESOYE        | DURAMO 8 K-MYSINK/BLUE/SESOYE                                                            |           6 |            6 | warna       | bukan_warna | 46.29%         |
| 16521 | ADI     | RUNWHT        | STAN SMITH-RUNWHT/RUNWHI/FAIRWA                                                          |           3 |            5 | warna       | bukan_warna | 46.17%         |
| 21386 | HER     | BRBDSCHRY     | SEVENTEEN-BRBDSCHRY/BKCRSHTCH                                                            |           2 |            3 | warna       | bukan_warna | 45.77%         |
|  1017 | ADI     | NTGREY        | FLB_RUNNER W-NTGREY/NTGREY/REDNIT                                                        |           4 |            5 | warna       | bukan_warna | 44.82%         |
|  1018 | ADI     | REDNIT        | FLB_RUNNER W-NTGREY/NTGREY/REDNIT                                                        |           5 |            5 | warna       | bukan_warna | 43.67%         |
|  2001 | ADI     | NGTCAR        | PRO SPARK 2018-NGTCAR/FTWWHT/SESAME                                                      |           4 |            6 | warna       | bukan_warna | 42.73%         |
|  7175 | ADI     | HIRAQU        | EPP II-HIRAQU/BLACK                                                                      |           3 |            4 | warna       | bukan_warna | 42.26%         |
|   331 | ADI     | HIRAQU        | FORTARUN X CF K-MYSINK/HIRAQU/CROYAL                                                     |           6 |            7 | warna       | bukan_warna | 42.26%         |
|  1899 | ADI     | HIRAQU        | ALTASWIM I-HIRAQU/HIRAQU/MYSINK                                                          |           3 |            5 | warna       | bukan_warna | 42.26%         |
|   577 | ADI     | HIRAQU        | AEROBOUNCE 2 M-CBLACK/SILVMT/HIRAQU                                                      |           6 |            6 | warna       | bukan_warna | 42.26%         |
|  1900 | ADI     | HIRAQU        | ALTASWIM I-HIRAQU/HIRAQU/MYSINK                                                          |           4 |            5 | warna       | bukan_warna | 42.26%         |
|  5935 | ADI     | TRACAR        | PUREBOOST DPR-CBLACK/TRACAR/CLOWHI                                                       |           4 |            5 | warna       | bukan_warna | 41.94%         |
|  3992 | ADI     | TRACAR        | ALPHABOUNCE EM M-TRAOLI/TRACAR/GREONE                                                    |           5 |            6 | warna       | bukan_warna | 41.94%         |
|  6760 | ADI     | TRACAR        | X_PLR-CBROWN/FTWWHT/TRACAR                                                               |           4 |            4 | warna       | bukan_warna | 41.94%         |
|  1403 | ADI     | F17           | NMD_R1-GREY TWO F17                                                                      |           4 |            4 | warna       | bukan_warna | 41.01%         |
|  4275 | ADI     | MGREYH        | ALPHABOUNCE HPC AMS M-MGREYH/GREFOU/FTWWHT                                               |           5 |            7 | warna       | bukan_warna | 40.59%         |
| 16994 | ADI     | MGREYH        | ESS BIGLOGO TEE-MGREYH                                                                   |           4 |            4 | warna       | bukan_warna | 40.59%         |
|  3406 | ADI     | BROWN         | CONTINENTAL 80-CLEAR BROWN                                                               |           4 |            4 | warna       | bukan_warna | 39.65%         |
| 32645 | NIK     | BROWN         | NIKE FLYKNIT TRAINER-VELVET BROWN/NEUTRAL OLIVE-SAIL-BLACK                               |           5 |            9 | warna       | bukan_warna | 39.65%         |
| 35687 | NIK     | BROWN         | NIKE SB ZOOM JANOSKI CNVS RM-BLACK/WHITE-THUNDER GREY-GUM LIGHT BROWN                    |          13 |           13 | warna       | bukan_warna | 39.65%         |
| 45362 | NIK     | BROWN         | AIR FORCE 1 BOOT NN-BROWN KELP/SEQUOIA-MEDIUM OLIVE                                      |           6 |           10 | warna       | bukan_warna | 39.65%         |
| 35206 | NIK     | BROWN         | WMNS NIKE MD RUNNER 2 SE-PHANTOM/ATMOSPHERE GREY-GUM LIGHT BROWN                         |          12 |           12 | warna       | bukan_warna | 39.65%         |
| 46829 | NIK     | BROWN         | NIKE DUNK LOW SE-BLACK/VELVET BROWN-VELVET BROWN-SAIL                                    |           9 |           10 | warna       | bukan_warna | 39.65%         |
| 21604 | HER     | BROWN         | HERITAGE-DARK OLIVE/SADDLE BROWN                                                         |           5 |            5 | warna       | bukan_warna | 39.65%         |
| 36182 | NIK     | BROWN         | NIKE AIR HUARACHE RUN-CARGO KHAKI/VOLT-SEQUOIA-GUM DARK BROWN                            |          11 |           11 | warna       | bukan_warna | 39.65%         |
| 32639 | NIK     | BROWN         | NIKE FLYKNIT TRAINER-WHITE/WHITE-WHITE-GUM LIGHT BROWN                                   |           9 |            9 | warna       | bukan_warna | 39.65%         |
| 29739 | NIK     | BROWN         | SF AF1 MID-BLACK/BLACK-GUM LIGHT BROWN                                                   |           8 |            8 | warna       | bukan_warna | 39.65%         |
| 51324 | PUM     | BROWN         | LEADCAT FENTY FU-GOLDEN BROWN-SCARAB                                                     |           5 |            6 | warna       | bukan_warna | 39.65%         |
| 40776 | NIK     | BROWN         | AIR FORCE 1 PRM / CLOT-GAME ROYAL/WHITE-GUM LIGHT BROWN                                  |          11 |           11 | warna       | bukan_warna | 39.65%         |
| 24199 | NIK     | BROWN         | SOLARSOFT THONG 2-DARK BROWN/ORANGE                                                      |           5 |            6 | warna       | bukan_warna | 39.65%         |
|  5630 | ADI     | BROWN         | SUPERSTAR 80S W-CLEAR BROWN                                                              |           5 |            5 | warna       | bukan_warna | 39.65%         |
| 22789 | KIP     | BROWN         | FS73-WALNUT BROWN-90                                                                     |           3 |            4 | warna       | bukan_warna | 39.65%         |
| 22291 | KIP     | BROWN         | FLAT F5-BROWN 90CM                                                                       |           3 |            4 | warna       | bukan_warna | 39.65%         |
| 46828 | NIK     | BROWN         | NIKE DUNK LOW SE-BLACK/VELVET BROWN-VELVET BROWN-SAIL                                    |           7 |           10 | warna       | bukan_warna | 39.65%         |
| 42240 | NIK     | BROWN         | NIKE AIR MAX 90 NRG-SAIL/SHEEN-STRAW-MEDIUM BROWN                                        |          10 |           10 | warna       | bukan_warna | 39.65%         |
| 22176 | KIP     | BROWN         | WAXED W6-BROWN 100CM                                                                     |           3 |            4 | warna       | bukan_warna | 39.65%         |
| 46128 | NIK     | BROWN         | NIKE DUNK LOW RETRO PRM-BEACH/BAROQUE BROWN-CANVAS-SAIL                                  |           8 |           10 | warna       | bukan_warna | 39.65%         |
| 23757 | NIK     | BROWN         | NIKE AIR WOVEN-VELVET BROWN/TEAM GOLD-SAIL-ALE BROWN                                     |          10 |           10 | warna       | bukan_warna | 39.65%         |
| 51188 | PUM     | BROWN         | PUMA X TC BASKET POMPOM INFANT-RUSSET BROWN-BIRCH                                        |           8 |            9 | warna       | bukan_warna | 39.65%         |
| 26661 | NIK     | BROWN         | ZOOM STEFAN JANOSKI OG-BLACK/WHITE-GUM LIGHT BROWN                                       |           9 |            9 | warna       | bukan_warna | 39.65%         |
| 17938 | AND     | BROWN         | PREMIUM BRUSH-BLACK/BROWN                                                                |           4 |            4 | warna       | bukan_warna | 39.65%         |
| 55857 | STN     | BROWN         | TUPAC-BROWN                                                                              |           2 |            2 | warna       | bukan_warna | 39.65%         |
| 10071 | ADI     | BROWN         | CRAZY BYW III-LIGHT BROWN                                                                |           5 |            5 | warna       | bukan_warna | 39.65%         |
| 33469 | NIK     | BROWN         | BLAZER LOW LTHR-BLACK/SAIL-SAIL-GUM MED BROWN                                            |           9 |            9 | warna       | bukan_warna | 39.65%         |
| 31343 | NIK     | BROWN         | NIKE AIR PRECISION II-BLACK/BLACK-ANTHRACITE-GUM LIGHT BROWN                             |          10 |           10 | warna       | bukan_warna | 39.65%         |
| 38771 | NIK     | BROWN         | NIKE ZOOM WINFLO 6 SE-BLACK/WHITE-GUM LIGHT BROWN                                        |          10 |           10 | warna       | bukan_warna | 39.65%         |
| 30907 | NIK     | BROWN         | AIR FORCE 1 07 LV8 SUEDE-MUSHROOM/MUSHROOM-GUM MED BROWN-IVO                             |          11 |           12 | warna       | bukan_warna | 39.65%         |
| 29687 | NIK     | BROWN         | NIKE ZOOM ASSERSION-BLACK/BLACK-GUM LIGHT BROWN                                          |           8 |            8 | warna       | bukan_warna | 39.65%         |
| 13069 | ADI     | BROWN         | ULTRABOOST 2.0-LIGHT BROWN                                                               |           4 |            4 | warna       | bukan_warna | 39.65%         |
|  3701 | ADI     | CROYAL        | JUVE A JSY-BOGOLD/CROYAL                                                                 |           5 |            5 | warna       | bukan_warna | 39.06%         |
|  2801 | ADI     | CROYAL        | GAZELLE STITCH AND TURN-CROYAL/CROYAL/FTWWHT                                             |           6 |            7 | warna       | bukan_warna | 39.06%         |
| 21457 | HER     | ARROWWOOD     | BRITANNIA-ARROWWOOD                                                                      |           2 |            2 | warna       | bukan_warna | 37.38%         |
|  2704 | ADI     | ASHPEA        | ULTRABOOST W-ASHPEA/ASHPEA/ASHPEA                                                        |           3 |            5 | warna       | bukan_warna | 37.38%         |
|  4355 | ADI     | ICEPNK        | INIKI RUNNER W-ICEPNK/FTWWHT/GUM3                                                        |           4 |            6 | warna       | bukan_warna | 36.66%         |
|  4083 | ADI     | ICEPNK        | TUBULAR VIRAL2 W-ICEPNK/ICEPNK/FTWWHT                                                    |           4 |            6 | warna       | bukan_warna | 36.66%         |
|  4084 | ADI     | ICEPNK        | TUBULAR VIRAL2 W-ICEPNK/ICEPNK/FTWWHT                                                    |           5 |            6 | warna       | bukan_warna | 36.66%         |
| 16824 | ADI     | ICEPNK        | FORTAPLAY AC I-LEGINK/ICEPNK/ENEINK                                                      |           5 |            6 | warna       | bukan_warna | 36.66%         |
|  1126 | ADI     | GUM4          | SUPERSTAR W-CBLACK/CBLACK/GUM4                                                           |           5 |            5 | warna       | bukan_warna | 34.82%         |
| 33814 | NIK     | EXPZ07WHITE   | NIKE EXPZ07WHITE/BLACK                                                                   |           2 |            3 | warna       | bukan_warna | 34.37%         |
|   325 | ADI     | BRBLUE        | FORTAPLAY AC I-DKBLUE/VIVGRN/BRBLUE                                                      |           6 |            6 | warna       | bukan_warna | 34.32%         |
|  1759 | ADI     | BRBLUE        | X_PLR J-BRBLUE/CBLACK/FTWWHT                                                             |           3 |            5 | warna       | bukan_warna | 34.32%         |
|  7279 | ADI     | CWHITE        | NMD_R1-CWHITE/CWHITE/SESOYE                                                              |           3 |            4 | warna       | bukan_warna | 34.19%         |
|  2547 | ADI     | CWHITE        | TUBULAR DEFIANT W-CBLACK/CBLACK/CWHITE                                                   |           6 |            6 | warna       | bukan_warna | 34.19%         |
|  7119 | ADI     | CWHITE        | REAL MADRID FBL-CWHITE/GREONE/BLACK                                                      |           4 |            6 | warna       | bukan_warna | 34.19%         |
|  6898 | ADI     | CWHITE        | PW TENNIS HU PK-CWHITE/CBLACK/FTWWHT                                                     |           5 |            7 | warna       | bukan_warna | 34.19%         |
|  2639 | ADI     | CWHITE        | ULTRABOOST X-ASHPEA/ASHPEA/CWHITE                                                        |           5 |            5 | warna       | bukan_warna | 34.19%         |
|  4532 | ADI     | CWHITE        | TUBULAR SHADOW W-TRACAR/TRACAR/CWHITE                                                    |           6 |            6 | warna       | bukan_warna | 34.19%         |
|  1592 | ADI     | CWHITE        | STAN SMITH-CWHITE/CWHITE/CROYAL                                                          |           4 |            5 | warna       | bukan_warna | 34.19%         |
|  7241 | ADI     | CWHITE        | REAL 3S CAP-CWHITE/BLACK                                                                 |           4 |            5 | warna       | bukan_warna | 34.19%         |
|  6798 | ADI     | CWHITE        | TUBULAR DOOM SOCK PK W-TRAPUR/CBLACK/CWHITE                                              |           8 |            8 | warna       | bukan_warna | 34.19%         |
|  7278 | ADI     | CWHITE        | NMD_R1-CWHITE/CWHITE/SESOYE                                                              |           2 |            4 | warna       | bukan_warna | 34.19%         |
|  1334 | ADI     | SUBGRN        | EQT SUPPORT PK 2/3-CBLACK/GREONE/SUBGRN                                                  |           8 |            8 | warna       | bukan_warna | 33.45%         |
|  4501 | ADI     | SUBGRN        | EQT SUPPORT ADV-CBLACK/CBLACK/SUBGRN                                                     |           6 |            6 | warna       | bukan_warna | 33.45%         |
|  7748 | ADI     | LTPINK        | LITE RACER INF-REAPNK/LTPINK/FTWWHT                                                      |           5 |            6 | warna       | bukan_warna | 33.09%         |
|  2197 | ADI     | EASGRN        | EQT SUPPORT RF PK-FROGRN/CBLACK/EASGRN                                                   |           7 |            7 | warna       | bukan_warna | 32.5%          |
|  2055 | ADI     | TECINK        | CF RACER TR-DKBLUE/TECINK/HIRERE                                                         |           5 |            6 | warna       | bukan_warna | 32.44%         |
|   256 | ADI     | TECINK        | ULTRABOOST LACELESS W-TECINK/RAWGRE/CBLACK                                               |           4 |            6 | warna       | bukan_warna | 32.44%         |
|  1387 | ADI     | SBROWN        | PROPHERE-SBROWN/CBLACK/CBROWN                                                            |           2 |            4 | warna       | bukan_warna | 32.06%         |
|  4623 | ADI     | SBROWN        | NMD C2-SBROWN/SBROWN/CBLACK                                                              |           3 |            5 | warna       | bukan_warna | 32.06%         |
|  1798 | ADI     | CLPINK        | CAMPUS J-REAMAG/CLPINK/CLPINK                                                            |           5 |            5 | warna       | bukan_warna | 31.94%         |
| 46940 | NIK     | REACTBRIGHT   | NK REACTBRIGHT CRIMSON/DARK GREY/PURE PLATINUM                                           |           2 |            7 | warna       | bukan_warna | 31.82%         |
| 50514 | PUM     | PEACOAT       | DIVECAT-PEACOAT-WHITE                                                                    |           2 |            3 | warna       | bukan_warna | 29.83%         |
|  7170 | ADI     | SHOYEL        | STARLANCER V-SHOYEL/BLACK/WHITE                                                          |           3 |            5 | warna       | bukan_warna | 28.61%         |
|  7370 | ADI     | SHOYEL        | DEERUPT RUNNER-SHOPUR/REDNIT/SHOYEL                                                      |           5 |            5 | warna       | bukan_warna | 28.61%         |
|   969 | ADI     | SHOYEL        | EQT ADV 360 I-CBLACK/SHOYEL/FTWWHT                                                       |           6 |            7 | warna       | bukan_warna | 28.61%         |
|   518 | ADI     | SCARLE        | D ROSE LETHALITY-CBLACK/SCARLE/FTWWHT                                                    |           5 |            6 | warna       | bukan_warna | 28.57%         |
|   778 | ADI     | SCARLE        | RACER TR INF-SCARLE/CBLACK/FTWWHT                                                        |           4 |            6 | warna       | bukan_warna | 28.57%         |
| 51425 | PUM     | SCARLE        | TSUGI JUN ANR PEBBLE-OLIVE BRANCH-SCARLE                                                 |           7 |            7 | warna       | bukan_warna | 28.57%         |
|  5509 | ADI     | SCARLE        | DUAL THREAT 2017 J-SCARLE/CBLACK/FTWWHT                                                  |           5 |            7 | warna       | bukan_warna | 28.57%         |
|  4283 | ADI     | SCARLE        | CRAZY TEAM 2017-FTWWHT/CBLACK/SCARLE                                                     |           6 |            6 | warna       | bukan_warna | 28.57%         |
| 21687 | HER     | DARK          | FOURTEEN-DARK GRID/BLACK                                                                 |           2 |            4 | warna       | bukan_warna | 27.8%          |
| 21603 | HER     | DARK          | HERITAGE-DARK OLIVE/SADDLE BROWN                                                         |           2 |            5 | warna       | bukan_warna | 27.8%          |
|  2613 | ADI     | ENEBLU        | NEMEZIZ 17.1 FG-LEGINK/SYELLO/ENEBLU                                                     |           6 |            6 | warna       | bukan_warna | 27.69%         |
| 16925 | ADI     | ENEBLU        | X 17.4 IN J-FTWWHT/ENEBLU/CLEGRE                                                         |           6 |            7 | warna       | bukan_warna | 27.69%         |
|  1562 | ADI     | RAWGRE        | GAZELLE S&T-RAWGRE/RAWGRE/OWHITE                                                         |           3 |            5 | warna       | bukan_warna | 27.69%         |
|  6786 | ADI     | RAWGRE        | CAMPUS STITCH AND TURN-RAWGRE/RAWGRE/FTWWHT                                              |           5 |            7 | warna       | bukan_warna | 27.69%         |
|  7740 | ADI     | RAWGRE        | CF RACER TR-RAWSTE/CONAVY/RAWGRE                                                         |           6 |            6 | warna       | bukan_warna | 27.69%         |
|  5576 | ADI     | RAWGRE        | ALPHABOUNCE BEYOND W-RAWGRE/ORCTIN/LEGINK                                                |           4 |            6 | warna       | bukan_warna | 27.69%         |
|  1563 | ADI     | RAWGRE        | GAZELLE S&T-RAWGRE/RAWGRE/OWHITE                                                         |           4 |            5 | warna       | bukan_warna | 27.69%         |
| 46960 | NIK     | FTR10PURE     | NK FTR10PURE PLATINUM/BRIGHT CRIMSON/DARK GREY                                           |           2 |            7 | warna       | bukan_warna | 27.53%         |
|  3658 | ADI     | VIVTEA        | REAL MADRID FBL-WHITE/VIVTEA/SILVMT                                                      |           5 |            6 | warna       | bukan_warna | 26.08%         |
|  6870 | ADI     | BLUBIR        | DEERUPT RUNNER-SOLRED/SOLRED/BLUBIR                                                      |           5 |            5 | warna       | bukan_warna | 25.92%         |
|  8171 | ADI     | ASHSIL        | NEMEZIZ TANGO 18.3 IN-ASHSIL/ASHSIL/WHITIN                                               |           6 |            7 | warna       | bukan_warna | 24.08%         |
|  5892 | ADI     | ASHSIL        | ULTRABOOST LACELESS-ASHSIL/ASHSIL/CBLACK                                                 |           3 |            5 | warna       | bukan_warna | 24.08%         |
|  5873 | ADI     | ASHSIL        | ULTRABOOST ALL TERRAIN LTD-ASHSIL/CARBON/CBLACK                                          |           5 |            7 | warna       | bukan_warna | 24.08%         |
|   235 | ADI     | ASHSIL        | NEMEZIZ TANGO 18.1 TR-ASHSIL/ASHSIL/WHITIN                                               |           6 |            7 | warna       | bukan_warna | 24.08%         |
|  1605 | ADI     | GOLDMT        | STAN SMITH PREMIUM-CBLACK/CBLACK/GOLDMT                                                  |           6 |            6 | warna       | bukan_warna | 23.74%         |
|   681 | ADI     | GOLDMT        | SUPERSTAR 80S CLEAN-FTWWHT/FTWWHT/GOLDMT                                                 |           6 |            6 | warna       | bukan_warna | 23.74%         |
|  1270 | ADI     | SHOLIM        | PROPHERE-CBLACK/FTWWHT/SHOLIM                                                            |           4 |            4 | warna       | bukan_warna | 23.63%         |
|   540 | ADI     | REATEA        | SUPERSTAR WM-LGREYH/REATEA/FTWWHT                                                        |           4 |            5 | warna       | bukan_warna | 23.55%         |
|  7564 | ADI     | OWHITE        | ULTRABOOST UNCAGED-OWHITE/CHAPEA/VAPGRE                                                  |           3 |            5 | warna       | bukan_warna | 22.71%         |
|  1715 | ADI     | OWHITE        | DEERUPT W-CLEORA/CLEORA/OWHITE                                                           |           5 |            5 | warna       | bukan_warna | 22.71%         |
|  8187 | ADI     | OWHITE        | X 18+ FG-OWHITE/FTWWHT/OWHITE                                                            |           6 |            6 | warna       | bukan_warna | 22.71%         |
|  8185 | ADI     | OWHITE        | X 18+ FG-OWHITE/FTWWHT/OWHITE                                                            |           4 |            6 | warna       | bukan_warna | 22.71%         |
| 16626 | ADI     | OWHITE        | STAN SMITH CF W-CBLACK/CBLACK/OWHITE                                                     |           7 |            7 | warna       | bukan_warna | 22.71%         |
|  6830 | ADI     | OWHITE        | TUBULAR DAWN W-CBLACK/CBLACK/OWHITE                                                      |           6 |            6 | warna       | bukan_warna | 22.71%         |
|  8219 | ADI     | OWHITE        | X TANGO 18.1 TR-OWHITE/FTWWHT/OWHITE                                                     |           5 |            7 | warna       | bukan_warna | 22.71%         |
|  8313 | ADI     | OWHITE        | X 18.3 FG J-OWHITE/FTWWHT/OWHITE                                                         |           5 |            7 | warna       | bukan_warna | 22.71%         |
|  6845 | ADI     | OWHITE        | SUPERSTAR BW3S SLIPON W-CRYWHT/OWHITE/CBLACK                                             |           6 |            7 | warna       | bukan_warna | 22.71%         |
|  3088 | ADI     | OWHITE        | X 18.4 FXG J-CBLACK/OWHITE/ACTRED                                                        |           6 |            7 | warna       | bukan_warna | 22.71%         |
|  2569 | ADI     | OWHITE        | STAN SMITH W-OWHITE/OWHITE/STPANU                                                        |           4 |            6 | warna       | bukan_warna | 22.71%         |
|  2403 | ADI     | OWHITE        | ZX FLUX ADV VERVE W-MYSBLU/MYSBLU/OWHITE                                                 |           8 |            8 | warna       | bukan_warna | 22.71%         |
|  6772 | ADI     | OWHITE        | TUBULAR DOOM SOCK W-CBROWN/OWHITE/OWHITE                                                 |           7 |            7 | warna       | bukan_warna | 22.71%         |
|  6913 | ADI     | OWHITE        | SUPERSTAR 80S-OWHITE/CBLACK/OWHITE                                                       |           5 |            5 | warna       | bukan_warna | 22.71%         |
| 16063 | ADI     | PRIMEBLUE     | GEODIVER+ PRIMEBLUE-FTWR WHITE                                                           |           2 |            4 | warna       | bukan_warna | 22.63%         |
|  7623 | ADI     | ORCTIN        | ULTRABOOST UNCAGED W-AERGRN/ORCTIN/FTWWHT                                                |           5 |            6 | warna       | bukan_warna | 21.4%          |
|  1686 | ADI     | ORCTIN        | STAN SMITH W-ORCTIN/ORCTIN/OWHITE                                                        |           5 |            6 | warna       | bukan_warna | 21.4%          |
| 33831 | NIK     | EXPX14WHITE   | NIKE EXPX14WHITE/WOLF GREYBLACK                                                          |           2 |            4 | warna       | bukan_warna | 20.9%          |
| 20987 | HER     | 600D          | EIGHTEEN-600D POLY NAVY/RED                                                              |           2 |            5 | warna       | bukan_warna | 20.88%         |
| 21034 | HER     | 600D          | SIXTEEN-600D POLY W CAMO                                                                 |           2 |            5 | warna       | bukan_warna | 20.88%         |
| 21037 | HER     | 600D          | FIFTEEN-600D POLY BLACK                                                                  |           2 |            4 | warna       | bukan_warna | 20.88%         |
| 21042 | HER     | 600D          | FIFTEEN-600D POLY NAVY                                                                   |           2 |            4 | warna       | bukan_warna | 20.88%         |
| 17080 | ADI     | WHT           | ADILETTE PLAY I-BLACK1/BLACK1/WHT                                                        |           6 |            6 | warna       | bukan_warna | 20.68%         |
| 26015 | NIK     | WHT           | NIKE AIR PEGASUS 83-DP ROYAL/DP RYL-HYPR CBLT-WHT                                        |          11 |           11 | warna       | bukan_warna | 20.68%         |
| 14544 | ADI     | WHT           | PUMP - BLACK/WHT                                                                         |           3 |            3 | warna       | bukan_warna | 20.68%         |
|   319 | ADI     | CLEMIN        | FORTAPLAY AC I-CLELIL/FTWWHT/CLEMIN                                                      |           6 |            6 | warna       | bukan_warna | 20.49%         |
|   656 | ADI     | BGREEN        | NMD R1 STLT PK-CBLACK/NOBGRN/BGREEN                                                      |           7 |            7 | warna       | bukan_warna | 20.2%          |
|  5488 | ADI     | CBROWN        | SWIFT RUN W-CBROWN/FTWWHT/CRYWHT                                                         |           4 |            6 | warna       | bukan_warna | 19.14%         |
|  3421 | ADI     | CBROWN        | NEMEZIZ 18.1 TR-OWHITE/OWHITE/CBROWN                                                     |           6 |            6 | warna       | bukan_warna | 19.14%         |
|  2279 | ADI     | CBROWN        | PUREBOOST XPOSE-CRYWHT/SILVMT/CBROWN                                                     |           5 |            5 | warna       | bukan_warna | 19.14%         |
|  6770 | ADI     | CBROWN        | TUBULAR DOOM SOCK W-CBROWN/OWHITE/OWHITE                                                 |           5 |            7 | warna       | bukan_warna | 19.14%         |
| 56661 | WAR     | THE           | 125CM THE BLUES FLAT LACES                                                               |           2 |            5 | warna       | bukan_warna | 18.78%         |
|  2588 | ADI     | CONAVY        | GAZELLE-CONAVY/WHITE/GOLDMT                                                              |           2 |            4 | warna       | bukan_warna | 18.74%         |
| 12118 | ADI     | CONAVY        | VS ADVANTAGE CL-FTWWHT/FTWWHT/CONAVY                                                     |           6 |            6 | warna       | bukan_warna | 18.74%         |
|  9207 | ADI     | CONAVY        | GYMSACK TREFOIL-CONAVY/RAWSAN                                                            |           3 |            4 | warna       | bukan_warna | 18.74%         |
|  5265 | ADI     | CONAVY        | CLIMACOOL 02/17-CONAVY/MYSBLU/FTWWHT                                                     |           4 |            6 | warna       | bukan_warna | 18.74%         |
|   800 | ADI     | CONAVY        | ADILETTE SHOWER-CONAVY/FTWWHT/CONAVY                                                     |           3 |            5 | warna       | bukan_warna | 18.74%         |
|  9537 | ADI     | CONAVY        | TEE SS-CONAVY                                                                            |           3 |            3 | warna       | bukan_warna | 18.74%         |
|  1697 | ADI     | CONAVY        | RASCAL-CBLACK/SCARLE/CONAVY                                                              |           4 |            4 | warna       | bukan_warna | 18.74%         |
|  3555 | ADI     | CONAVY        | 6P 3S CAP COTTO-CONAVY/CONAVY/WHITE                                                      |           6 |            7 | warna       | bukan_warna | 18.74%         |
|  8701 | ADI     | CONAVY        | CLASSIC BP-CONAVY/CONAVY/WHITE                                                           |           4 |            5 | warna       | bukan_warna | 18.74%         |
|   836 | ADI     | CONAVY        | ADILETTE CF ULTRA-CONAVY/CONAVY/CONAVY                                                   |           5 |            6 | warna       | bukan_warna | 18.74%         |
| 11431 | ADI     | CONAVY        | SPORTIVE TRKPNT-CONAVY                                                                   |           3 |            3 | warna       | bukan_warna | 18.74%         |
|  2673 | ADI     | CONAVY        | PUREBOOST-CONAVY/TRABLU/TRABLU                                                           |           2 |            4 | warna       | bukan_warna | 18.74%         |
|  9534 | ADI     | CONAVY        | CREW-CONAVY                                                                              |           2 |            2 | warna       | bukan_warna | 18.74%         |
|  6095 | ADI     | CONAVY        | GALAXY 4 M-CONAVY/CONAVY/ASHBLU                                                          |           5 |            6 | warna       | bukan_warna | 18.74%         |
|  3218 | ADI     | CONAVY        | LITE RACER-CONAVY/CBLACK/SOLBLU                                                          |           3 |            5 | warna       | bukan_warna | 18.74%         |
|  9541 | ADI     | CONAVY        | WAISTBAG-CONAVY/TRIPUR                                                                   |           2 |            3 | warna       | bukan_warna | 18.74%         |
|  4254 | ADI     | CONAVY        | COURT FURY 2017-CONAVY/FTWWHT/CONAVY                                                     |           6 |            6 | warna       | bukan_warna | 18.74%         |
|  3965 | ADI     | CONAVY        | TIRO SB-BLUE/CONAVY/WHITE                                                                |           4 |            5 | warna       | bukan_warna | 18.74%         |
|   479 | ADI     | CCMELS        | AEROK TANK-CCMELS                                                                        |           3 |            3 | warna       | bukan_warna | 18.28%         |
| 40232 | NIK     | GLOW          | NIKE REACT ELEMENT 55 SE-EMBER GLOW/BLACK-LIGHT BONE-WHITE                               |           7 |           11 | warna       | bukan_warna | 17.97%         |
| 45302 | NIK     | GLOW          | NK MERC FADE - SP21-GREEN GLOW/AQUAMARINE/LIME GLOW                                      |           9 |            9 | warna       | bukan_warna | 17.97%         |
| 52652 | PUM     | GLOW          | SUEDE BLOC IVORY GLOW-PUMA BLACK                                                         |           4 |            6 | warna       | bukan_warna | 17.97%         |
| 35941 | NIK     | GLOW          | LEGEND 8 ACADEMY FG/MG-AQUAMARINE/WHITE-LIME GLOW                                        |           9 |            9 | warna       | bukan_warna | 17.97%         |
| 43160 | NIK     | GLOW          | WMNS NIKE DOWNSHIFTER 11-SUMMIT WHITE/WHITE-LIME ICE-VOLT GLOW                           |          11 |           11 | warna       | bukan_warna | 17.97%         |
| 27162 | NIK     | GLOW          | WMNS NIKE LUNARSTELOS-MEDIUM BLUE/BLACK-SUNSET GLOW-ALUMINUM                             |           8 |            9 | warna       | bukan_warna | 17.97%         |
| 28249 | NIK     | GLOW          | AS W NK RUN TOP SS-EMBER GLOW/EMBER GLOW/REFLECTIVE SILV                                 |          10 |           12 | warna       | bukan_warna | 17.97%         |
| 42406 | NIK     | GLOW          | VAPOR 14 ACADEMY FG/MG-DYNAMIC TURQ/LIME GLOW                                            |           9 |            9 | warna       | bukan_warna | 17.97%         |
| 36580 | NIK     | GLOW          | NIKE AIR ZOOM VAPOR X HC PRM-BLACK/WHITE-VOLT GLOW                                       |          11 |           11 | warna       | bukan_warna | 17.97%         |
| 10249 | ADI     | GLOW          | RUNFALCON-GLOW PINK                                                                      |           2 |            3 | warna       | bukan_warna | 17.97%         |
| 38091 | NIK     | GLOW          | WMNS NIKE REVOLUTION 5-WHITE/VIOLET SHOCK-GREEN GLOW                                     |           9 |            9 | warna       | bukan_warna | 17.97%         |
| 27450 | NIK     | GLOW          | NIKE DOWNSHIFTER 7-THUNDER BLUE/VOLT GLOW-OBSIDIAN-BLACK                                 |           7 |            9 | warna       | bukan_warna | 17.97%         |
| 28689 | NIK     | GLOW          | WMNS NIKE FLEX 2017 RN-WOLF GREY/SUNSET GLOW-COOL GREY                                   |           9 |           11 | warna       | bukan_warna | 17.97%         |
| 40901 | NIK     | GLOW          | AIR FORCE 1 07 RS-WHITE/BLACK-LIGHT BONE-EMBER GLOW                                      |          11 |           11 | warna       | bukan_warna | 17.97%         |
| 35213 | NIK     | GLOW          | AIR MAX 270 SE-BLACK/BLACK-LASER ORANGE-EMBER GLOW                                       |          10 |           10 | warna       | bukan_warna | 17.97%         |
| 25788 | NIK     | GLOW          | WMNS NIKE REVOLUTION 3-BLACK/LAVA GLOW-HOT PUNCH-COOL GREY                               |           7 |           11 | warna       | bukan_warna | 17.97%         |
| 12104 | ADI     | GLOW          | TEMPER RUN-CBLACK/CBLACK/GLOW                                                            |           5 |            5 | warna       | bukan_warna | 17.97%         |
|   790 | ADI     | SUPPNK        | RACER TR INF-SUPPNK/CBLACK/FTWWHT                                                        |           4 |            6 | warna       | bukan_warna | 17.73%         |
|  5607 | ADI     | SUPPNK        | LITE RACER INF-SUPPNK/FTWWHT/ICEBLU                                                      |           4 |            6 | warna       | bukan_warna | 17.73%         |
|   762 | ADI     | MSILVE        | CLOUDFOAM ILATION MID-FTWWHT/CBLACK/MSILVE                                               |           6 |            6 | warna       | bukan_warna | 17.38%         |
|  2027 | ADI     | MSILVE        | CF ADVANTAGE-CBLACK/FTWWHT/MSILVE                                                        |           5 |            5 | warna       | bukan_warna | 17.38%         |
| 21519 | HER     | PRPL          | CRUZ-PRPL VELVT                                                                          |           2 |            3 | warna       | bukan_warna | 17.27%         |
|  1673 | ADI     | SHOPNK        | N-5923 J-SHOPNK/FTWWHT/CBLACK                                                            |           4 |            6 | warna       | bukan_warna | 17.0%          |
|  2834 | ADI     | SHOPNK        | LITE RACER CLN I-TRABLU/SHOPNK/FTWWHT                                                    |           6 |            7 | warna       | bukan_warna | 17.0%          |
|  2884 | ADI     | LGSOGR        | EXPLOSIVE BOUNCE 2018-CBLACK/SILVMT/LGSOGR                                               |           6 |            6 | warna       | bukan_warna | 16.81%         |
|  2896 | ADI     | CLEORA        | SOBAKOV-CLEORA/CBLACK/CRYWHT                                                             |           2 |            4 | warna       | bukan_warna | 15.4%          |
|  8292 | ADI     | CLEORA        | PREDATOR TANGO 18.4 IN J-CLEORA/CBLACK/GOLDMT                                            |           6 |            8 | warna       | bukan_warna | 15.4%          |
|   725 | ADI     | CLEORA        | NMD_R1 STLT PK W-CLEORA/CLEORA/CLOWHI                                                    |           6 |            7 | warna       | bukan_warna | 15.4%          |
|  8091 | ADI     | CLEORA        | PREDATOR TANGO 18.3 IN-CLEORA/CLEORA/CLEORA                                              |           6 |            7 | warna       | bukan_warna | 15.4%          |
|  4659 | ADI     | BOAQUA        | CAMPUS-BOAQUA/FTWWHT/CWHITE                                                              |           2 |            4 | warna       | bukan_warna | 15.24%         |
|  6701 | ADI     | CRYWHT        | EQT CUSHION ADV-CROYAL/FTWWHT/CRYWHT                                                     |           6 |            6 | warna       | bukan_warna | 15.21%         |
|  6570 | ADI     | CRYWHT        | NMD R2 W-ASHPNK/CRYWHT/FTWWHT                                                            |           5 |            6 | warna       | bukan_warna | 15.21%         |
|  7677 | ADI     | CRYWHT        | EDGE LUX 2 W-FTWWHT/CRYWHT/CBLACK                                                        |           6 |            7 | warna       | bukan_warna | 15.21%         |
|  2066 | ADI     | CRYWHT        | SAMBA OG-FTWWHT/CGREEN/CRYWHT                                                            |           5 |            5 | warna       | bukan_warna | 15.21%         |
|  7052 | ADI     | CRYWHT        | SWIFT RUN SUMMER-CRYWHT/GREONE/WHITIN                                                    |           4 |            6 | warna       | bukan_warna | 15.21%         |
|  2827 | ADI     | CHACOR        | DURAMO 9-GRETWO/GRETWO/CHACOR                                                            |           5 |            5 | warna       | bukan_warna | 15.05%         |
| 15999 | ADI     | SPECTOO       | NMD_R1 SPECTOO-CORE BLACK                                                                |           2 |            4 | warna       | bukan_warna | 14.74%         |
|  3977 | ADI     | CLONIX        | MANA BOUNCE 2 M ARAMIS-FTWWHT/SILVMT/CLONIX                                              |           8 |            8 | warna       | bukan_warna | 14.34%         |
| 54951 | SAU     | TAN           | COURAGEOUS-TAN/PNK                                                                       |           2 |            3 | warna       | bukan_warna | 14.28%         |
| 16112 | ADI     | VAPOUR        | FLUIDSTREET-VAPOUR PINK                                                                  |           2 |            3 | warna       | bukan_warna | 14.01%         |
|  2168 | ADI     | DKBLUE        | LITE RACER CLN-DKBLUE/FTWWHT/DKBLUE                                                      |           6 |            6 | warna       | bukan_warna | 13.2%          |
|  1914 | ADI     | DKBLUE        | CF ADVANTAGE-DKBLUE/FTWWHT/CBLACK                                                        |           3 |            5 | warna       | bukan_warna | 13.2%          |
|  4318 | ADI     | DKBLUE        | PW TENNIS HU-DKBLUE/DKBLUE/FTWWHT                                                        |           4 |            6 | warna       | bukan_warna | 13.2%          |
|   190 | ADI     | DKBLUE        | MILANO 16 SOCK-DKBLUE/WHITE                                                              |           4 |            5 | warna       | bukan_warna | 13.2%          |
|  1749 | ADI     | DKBLUE        | DEERUPT RUNNER-DKBLUE/DKBLUE/ASHBLU                                                      |           4 |            5 | warna       | bukan_warna | 13.2%          |
| 22729 | KIP     | BEIGE         | FS68-BEIGE BRAID-140                                                                     |           2 |            4 | warna       | bukan_warna | 12.74%         |
| 29098 | NIK     | 8ASHEN        | NIKE DOWNSHIFTER 8ASHEN SLATE/OBSIDIANDIFFUSED BLUEBLACK                                 |           3 |            6 | warna       | bukan_warna | 12.62%         |
| 21982 | HER     | FLORAL        | HANSON-FLORAL BLR                                                                        |           2 |            3 | warna       | bukan_warna | 12.21%         |
| 11061 | ADI     | CORE          | OZWEEGO-CORE BLACK                                                                       |           2 |            3 | warna       | bukan_warna | 12.15%         |
| 11997 | ADI     | CORE          | RUNFALCON-CORE BLACK                                                                     |           2 |            3 | warna       | bukan_warna | 12.15%         |
| 13308 | ADI     | CORE          | X9000L4-CORE BLACK                                                                       |           2 |            3 | warna       | bukan_warna | 12.15%         |
| 10929 | ADI     | CORE          | ROGUERA-CORE BLACK                                                                       |           2 |            3 | warna       | bukan_warna | 12.15%         |
| 13299 | ADI     | CORE          | X9000L4-CORE BLACK                                                                       |           2 |            3 | warna       | bukan_warna | 12.15%         |
| 11694 | ADI     | CORE          | ULTRABOOST-CORE BLACK                                                                    |           2 |            3 | warna       | bukan_warna | 12.15%         |
| 12923 | ADI     | CORE          | NMD_R1.V2-CORE BLACK                                                                     |           2 |            3 | warna       | bukan_warna | 12.15%         |
| 12017 | ADI     | CORE          | ASWEERUN-CORE BLACK                                                                      |           2 |            3 | warna       | bukan_warna | 12.15%         |
| 14243 | ADI     | CORE          | QUESTARSTRIKE-CORE BLACK/CORE BLACK/BOLD GOLD                                            |           2 |            7 | warna       | bukan_warna | 12.15%         |
| 13380 | ADI     | CORE          | CORERACER-CORE BLACK                                                                     |           2 |            3 | warna       | bukan_warna | 12.15%         |
| 15946 | ADI     | CORE          | NMD_R1-CORE BLACK                                                                        |           2 |            3 | warna       | bukan_warna | 12.15%         |
| 10996 | ADI     | CORE          | SUPERSTAR-CORE BLACK                                                                     |           2 |            3 | warna       | bukan_warna | 12.15%         |
| 12025 | ADI     | CORE          | ASWEERUN-CORE BLACK                                                                      |           2 |            3 | warna       | bukan_warna | 12.15%         |
| 15445 | ADI     | CORE          | NMD_R1.V2-CORE BLACK                                                                     |           2 |            3 | warna       | bukan_warna | 12.15%         |
|  9791 | ADI     | CORE          | RISINGSTARXR1-CORE BLACK                                                                 |           2 |            3 | warna       | bukan_warna | 12.15%         |
| 12007 | ADI     | CORE          | RUNFALCON-CORE BLACK                                                                     |           2 |            3 | warna       | bukan_warna | 12.15%         |
| 10424 | ADI     | CORE          | RUNTHEGAME-CORE BLACK/CORE BLACK/ACTIVE RED                                              |           2 |            7 | warna       | bukan_warna | 12.15%         |
| 10320 | ADI     | CORE          | RUNTHEGAME-CORE BLACK/CORE BLACK/GREY SIX                                                |           2 |            7 | warna       | bukan_warna | 12.15%         |
| 11697 | ADI     | CORE          | ULTRABOOST-CORE BLACK                                                                    |           2 |            3 | warna       | bukan_warna | 12.15%         |
|  8479 | ADI     | CORE          | 3MC-CORE BLACK                                                                           |           2 |            3 | warna       | bukan_warna | 12.15%         |
| 10134 | ADI     | CORE          | X_PLR-CORE BLACK                                                                         |           2 |            3 | warna       | bukan_warna | 12.15%         |
|  6039 | ADI     | GREFOU        | DURAMO 8 W-GREFOU/GRETWO/GRETWO                                                          |           4 |            6 | warna       | bukan_warna | 11.85%         |
|  6884 | ADI     | GREFOU        | DEERUPT RUNNER-GRETHR/GREFOU/FTWWHT                                                      |           4 |            5 | warna       | bukan_warna | 11.85%         |
|  5151 | ADI     | GREFOU        | H90 LOGO CAP-BLACK/BLACK/GREFOU                                                          |           6 |            6 | warna       | bukan_warna | 11.85%         |
| 17275 | AGL     | 5             | ITALIC 5 PANEL MAROON 005-MAROON                                                         |           5 |            6 | warna       | bukan_warna | 11.46%         |
|  2592 | ADI     | ICEPUR        | GAZELLE-ICEPUR/WHITE/GOLDMT                                                              |           2 |            4 | warna       | bukan_warna | 11.32%         |
|  5248 | ADI     | CORBLU        | FORTAPLAY AC I-CORBLU/FTWWHT/EQTYEL                                                      |           4 |            6 | warna       | bukan_warna | 10.96%         |
| 22744 | KIP     | TAUPE         | FS69-TAUPE BROWN-160                                                                     |           2 |            4 | warna       | bukan_warna | 10.94%         |
|  6090 | ADI     | GREFIV        | GALAXY 4 M-GREFIV/GREFIV/GRETWO                                                          |           5 |            6 | warna       | bukan_warna | 10.86%         |
|  5289 | ADI     | GREFIV        | SUPERSTAR BW35 SLIPON W-GREFIV/GREFIV/GRETHR                                             |           5 |            7 | warna       | bukan_warna | 10.86%         |
|  3245 | ADI     | GREFIV        | CF SWIFT RACER-CBLACK/UTIBLK/GREFIV                                                      |           6 |            6 | warna       | bukan_warna | 10.86%         |
|  4248 | ADI     | GREFIV        | DUAL THREAT 2017-CBLACK/FTWWHT/GREFIV                                                    |           6 |            6 | warna       | bukan_warna | 10.86%         |
|  7272 | ADI     | GREFIV        | NMD_R1-CBLACK/GREFOU/GREFIV                                                              |           4 |            4 | warna       | bukan_warna | 10.86%         |
|  1911 | ADI     | GREFIV        | CF RACER TR-CBLACK/CBLACK/GREFIV                                                         |           6 |            6 | warna       | bukan_warna | 10.86%         |
|  4815 | ADI     | GREFIV        | SUPERSTAR 80S CLEAN-UTIBLK/UTIBLK/GREFIV                                                 |           6 |            6 | warna       | bukan_warna | 10.86%         |
|  3981 | ADI     | GREFIV        | ALPHABOUNCE LUX W-ENEAQU/GREFIV/FTWWHT                                                   |           5 |            6 | warna       | bukan_warna | 10.86%         |
|  4385 | ADI     | GREFIV        | X PLR-GREFIV/GREFIV/FTWWHT                                                               |           4 |            5 | warna       | bukan_warna | 10.86%         |
|  4237 | ADI     | GREFIV        | EXPLOSIVE BOUNCE-GREFOU/SILVMT/GREFIV                                                    |           5 |            5 | warna       | bukan_warna | 10.86%         |
|  3715 | ADI     | GREFIV        | ADIDAS EQT SOCK-BLACK/DGSOGR/GREFIV                                                      |           6 |            6 | warna       | bukan_warna | 10.86%         |
| 56746 | WAR     | PAISLEY       | 125CM PAISLEY WHITE FLAT                                                                 |           2 |            4 | warna       | bukan_warna | 10.4%          |
|   873 | ADI     | VAPGRE        | CLOUDFOAM RACE W-VAPGRE/VAGRME/FTWWHT                                                    |           4 |            6 | warna       | bukan_warna | 10.29%         |
|  7565 | ADI     | VAPGRE        | ULTRABOOST UNCAGED-OWHITE/CHAPEA/VAPGRE                                                  |           5 |            5 | warna       | bukan_warna | 10.29%         |
|  7870 | ADI     | VAPGRE        | CF QT RACER W-ICEPUR/VAGRME/VAPGRE                                                       |           7 |            7 | warna       | bukan_warna | 10.29%         |
|  6943 | ADI     | HIRERE        | SWIFT RUN PK-HIRERE/FTWWHT/CBLACK                                                        |           4 |            6 | warna       | bukan_warna | 10.23%         |
|  6086 | ADI     | HIRERE        | GALAXY 4 M-CARBON/CARBON/HIRERE                                                          |           6 |            6 | warna       | bukan_warna | 10.23%         |
|  4879 | ADI     | HIRERE        | PERF BOTTL 0,75-HIRERE/CARBON/CARBON                                                     |           4 |            6 | warna       | bukan_warna | 10.23%         |
|  3535 | ADI     | POWRED        | SQUAD 17 SHO-POWRED/WHITE                                                                |           4 |            5 | warna       | bukan_warna | 10.18%         |
|  7116 | ADI     | POWRED        | MUFC FBL-REARED/BLACK/POWRED                                                             |           5 |            5 | warna       | bukan_warna | 10.18%         |
|   399 | ADI     | POWRED        | CFC H JSY-CHEBLU/WHITE/POWRED                                                            |           6 |            6 | warna       | bukan_warna | 10.18%         |
| 12005 | ADI     | RAW           | RUNFALCON-RAW INDIGO                                                                     |           2 |            3 | warna       | bukan_warna | 10.17%         |
|    12 | ADI     | BASKETBALL    | 3 STRIPE D 29.5-BASKETBALL NATURAL                                                       |           5 |            6 | warna       | bukan_warna | 9.97%          |
|  1407 | ADI     | CARGO         | NMD_TS1 PK-NIGHT CARGO                                                                   |           4 |            4 | warna       | bukan_warna | 9.55%          |
|  1810 | ADI     | REAMAG        | CAMPUS C-REAMAG/CLPINK/CLPINK                                                            |           3 |            5 | warna       | bukan_warna | 9.48%          |
|  1796 | ADI     | REAMAG        | CAMPUS J-REAMAG/CLPINK/CLPINK                                                            |           3 |            5 | warna       | bukan_warna | 9.48%          |
|  7372 | ADI     | SGREEN        | PROPHERE-SGREEN/CGREEN/CBLACK                                                            |           2 |            4 | warna       | bukan_warna | 9.32%          |
| 56715 | WAR     | MINT          | 90CM MINT-BLACK ROPE LACES                                                               |           2 |            5 | warna       | bukan_warna | 8.9%           |
| 10576 | ADI     | SAND          | NMD_R1-SAND                                                                              |           2 |            2 | warna       | bukan_warna | 8.78%          |
| 15571 | ADI     | SAND          | OZRAH-SAND                                                                               |           2 |            2 | warna       | bukan_warna | 8.78%          |
| 21085 | HER     | POLY          | HERITAGE-POLY PERF BLACK/BLK                                                             |           2 |            5 | warna       | bukan_warna | 8.66%          |
| 21106 | HER     | POLY          | CLASSIC-POLY RAVEN X                                                                     |           2 |            4 | warna       | bukan_warna | 8.66%          |
|  3906 | ADI     | REARED        | MUFC H JSY-REARED/WHITE/BLACK                                                            |           4 |            6 | warna       | bukan_warna | 8.44%          |
|  3866 | ADI     | REARED        | MUFC 3S CAP-REARED/WHITE                                                                 |           4 |            5 | warna       | bukan_warna | 8.44%          |
|  3677 | ADI     | REARED        | MUFC 3S TRK TOP-REARED/WHITE/BLACK                                                       |           5 |            7 | warna       | bukan_warna | 8.44%          |
| 15766 | ADI     | CHALK         | OZWEEGO-CHALK WHITE                                                                      |           2 |            3 | warna       | bukan_warna | 8.35%          |
| 23752 | NIK     | NAVY          | WMNS AIR MAX 95-MIDNIGHT NAVY/LASER ORANGE-PURE PLATINUM                                 |           6 |           10 | warna       | bukan_warna | 8.11%          |
| 46230 | NIK     | NAVY          | NK HERITAGE HIP PACK - TRL-MIDNIGHT NAVY/OBSIDIAN/BLACK                                  |           7 |            9 | warna       | bukan_warna | 8.11%          |
| 20988 | HER     | NAVY          | EIGHTEEN-600D POLY NAVY/RED                                                              |           4 |            5 | warna       | bukan_warna | 8.11%          |
| 23451 | NIC     | NAVY          | NIKE CHALLENGER WAIST PACK LARGE-DARK TEAL GREEN/MIDNIGHT NAVY/PURE PLATINUM/SILVER OSFM |          10 |           14 | warna       | bukan_warna | 8.11%          |
| 36627 | NIK     | NAVY          | NIKE M2K TEKNO-MIDNIGHT NAVY/UNIVERSITY GOLD-BORDEAUX                                    |           5 |            8 | warna       | bukan_warna | 8.11%          |
| 46057 | NIK     | NAVY          | AIR FORCE 1 MID QS-WHITE/WHITE-MIDNIGHT NAVY-GUM YELLOW                                  |           9 |           11 | warna       | bukan_warna | 8.11%          |
| 48138 | PTG     | NAVY          | VERMI NAVY-NAVY                                                                          |           2 |            3 | warna       | bukan_warna | 8.11%          |
| 34540 | NIK     | NAVY          | HYPERDUNK X EP-MIDNIGHT NAVY/UNIVERSITY RED-WHITE                                        |           5 |            8 | warna       | bukan_warna | 8.11%          |
| 18808 | BBC     | NAVY          | SMALL ARCH LOGO SWEATPANTS-NAVY                                                          |           5 |            5 | warna       | bukan_warna | 8.11%          |
| 29512 | NIK     | NAVY          | AS M NK SHORT HBR-MIDNIGHT NAVY/MIDNIGHT NAVY/HYPER COBALT                               |           9 |           11 | warna       | bukan_warna | 8.11%          |
| 15225 | ADI     | NAVY          | RIB DETAIL SS T-CREW NAVY                                                                |           6 |            6 | warna       | bukan_warna | 8.11%          |
|  9222 | ADI     | NAVY          | CLASSIC BP-COLLEGIATE NAVY                                                               |           4 |            4 | warna       | bukan_warna | 8.11%          |
| 23381 | NIC     | NAVY          | NIKE WOMENS GYM ESSENTIAL FITNESS GLOVES-GAME ROYAL/MIDNIGHT NAVY/WHITE                  |          10 |           11 | warna       | bukan_warna | 8.11%          |
| 33002 | NIK     | NAVY          | PSG M NK BRT STAD JSY SS HM-MIDNIGHT NAVY/WHITE                                          |          10 |           11 | warna       | bukan_warna | 8.11%          |
| 11354 | ADI     | NAVY          | FEF H SHO- COLLEGIATE NAVY                                                               |           5 |            5 | warna       | bukan_warna | 8.11%          |
| 38254 | NIK     | NAVY          | AIR FORCE 1  07 LV8 1-MIDNIGHT NAVY/WHITE-BLACK-WHITE                                    |           8 |           11 | warna       | bukan_warna | 8.11%          |
| 21092 | HER     | NAVY          | POP QUIZ-600D POLY NAVY/ZIP                                                              |           5 |            6 | warna       | bukan_warna | 8.11%          |
| 34903 | NIK     | NAVY          | NIKE AIR MAX 97 SE-MIDNIGHT NAVY/LASER ORANGE-OBSIDIAN MIST                              |           7 |           11 | warna       | bukan_warna | 8.11%          |
| 21032 | HER     | NAVY          | HER-SIXTEEN-NAVY-(5L)-BAG-US                                                             |           3 |            6 | warna       | bukan_warna | 8.11%          |
| 10083 | ADI     | NAVY          | SUPERCOURT-COLLEGIATE NAVY                                                               |           3 |            3 | warna       | bukan_warna | 8.11%          |
| 21043 | HER     | NAVY          | FIFTEEN-600D POLY NAVY                                                                   |           4 |            4 | warna       | bukan_warna | 8.11%          |
| 40860 | NIK     | NAVY          | B NSW CORE AMPLIFY FZ-MIDNIGHT NAVY/WHITE/MIDNIGHT NAVY                                  |          10 |           10 | warna       | bukan_warna | 8.11%          |
| 23016 | KIP     | NAVY          | XS35-WHITE NAVY-140                                                                      |           3 |            4 | warna       | bukan_warna | 8.11%          |
| 25404 | NIK     | NAVY          | AS M NSW JGGR WVN V442-ARMORY NAVY/BLACK                                                 |           8 |            9 | warna       | bukan_warna | 8.11%          |
| 15727 | ADI     | NAVY          | ULTRABOOST 22 W-COLLEGIATE NAVY                                                          |           5 |            5 | warna       | bukan_warna | 8.11%          |
| 14254 | ADI     | NAVY          | PRO SPARK 2018-COLLEGIATE NAVY/SILVER MET./FTWR WHITE                                    |           5 |            9 | warna       | bukan_warna | 8.11%          |
| 21095 | HER     | NAVY          | LIL AMER M-600D POLY NAVY BLOCK                                                          |           6 |            7 | warna       | bukan_warna | 8.11%          |
| 19482 | BBC     | NAVY          | DAMAGED CREWNECK SWEATSHIRT-NAVY                                                         |           4 |            4 | warna       | bukan_warna | 8.11%          |
| 15190 | ADI     | NAVY          | SURREAL SUMMER-CREW NAVY                                                                 |           4 |            4 | warna       | bukan_warna | 8.11%          |
| 12683 | ADI     | NAVY          | NMD_R1-COLLEGIATE NAVY                                                                   |           3 |            3 | warna       | bukan_warna | 8.11%          |
| 54866 | SAU     | NAVY          | SHADOW ORIGINAL-NAVY/BROWN                                                               |           3 |            4 | warna       | bukan_warna | 8.11%          |
| 44628 | NIK     | NAVY          | NIKE DUNK HIGH (GS)-WHITE/MIDNIGHT NAVY-TOTAL ORANGE                                     |           7 |            9 | warna       | bukan_warna | 8.11%          |
|  9282 | ADI     | NAVY          | ADIDAS 3S BP-COLLEGIATE NAVY/ACTIVE RED/WHITE                                            |           5 |            8 | warna       | bukan_warna | 8.11%          |
| 33862 | NIK     | NAVY          | NIKE JOYRIDE CC-MIDNIGHT NAVY/DARK OBSIDIAN-DARK SULFUR                                  |           5 |            9 | warna       | bukan_warna | 8.11%          |
| 27017 | NIK     | NAVY          | NIKE SB CHECK SOLAR-MIDNIGHT NAVY/WHITE                                                  |           6 |            7 | warna       | bukan_warna | 8.11%          |
|  9750 | ADI     | NAVY          | AC CLASS BP-COLLEGIATE NAVY                                                              |           5 |            5 | warna       | bukan_warna | 8.11%          |
| 13531 | ADI     | NAVY          | ULTRABOOST 20 CITY PACK HYPE-COLLEGIATE NAVY                                             |           7 |            7 | warna       | bukan_warna | 8.11%          |
| 14748 | ADI     | NAVY          | B TF AB POLY TT-COLLEGIATE NAVY                                                          |           7 |            7 | warna       | bukan_warna | 8.11%          |
|  9134 | ADI     | NAVY          | SB CLASSIC TRE-COLLEGIATE NAVY                                                           |           5 |            5 | warna       | bukan_warna | 8.11%          |
| 14005 | ADI     | NAVY          | RESPONSE SR-CREW NAVY                                                                    |           4 |            4 | warna       | bukan_warna | 8.11%          |
| 11491 | ADI     | NAVY          | FORTAFAITO EL K-COLLEGIATE NAVY                                                          |           5 |            5 | warna       | bukan_warna | 8.11%          |
| 44843 | NIK     | NAVY          | PHANTOM GT2 ACADEMY DF FG/MG-COLLEGE NAVY/WHITE-VIVID PURPLE                             |           8 |           11 | warna       | bukan_warna | 8.11%          |
| 42214 | NIK     | NAVY          | AIR MAX TAILWIND V SP-MIDNIGHT NAVY/UNIVERSITY RED                                       |           7 |            9 | warna       | bukan_warna | 8.11%          |
| 21051 | HER     | NAVY          | LAKE SM-SEAMLESS NAVY                                                                    |           4 |            4 | warna       | bukan_warna | 8.11%          |
| 30205 | NIK     | NAVY          | DOWNSHIFTER 8 (TDV)-MIDNIGHT NAVY/FLASH CRIMSON-OIL GREY                                 |           5 |            9 | warna       | bukan_warna | 8.11%          |
| 45351 | NIK     | NAVY          | NK ELMNTL BKPK ? HBR-MIDNIGHT NAVY/MIDNIGHT NAVY/POLLEN                                  |           9 |           10 | warna       | bukan_warna | 8.11%          |
| 27364 | NIK     | NAVY          | NIKE AIR VERSITILE-MIDNIGHT NAVY/WHITE-GAME ROYAL                                        |           5 |            8 | warna       | bukan_warna | 8.11%          |
| 55064 | SOC     | NAVY          | TRICOLORE LOW-NAVY/RED                                                                   |           3 |            4 | warna       | bukan_warna | 8.11%          |
| 48114 | PTG     | NAVY          | SOLID 1 (PACK OF THREE)-GREY/WHITE/NAVY                                                  |           8 |            8 | warna       | bukan_warna | 8.11%          |
| 13114 | ADI     | NAVY          | ULTRABOOST 2.0-COLLEGIATE NAVY                                                           |           4 |            4 | warna       | bukan_warna | 8.11%          |
| 36821 | NIK     | NAVY          | NK ALPHA GMSK-MIDNIGHT NAVY/BLACK/WHITE                                                  |           5 |            7 | warna       | bukan_warna | 8.11%          |
| 42835 | NIK     | NAVY          | PSG MNK DF STAD JSY SS HM-MIDNIGHT NAVY/UNIVERSITY RED/WHITE                             |           9 |           12 | warna       | bukan_warna | 8.11%          |
| 45615 | NIK     | NAVY          | NIKE DUNK HIGH (PS)-WHITE/MIDNIGHT NAVY-TOTAL ORANGE                                     |           7 |            9 | warna       | bukan_warna | 8.11%          |
| 30182 | NIK     | NAVY          | DOWNSHIFTER 8 (GS)-MIDNIGHT NAVY/FLASH CRIMSON-OIL GREY                                  |           5 |            9 | warna       | bukan_warna | 8.11%          |
| 32754 | NIK     | NAVY          | AS M NSW AF1 HOODIE FZ FT-ARMORY NAVY/ARMORY NAVY/BLACK                                  |           9 |           12 | warna       | bukan_warna | 8.11%          |
| 43926 | NIK     | NAVY          | NIKE DUNK HI SP-VARSITY MAIZE/MIDNIGHT NAVY                                              |           8 |            8 | warna       | bukan_warna | 8.11%          |
| 42234 | NIK     | NAVY          | NIKE DBREAK-TYPE SE-MIDNIGHT NAVY/WHITE-TEAM ORANGE-BLUSTERY                             |           6 |           10 | warna       | bukan_warna | 8.11%          |
| 26752 | NIK     | PEELORANGE    | WMNS KAWA SLIDEPINK PRIME/ORANGE PEELORANGE PEEL                                         |           6 |            7 | warna       | bukan_warna | 8.1%           |
| 55804 | STN     | VOLT          | RAILWAY-VOLT                                                                             |           2 |            2 | warna       | bukan_warna | 8.05%          |
| 23355 | NIC     | 7             | NIKE KD FULL COURT 8P-AMBER/BLACK/METALLIC SILVER/BLACK 07                               |          11 |           11 | warna       | bukan_warna | 7.83%          |
|  1405 | ADI     | PK            | NMD_TS1 PK-NIGHT CARGO                                                                   |           2 |            4 | warna       | bukan_warna | 7.75%          |
|  3490 | ADI     | SHOCK         | FUTUREPACER-SHOCK RED                                                                    |           2 |            3 | warna       | bukan_warna | 7.61%          |
|  3844 | ADI     | PETNIT        | 3S PER SHOEBAG-PETNIT/PETNIT/TRAOLI                                                      |           5 |            6 | warna       | bukan_warna | 7.57%          |
|  3843 | ADI     | PETNIT        | 3S PER SHOEBAG-PETNIT/PETNIT/TRAOLI                                                      |           4 |            6 | warna       | bukan_warna | 7.57%          |
|  3774 | ADI     | PETNIT        | REAL 3S TEE-WHITE/PETNIT                                                                 |           5 |            5 | warna       | bukan_warna | 7.57%          |
| 21620 | HER     | PINE          | SEVENTEEN-PINE BARK/BLACK                                                                |           2 |            4 | warna       | bukan_warna | 6.69%          |
|  3783 | ADI     | EQTGRN        | DFB A JSY-EQTGRN/WHITE/REATEA                                                            |           4 |            6 | warna       | bukan_warna | 6.68%          |
|  5099 | ADI     | EQTGRN        | DFB 3S TEE-BLACK/EQTGRN                                                                  |           5 |            5 | warna       | bukan_warna | 6.68%          |
| 56444 | WAR     | OREO          | 90CM OREO ROPE                                                                           |           2 |            3 | warna       | bukan_warna | 5.74%          |
| 21469 | HER     | IVY           | IONA-IVY GREEN/SMOKED PEARL                                                              |           2 |            5 | warna       | bukan_warna | 5.67%          |
| 24523 | NIK     | ANTHRACITE    | NIKE GENICCO-ANTHRACITE/WHITE-BLACK                                                      |           3 |            5 | warna       | bukan_warna | 5.65%          |
| 31116 | NIK     | ANTHRACITE    | NIKE VIALE-ANTHRACITE/WHITE-INFRARED 23                                                  |           3 |            6 | warna       | bukan_warna | 5.65%          |
| 54953 | SAU     | BRN           | COURAGEOUS-BRN/YEL                                                                       |           2 |            3 | warna       | bukan_warna | 5.43%          |
| 22804 | KIP     | POWDER        | FS74-POWDER WHITE-90                                                                     |           2 |            4 | warna       | bukan_warna | 5.1%           |
| 55759 | STN     | RASTA         | VIARTA-RASTA                                                                             |           2 |            2 | warna       | bukan_warna | 5.01%          |
|  5879 | ADI     | NOBMAR        | ULTRABOOST ALL TERRAIN-NGTRED/NOBMAR/BRBLUE                                              |           5 |            6 | warna       | bukan_warna | 4.71%          |
| 13918 | ADI     | CARDBOARD     | NMD_R1.V2-CARDBOARD                                                                      |           2 |            2 | warna       | bukan_warna | 4.66%          |
| 54661 | REL     | 35            | REGULAR LACES-WHITE ROPE 35                                                              |           5 |            5 | warna       | bukan_warna | 4.4%           |
| 54596 | REL     | 35            | UB&NMD-SALMON PINK FLAT 35                                                               |           5 |            5 | warna       | bukan_warna | 4.4%           |
| 54612 | REL     | 35            | REFLECTIVE LACES-WHITE STRIPES 3M FLAT 35                                                |           7 |            7 | warna       | bukan_warna | 4.4%           |
| 54724 | REL     | 35            | UB&NMD-LIGHTBLUE/WHITE ROPE 35/89cm                                                      |           5 |            6 | warna       | bukan_warna | 4.4%           |
| 54636 | REL     | 35            | REGULAR LACES-BLACK/WHITE FLAT 35                                                        |           6 |            6 | warna       | bukan_warna | 4.4%           |
| 50395 | PUM     | PUMA          | RESOLVE PUMA BLACK-PUMA SILVER                                                           |           2 |            5 | warna       | bukan_warna | 4.21%          |
| 55911 | STN     | MELANGE       | BULLS MELANGE-RED                                                                        |           2 |            3 | warna       | bukan_warna | 4.09%          |
| 15466 | ADI     | SAVANNAH      | OZELIA-SAVANNAH                                                                          |           2 |            2 | warna       | bukan_warna | 4.09%          |
| 22760 | KIP     | ARMOR         | FS71-ARMOR GREY-90                                                                       |           2 |            4 | warna       | bukan_warna | 3.96%          |
| 21242 | HER     | WINE          | SEVENTEEN-WINE GRID                                                                      |           2 |            3 | warna       | bukan_warna | 3.68%          |
| 22828 | KIP     | VANILLA       | FS75-VANILLA WHITE-140                                                                   |           2 |            4 | warna       | bukan_warna | 3.61%          |
| 22824 | KIP     | VANILLA       | FS75-VANILLA WHITE-115                                                                   |           2 |            4 | warna       | bukan_warna | 3.61%          |
|  7274 | ADI     | SESAME        | NMD_R1-SESAME/TRACAR/BASGRN                                                              |           2 |            4 | warna       | bukan_warna | 3.41%          |
|  4222 | ADI     | SESAME        | TUBULAR DOOM SOCK PK-SESAME/SESAME/CRYWHT                                                |           5 |            7 | warna       | bukan_warna | 3.41%          |
|  6532 | ADI     | SESAME        | TUBULAR DOOM SOCK PK-BASGRN/SESAME/CWHITE                                                |           6 |            7 | warna       | bukan_warna | 3.41%          |
|  1656 | ADI     | ONIX          | MANA BOUNCE 2 M ARAMIS-CBLACK/SILVMT/ONIX                                                |           8 |            8 | warna       | bukan_warna | 2.87%          |
|  8858 | ADI     | ONIX          | REAL PRESHI-TECH ONIX                                                                    |           4 |            4 | warna       | bukan_warna | 2.87%          |
|  2910 | ADI     | ONIX          | DURAMO 9-CARBON/ONIX/GRETWO                                                              |           4 |            5 | warna       | bukan_warna | 2.87%          |
|  8894 | ADI     | ONIX          | MUFC BP-CLEAR GREY/CLEAR ONIX/BLAZE RED                                                  |           6 |            8 | warna       | bukan_warna | 2.87%          |
|  1664 | ADI     | ONIX          | MANA BOUNCE 2 W ARAMIS-CBLACK/SILVMT/ONIX                                                |           8 |            8 | warna       | bukan_warna | 2.87%          |
| 55259 | STN     | AQUA          | FAMILY FORCE-AQUA                                                                        |           3 |            3 | warna       | bukan_warna | 2.84%          |
| 21264 | HER     | FROG          | SIXTEEN-FROG CAMO                                                                        |           2 |            3 | warna       | bukan_warna | 2.64%          |
| 35787 | NIK     | VARSITY       | NIKE VARSITY COMPETE TR 2-BLACK/GHOST GREEN-SMOKE GREY                                   |           2 |           10 | warna       | bukan_warna | 2.61%          |
| 31261 | NIK     | VARSITY       | NIKE VARSITY COMPETE TRAINER-BLACK/WHITE                                                 |           2 |            6 | warna       | bukan_warna | 2.61%          |
|   263 | ADI     | BLUSPI        | ULTRABOOST W-TECINK/CARBON/BLUSPI                                                        |           5 |            5 | warna       | bukan_warna | 2.56%          |
|  5908 | ADI     | BLUSPI        | ULTRABOOST LACELESS-RAWGRE/CARBON/BLUSPI                                                 |           5 |            5 | warna       | bukan_warna | 2.56%          |
|  3316 | ADI     | BLUSPI        | ULTRABOOST-FTWWHT/CARBON/BLUSPI                                                          |           4 |            4 | warna       | bukan_warna | 2.56%          |
|  5285 | ADI     | BLUSPI        | ULTRABOOST PARLEY-CARBON/CARBON/BLUSPI                                                   |           5 |            5 | warna       | bukan_warna | 2.56%          |
|  6865 | ADI     | BLUSPI        | DEERUPT RUNNER PARLEY-CBLACK/CBLACK/BLUSPI                                               |           6 |            6 | warna       | bukan_warna | 2.56%          |
| 14727 | ADI     | LEGEND        | SHOPPER-LEGEND INK                                                                       |           2 |            3 | warna       | bukan_warna | 2.54%          |
| 12023 | ADI     | LEGEND        | ASWEERUN-LEGEND INK                                                                      |           2 |            3 | warna       | bukan_warna | 2.54%          |
| 22756 | KIP     | METRO         | FS70-METRO GREY-160                                                                      |           2 |            4 | warna       | bukan_warna | 2.3%           |
| 22752 | KIP     | METRO         | FS70-METRO GREY-140                                                                      |           2 |            4 | warna       | bukan_warna | 2.3%           |
| 15761 | ADI     | ALUMINA       | FLUIDFLOW 2.0-ALUMINA                                                                    |           3 |            3 | warna       | bukan_warna | 2.21%          |
|  8759 | ADI     | ACTIVE        | X LESTO-ACTIVE RED/BLACK/OFF WHITE                                                       |           3 |            7 | warna       | bukan_warna | 2.2%           |
| 11545 | ADI     | ACTIVE        | DURAMO 9-ACTIVE RED                                                                      |           3 |            4 | warna       | bukan_warna | 2.2%           |
| 10328 | ADI     | ACTIVE        | RUN60S-ACTIVE MAROON                                                                     |           2 |            3 | warna       | bukan_warna | 2.2%           |
| 55981 | STN     | OATMEAL       | XYZ-OATMEAL                                                                              |           2 |            2 | warna       | bukan_warna | 1.98%          |
|   588 | ADI     | CARBON        | ALPHABOUNCE RC2 M-CARBON/CARBON/CBLACK                                                   |           5 |            6 | warna       | bukan_warna | 1.97%          |
|  2909 | ADI     | CARBON        | DURAMO 9-CARBON/ONIX/GRETWO                                                              |           3 |            5 | warna       | bukan_warna | 1.97%          |
| 30705 | NIK     | CARBON        | NIKE REVOLUTION 4 (TDV)-GRIDIRON/MTLC PEWTER-LIGHT CARBON-BLACK                          |           9 |           10 | warna       | bukan_warna | 1.97%          |
|  8629 | ADI     | CARBON        | BOS TEE M-CARBON                                                                         |           4 |            4 | warna       | bukan_warna | 1.97%          |
|  7864 | ADI     | CARBON        | QUESTAR DRIVE W-CARBON/CARBON/AERPNK                                                     |           4 |            6 | warna       | bukan_warna | 1.97%          |
|  6764 | ADI     | CARBON        | TUBULAR DOOM SOCK W-CARBON/TECEAR/OWHITE                                                 |           5 |            7 | warna       | bukan_warna | 1.97%          |
| 41017 | NIK     | CARBON        | AIR FORCE 1 GTX-BLACK/BLACK-LIGHT CARBON-BRIGHT CERAMIC                                  |           8 |           10 | warna       | bukan_warna | 1.97%          |
| 30760 | NIK     | CARBON        | NIKE REVOLUTION 4 (GS)-GRIDIRON/MTLC PEWTER-LIGHT CARBON-BLACK                           |           9 |           10 | warna       | bukan_warna | 1.97%          |
|  3315 | ADI     | CARBON        | ULTRABOOST-FTWWHT/CARBON/BLUSPI                                                          |           3 |            4 | warna       | bukan_warna | 1.97%          |
|   244 | ADI     | CARBON        | ULTRABOOST-LEGINK/CARBON/BLUSPI                                                          |           3 |            4 | warna       | bukan_warna | 1.97%          |
|  6806 | ADI     | CARBON        | TUBULAR DOOM SOCK PK W-CARBON/CBLACK/SYELLO                                              |           6 |            8 | warna       | bukan_warna | 1.97%          |
|  7805 | ADI     | CARBON        | QUESTAR RIDE W-CARBON/CBLACK/GRETWO                                                      |           4 |            6 | warna       | bukan_warna | 1.97%          |
| 31743 | NIK     | CARBON        | AIR FORCE 1 FOAMPOSITE CUP-LIGHT CARBON/LIGHT CARBON-BLACK                               |           7 |           10 | warna       | bukan_warna | 1.97%          |
|  6992 | ADI     | CARBON        | EQT BASK ADV-CARBON/CARBON/CROYAL                                                        |           5 |            6 | warna       | bukan_warna | 1.97%          |
|  1466 | ADI     | CARBON        | PROPHERE W-CBLACK/SHOPNK/CARBON                                                          |           5 |            5 | warna       | bukan_warna | 1.97%          |
|  7699 | ADI     | CARBON        | CF QT RACER W-CBLACK/FTWWHT/CARBON                                                       |           7 |            7 | warna       | bukan_warna | 1.97%          |
| 31880 | NIK     | CARBON        | AS NIKE SWOOSH BRA-CARBON HEATHER/ANTHRACITE/BLACK                                       |           5 |            8 | warna       | bukan_warna | 1.97%          |
|  7049 | ADI     | CARBON        | CLIMACOOL 02/17-FTWWHT/CARBON/GUM416                                                     |           5 |            6 | warna       | bukan_warna | 1.97%          |
| 46731 | NIK     | CARBON        | NIKE DUNK LOW PRM-CARBON GREEN/RIFTBLUE-SAIL-CHILE RED                                   |           5 |           10 | warna       | bukan_warna | 1.97%          |
| 29082 | NIK     | CARBON        | NIKE DOWNSHIFTER 8-LIGHT CARBON/MTLC PEWTER-PEAT MOSS-BLACK                              |           5 |           10 | warna       | bukan_warna | 1.97%          |
| 38387 | NIK     | CARBON        | AS M NSW TEE CAMO PACK 1 AS-LIGHT CARBON/BLUE RECALL/WHITE                               |          10 |           13 | warna       | bukan_warna | 1.97%          |
|  6076 | ADI     | CARBON        | ULTRABOOST J-CWHITE/CHAPEA/CARBON                                                        |           5 |            5 | warna       | bukan_warna | 1.97%          |
|  2698 | ADI     | CARBON        | PUREBOOST DPR LTD-CBLACK/CBLACK/CARBON                                                   |           6 |            6 | warna       | bukan_warna | 1.97%          |
| 15871 | ADI     | CARBON        | FLUIDSTREET-CARBON                                                                       |           2 |            2 | warna       | bukan_warna | 1.97%          |
|  8669 | ADI     | CARBON        | BP DAILY-RAWGRE/CARBON/WHITE                                                             |           4 |            5 | warna       | bukan_warna | 1.97%          |
|  1527 | ADI     | CARBON        | SWIFT RUN W-CBLACK/CARBON/FTWWHT                                                         |           5 |            6 | warna       | bukan_warna | 1.97%          |
|  9671 | ADI     | CARBON        | CAMO OTH HOODY-CARBON                                                                    |           4 |            4 | warna       | bukan_warna | 1.97%          |
|  5372 | ADI     | CARBON        | FLUIDCLOUD AMBITIOUS M-CBLACK/FTWWHT/CARBON                                              |           6 |            6 | warna       | bukan_warna | 1.97%          |
| 28969 | NIK     | CARBON        | AS W NSW TCH FLC CAPE FZ-CARBON HEATHER/BLACK                                            |           8 |           10 | warna       | bukan_warna | 1.97%          |
| 32998 | NIK     | 23            | PSG M NK BRT STAD JSY SS AW-INFRARED 23/BLACK                                            |          10 |           11 | warna       | bukan_warna | 1.88%          |
| 25371 | NIK     | 23            | JORDAN AIR JUMPMAN-BLACK/INFRARED 23                                                     |           6 |            6 | warna       | bukan_warna | 1.88%          |
| 21685 | HER     | NIGHT         | FOURTEEN-NIGHT CAMO                                                                      |           2 |            3 | warna       | bukan_warna | 1.85%          |
| 10573 | ADI     | METAL         | NMD_R1-METAL GREY                                                                        |           2 |            3 | warna       | bukan_warna | 1.74%          |
| 21167 | HER     | FOREST        | SETTLEMENT-FOREST                                                                        |           2 |            2 | warna       | bukan_warna | 1.72%          |
| 21507 | HER     | FOREST        | NOVEL-FOREST NIGHT/DARK DENIM                                                            |           2 |            5 | warna       | bukan_warna | 1.72%          |
|  5964 | ADI     | CLOUD         | FUTUREPACER-CLOUD WHITE                                                                  |           2 |            3 | warna       | bukan_warna | 1.68%          |
| 22780 | KIP     | SHADOW        | FS72-SHADOW BROWN-140                                                                    |           2 |            4 | warna       | bukan_warna | 1.52%          |
| 10776 | ADI     | CRYSTAL       | OZWEEGO-CRYSTAL WHITE                                                                    |           2 |            3 | warna       | bukan_warna | 1.25%          |
| 10029 | ADI     | ASH           | NITE JOGGER-ASH SILVER                                                                   |           3 |            4 | warna       | bukan_warna | 1.12%          |
| 21384 | HER     | ASH           | SEVENTEEN-ASH ROSE                                                                       |           2 |            3 | warna       | bukan_warna | 1.12%          |
|  8451 | ADI     | ASH           | FALCON W-ASH PEARL S18                                                                   |           3 |            5 | warna       | bukan_warna | 1.12%          |
| 21428 | HER     | ASH           | FIFTEEN-ASH ROSE                                                                         |           2 |            3 | warna       | bukan_warna | 1.12%          |
|  9705 | ADI     | ASH           | CROPPED SWEAT-ASH PEARL S18                                                              |           3 |            5 | warna       | bukan_warna | 1.12%          |
| 56484 | WAR     | NEON          | 125CM NEON REFLECTIVE ROPE LACES                                                         |           2 |            5 | warna       | bukan_warna | 1.01%          |
| 54580 | REL     | 49            | REGULAR LACES-LIGHTBLUE/WHITE ROPE 49                                                    |           6 |            6 | warna       | bukan_warna | 0.95%          |
| 54576 | REL     | 49            | REGULAR LACES-BLACK/BLUE ROPE 49                                                         |           6 |            6 | warna       | bukan_warna | 0.95%          |
| 54571 | REL     | 49            | METAL AGLETS-WHITE WAXED FLAT GOLD AGLET 49                                              |           8 |            8 | warna       | bukan_warna | 0.95%          |
| 54549 | REL     | 49            | METAL AGLETS-WHITE ROPE SILVER AGLETS ROPE 49                                            |           8 |            8 | warna       | bukan_warna | 0.95%          |
| 54527 | REL     | 49            | REFLECTIVE LACES-BLACK STRIPES 3M ROPE 49                                                |           7 |            7 | warna       | bukan_warna | 0.95%          |
| 54562 | REL     | 49            | REGULAR LACES-GREY/WHITE ROPE 49                                                         |           6 |            6 | warna       | bukan_warna | 0.95%          |
| 54733 | REL     | 49            | REFLECTIVE LACES-BLACK STRIPES 3M ROPE 49/125cm                                          |           7 |            8 | warna       | bukan_warna | 0.95%          |
| 54557 | REL     | 49            | REGULAR LACES-RED/WHITE ROPE 49                                                          |           6 |            6 | warna       | bukan_warna | 0.95%          |
|  2207 | ADI     | TURBO         | EQT SUPPORT RF-FTWWHT/FTWWHT/TURBO                                                       |           6 |            6 | warna       | bukan_warna | 0.94%          |
|  2424 | ADI     | TURBO         | EQT RACING 91 W-CBLACK/CBLACK/TURBO                                                      |           7 |            7 | warna       | bukan_warna | 0.94%          |
| 21145 | HER     | ECLIPSE       | RETREAT-ECLIPSE X                                                                        |           2 |            3 | warna       | bukan_warna | 0.64%          |
    


```python
# selesai dengan model 2, bersihkan memori di GPU terkait model_2
del model_2
gc.collect()
```




    26866



## Model 3: Menggunakan Posisi Kata, Tipe Brand dan Lapisan Embed Custom yang diconcatenate

Pada skenario training ini, kita akan mencoba untuk tidak hanya menggunakan input data kata dan label saja, tetapi juga menambahkan posisi kata dalam kalimat dan juga tipe brand sebagai variabel independen (fitur) yang mungkin dapat menentukan apakah sebuah kata adalah `bukan_warna` atau `warna`.
Pada dasarnya kita akan membuat tumpukan layer yang mempelajari input data per kata dan juga mempelajari posisi dari kata dan juga tipe brand yang kemudian akan ditumpuk (*stack*) menjadi satu lapisan baru.

### Preprocessing untuk `urut_kata` dan `total_kata`?
Sebelum kita melakukan *training* pada variabel `urut_kata` dan juga `total_kata`. Terdapat setidaknya dua opsi untuk menggunakan variabel independen ini:

1. Melakukan pembagian `urut_kata` dan `total_kata` sehingga nilai dari posisi adalah diantara 0 sampai dengan 1, dimana ~0 menunjukkan posisi kata yang berada dekat dengan awal kalimat dan ~1 adalah posisi kata yang berada dengan akhir kalimat. Namun hal ini akan menyebabkan kita akan kehilangan beberapa informasi dari input data, sebagai contoh nilai 0.5 bisa diartikan sebagai kata pertama dalam kalimat yang terdiri dari dua kata (1/2) atau bisa juga berarti kata kelima dalam kalimat yang terdiri dari sepuluh kata (5/10).
2. Melakukan encoding untuk `urut_kata` dan `total_kata` sehingga kita tidak kehilangan informasi dari kedua variabel ini.
3. Menggunakan `urut_kata` dan `total_kata` sebagaimana adanya tanpa melakukan pembagian dan membiarkan model berusaha mempelajari makna dari variabel independen ini.

Pada bagian ini, kita akan mencoba untuk menggunakan opsi kedua.


### USE atau Custom Embedding?
Dikarenakan pada model_2 kita dapat melihat bahwa penggunaan lapisan ekstraktor fitur USE tidak dapat dengan baik memprediksi label kata (hal ini bisa disebabkan oleh beberapa hal, seperti fakta bahwa USE dilatih pada dataset *corpus* bahasa internasional yang mungkin lebih baku), maka kita akan kembali menggunakan lapisan *custom embedding* menggunakan Conv1D atau lapisan LSTM dua arah (*bi-directional LSTM*) seperti pada model_1.

### `urut_kata` dan `total_kata` Embedding

Pada bagian ini kita akan melakukan embedding untuk posisional dari sebuah kata dalam kalimat seperti apa yang direpresentasikan oleh kolom `urut_kata` dan `total_kata`.


```python
# Distribusi jumlah kata dalam artikel - artikel
train_data['urut_kata'].value_counts()
```




    2.0     11981
    1.0     11950
    5.0      3770
    3.0      3638
    4.0      3540
    6.0      3237
    7.0      2568
    8.0      1907
    9.0      1273
    10.0      858
    11.0      444
    12.0      172
    13.0       46
    14.0       12
    15.0        4
    Name: urut_kata, dtype: int64




```python
train_data['urut_kata'].plot.hist()
plt.title('Distribusi Jumlah Kata dalam Artikel', fontsize=18)
plt.figure(facecolor='w')
plt.show()
```


    
![png](ColorSkim_AI_files/ColorSkim_AI_83_0.png)
    


```python
# Karena panjang kata berbeda - beda, maka kita akan menggunakan one_hot encoding
# dengan depth max
max_kata = int(np.max(train_data['urut_kata']))

# Membuat one_hot tensor untuk kolom 'urut_kata'
train_data_urut_kata_one_hot = tf.one_hot(train_data['urut_kata'].to_numpy(), 
                                          depth=max_kata)
test_data_urut_kata_one_hot = tf.one_hot(test_data['urut_kata'].to_numpy(),
                                         depth=max_kata)
train_data_urut_kata_one_hot[:15], train_data_urut_kata_one_hot.shape
```




    (<tf.Tensor: shape=(15, 15), dtype=float32, numpy=
     array([[0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 1., 0., 0.],
            [0., 0., 0., 0., 1., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0.],
            [0., 1., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0.],
            [0., 0., 1., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0.],
            [0., 0., 0., 1., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0.],
            [0., 0., 0., 0., 0., 1., 0., 0., 0., 0., 0., 0., 0., 0., 0.],
            [0., 0., 1., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0.],
            [0., 0., 1., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0.],
            [0., 0., 0., 1., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0.],
            [0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 1., 0., 0., 0.],
            [0., 0., 0., 0., 0., 0., 0., 0., 1., 0., 0., 0., 0., 0., 0.],
            [0., 1., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0.],
            [0., 0., 1., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0.],
            [0., 0., 1., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0.],
            [0., 1., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0.]],
           dtype=float32)>,
     TensorShape([45400, 15]))




```python
# Distribusi 'total_kata'
train_data['total_kata'].value_counts()
```




    6.0     6476
    5.0     6063
    7.0     5918
    8.0     5555
    9.0     5009
    4.0     4497
    10.0    4494
    11.0    2924
    3.0     2452
    12.0    1222
    13.0     405
    2.0      270
    14.0      77
    15.0      29
    1.0        9
    Name: total_kata, dtype: int64




```python
train_data['total_kata'].plot.hist()
plt.title('Distribusi Panjang Kalimat dalam Artikel', fontsize=18)
plt.figure(facecolor='w')
plt.show()
```


    
![png](ColorSkim_AI_files/ColorSkim_AI_86_0.png)
    



```python
# Lakukan hal yang sama untuk total_kata
max_kalimat = int(np.max(train_data['total_kata']))

# Membuat one_hot tensor untuk kolom 'total_kata'
train_data_total_kata_one_hot = tf.one_hot(train_data['total_kata'].to_numpy(),
                                           depth=max_kalimat)
test_data_total_kata_one_hot = tf.one_hot(test_data['total_kata'].to_numpy(),
                                          depth=max_kalimat)
train_data_total_kata_one_hot[:15], train_data_total_kata_one_hot.shape

```




    (<tf.Tensor: shape=(15, 15), dtype=float32, numpy=
     array([[0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 1., 0., 0.],
            [0., 0., 0., 0., 1., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0.],
            [0., 0., 0., 0., 0., 0., 0., 1., 0., 0., 0., 0., 0., 0., 0.],
            [0., 0., 0., 0., 1., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0.],
            [0., 0., 0., 0., 1., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0.],
            [0., 0., 0., 0., 0., 0., 0., 0., 1., 0., 0., 0., 0., 0., 0.],
            [0., 0., 0., 0., 0., 0., 1., 0., 0., 0., 0., 0., 0., 0., 0.],
            [0., 0., 0., 0., 0., 0., 0., 0., 0., 1., 0., 0., 0., 0., 0.],
            [0., 0., 0., 0., 1., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0.],
            [0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 1., 0., 0.],
            [0., 0., 0., 0., 0., 0., 0., 0., 1., 0., 0., 0., 0., 0., 0.],
            [0., 0., 0., 0., 0., 1., 0., 0., 0., 0., 0., 0., 0., 0., 0.],
            [0., 0., 0., 0., 1., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0.],
            [0., 0., 0., 0., 0., 0., 0., 0., 1., 0., 0., 0., 0., 0., 0.],
            [0., 0., 0., 0., 0., 0., 1., 0., 0., 0., 0., 0., 0., 0., 0.]],
           dtype=float32)>,
     TensorShape([45400, 15]))



#### Membuat TensorFlow Dataset, Batching dan Preteching untuk Model 3


```python
# Membuat train dan test dataset (dengan 4 jenis input data)
# Urutan dataset disesuaikan dengan urutan input data pada tf.keras.Model akhir (model_3)
train_fitur_data = from_tensor_slices((train_data.iloc[:, 0],
                                      train_data.iloc[:, 3:].to_numpy(),
                                      train_data_urut_kata_one_hot,
                                      train_data_total_kata_one_hot))
train_target_data = from_tensor_slices(train_target)
train_dataset = zip((train_fitur_data,
                     train_target_data))
train_dataset = train_dataset.batch(UKURAN_BATCH).prefetch(tf.data.AUTOTUNE)

test_fitur_data = from_tensor_slices((test_data.iloc[:, 0],
                                      test_data.iloc[:, 3:].to_numpy(),
                                      test_data_urut_kata_one_hot,
                                      test_data_total_kata_one_hot))
test_target_data = from_tensor_slices(test_target)
test_dataset = zip((test_fitur_data,
                    test_target_data))
test_dataset = test_dataset.batch(UKURAN_BATCH).prefetch(tf.data.AUTOTUNE)
```


```python
# Check ukuran dan dimensi dataset
train_dataset, test_dataset
```




    (<BatchDataset element_spec=((TensorSpec(shape=(None,), dtype=tf.string, name=None), TensorSpec(shape=(None, 37), dtype=tf.float64, name=None), TensorSpec(shape=(None, 15), dtype=tf.float32, name=None), TensorSpec(shape=(None, 15), dtype=tf.float32, name=None)), TensorSpec(shape=(None,), dtype=tf.int32, name=None))>,
     <BatchDataset element_spec=((TensorSpec(shape=(None,), dtype=tf.string, name=None), TensorSpec(shape=(None, 37), dtype=tf.float64, name=None), TensorSpec(shape=(None, 15), dtype=tf.float32, name=None), TensorSpec(shape=(None, 15), dtype=tf.float32, name=None)), TensorSpec(shape=(None,), dtype=tf.int32, name=None))>)



### Membangun dan Menjalankan Training Model 3

1. Membuat model untuk `kata`
2. Membuat model untuk `brand`
3. Membuat model untuk `urut_kata`
4. Membuat model untuk `total_kata`
5. Mengkombinasikan output 1 & 2 menggunakan `tf.keras.layers.Concatenate`
6. Menambahkan Dense dan Dropout layer untuk poin 5
7. Mengkombinasikan output dari poin 3 & 4 menggunakan `tf.keras.layers.Concatenate`
8. Mengkombinasikan output 6 & 7 menggunakan `tf.keras.layers.Concatenate`
9. Membuat lapisan output yang menerima input dari 2 lapisan embedding di poin 8 dan menghasilkan output probabilitas
8. Mengkombinasikan input 1, 2, 3, 4 dan output 9 dalam `tf.keras.Model`


```python
# Jika folder dengan path 'colorskim_checkpoint/{model.name}' sudah ada, maka skip fit model 
# untuk menghemat waktu pengembangan dan hanya load model yang sudah ada dalam folder tersebut 
# saja
if not os.path.isdir(f'colorskim_checkpoint/{MODEL[3]}'):
        # set random seed
        tf.random.set_seed(RANDOM_STATE)
        
        # 1. model kata
        input_kata = Input(shape=(1,), dtype=tf.string, name='lapisan_input_kata')
        lapisan_vektorisasi_kata = lapisan_vektorisasi(input_kata)
        lapisan_embedding_kata = lapisan_embedding(lapisan_vektorisasi_kata)
        lapisan_bi_lstm_kata = Bidirectional(LSTM(units=UKURAN_BATCH), 
                                        name='lapisan_bidirectional_lstm_kata')(lapisan_embedding_kata)
        # ubah bi_lstm menjadi Conv1D dengan GlobalMaxPooling
        # lapisan_Conv1D = Conv1D(filters=UKURAN_BATCH,
        #                         kernel_size=5,
        #                         padding='same',
        #                         activation='relu',
        #                         name='lapisan_konvolusional_1D')(lapisan_embedding_kata)
        # lapisan_max_pooling = GlobalMaxPooling1D(name='lapisan_max_pooling')(lapisan_Conv1D)
        model_kata = Model(input_kata,
                        lapisan_bi_lstm_kata,
                        name='model_kata')

        # 2. model brand
        input_brand = Input(shape=(train_data.iloc[:, 3:].shape[1],), 
                        dtype=tf.float32,
                        name='lapisan_input_brand')
        x = Dense(units=UKURAN_BATCH, 
                activation='relu',
                name='lapisan_dense_rectified_linear_unit_brand')(input_brand)
        model_brand = Model(input_brand,
                        x,
                        name='model_brand')

        # 3. model urut_kata
        input_urut_kata = Input(shape=(max_kata,),
                                dtype=tf.float32,
                                name='lapisan_input_urut_kata')
        x = Dense(units=UKURAN_BATCH,
                activation='relu',
                name='lapisan_dense_rectified_linear_unit_urut_kata')(input_urut_kata)
        model_urut_kata = Model(input_urut_kata,
                                x,
                                name='model_urut_kata')

        # 4. model total_kata
        input_total_kata = Input(shape=(max_kalimat,),
                                dtype=tf.float32,
                                name='lapisan_input_total_kata')
        x = Dense(units=UKURAN_BATCH,
                activation='relu',
                name='lapisan_dense_rectified_linear_unit_total_kata')(input_total_kata)
        model_total_kata = Model(input_total_kata,
                                x,
                                name='model_total_kata')

        # 5. Mengkombinasikan model kata dan brand
        kombinasi_kata_brand = Concatenate(name='lapisan_kombinasi_kata_brand')([model_kata.output, 
                                                                                model_brand.output])

        # 6. Menambahkan lapisan dense dan dropout dari output kombinasi kata dan brand
        z = Dense(256, 
                activation='relu',
                name='lapisan_dense_kombinasi_kata_brand')(kombinasi_kata_brand)
        z = Dropout(0.5)(z)

        # 7. Mengkombinasikan model 3 dan 4
        kombinasi_urut_total_kata = Concatenate(name='lapisan_kombinasi_urut_total_kata')([model_urut_kata.output, 
                                                                                        model_total_kata.output])

        # 8. Mengkombinasikan output di poin 6 dengan lapisan_kombinasi_urut_total_kata
        kombinasi_kata_brand_urut_total_kata = Concatenate(name='lapisan_kombinasi_kata_brand_urut_total_kata')([z,
                                                                                                                kombinasi_urut_total_kata])

        # 9. Membuat lapisan output final
        lapisan_output = Dense(units=1,
                        activation='sigmoid',
                        name='lapisan_output_final')(kombinasi_kata_brand_urut_total_kata)

        # 10. Menyatukan semua model dari berbagai input
        model_3 = Model(inputs=[model_kata.input,
                                model_brand.input,
                                model_urut_kata.input,
                                model_total_kata.input],
                        outputs=lapisan_output,
                        name=MODEL[3])

        # Compile model_3
        model_3.compile(loss=BinaryCrossentropy(),
                        optimizer=Adam(),
                        metrics=['accuracy'])

        # Setup wandb init dan config
        wb.init(project=wandb['proyek'],
                entity=wandb['user'],
                name=model_3.name,
                config={'epochs': EPOCHS,
                        'n_layers': len(model_3.layers)})


        # Fit model_3
        model_3.fit(train_dataset,
                    epochs=EPOCHS,
                    validation_data=test_dataset,
                    callbacks=[wandb_callback(train_dataset),
                               model_checkpoint(model_3.name),
                               reduce_lr_on_plateau(),
                               early_stopping()])
        
        # tutup logging wandb
        wb.finish()
        
        # load model_3
        model_3 = load_model(f'colorskim_checkpoint/{MODEL[3]}')
else:
        # load model_3
        model_3 = load_model(f'colorskim_checkpoint/{MODEL[3]}')
```


```python
# Ringkasan dari model_3
model_3.summary()
```

    Model: "model_3_quadbrid_embedding"
    __________________________________________________________________________________________________
     Layer (type)                   Output Shape         Param #     Connected to                     
    ==================================================================================================
     lapisan_input_kata (InputLayer  [(None, 1)]         0           []                               
     )                                                                                                
                                                                                                      
     lapisan_vektorisasi (TextVecto  (None, 1)           0           ['lapisan_input_kata[0][0]']     
     rization)                                                                                        
                                                                                                      
     lapisan_embedding (Embedding)  (None, 1, 32)        92992       ['lapisan_vektorisasi[0][0]']    
                                                                                                      
     lapisan_input_brand (InputLaye  [(None, 37)]        0           []                               
     r)                                                                                               
                                                                                                      
     lapisan_bidirectional_lstm_kat  (None, 64)          16640       ['lapisan_embedding[0][0]']      
     a (Bidirectional)                                                                                
                                                                                                      
     lapisan_dense_rectified_linear  (None, 32)          1216        ['lapisan_input_brand[0][0]']    
     _unit_brand (Dense)                                                                              
                                                                                                      
     lapisan_kombinasi_kata_brand (  (None, 96)          0           ['lapisan_bidirectional_lstm_kata
     Concatenate)                                                    [0][0]',                         
                                                                      'lapisan_dense_rectified_linear_
                                                                     unit_brand[0][0]']               
                                                                                                      
     lapisan_input_urut_kata (Input  [(None, 15)]        0           []                               
     Layer)                                                                                           
                                                                                                      
     lapisan_input_total_kata (Inpu  [(None, 15)]        0           []                               
     tLayer)                                                                                          
                                                                                                      
     lapisan_dense_kombinasi_kata_b  (None, 256)         24832       ['lapisan_kombinasi_kata_brand[0]
     rand (Dense)                                                    [0]']                            
                                                                                                      
     lapisan_dense_rectified_linear  (None, 32)          512         ['lapisan_input_urut_kata[0][0]']
     _unit_urut_kata (Dense)                                                                          
                                                                                                      
     lapisan_dense_rectified_linear  (None, 32)          512         ['lapisan_input_total_kata[0][0]'
     _unit_total_kata (Dense)                                        ]                                
                                                                                                      
     dropout (Dropout)              (None, 256)          0           ['lapisan_dense_kombinasi_kata_br
                                                                     and[0][0]']                      
                                                                                                      
     lapisan_kombinasi_urut_total_k  (None, 64)          0           ['lapisan_dense_rectified_linear_
     ata (Concatenate)                                               unit_urut_kata[0][0]',           
                                                                      'lapisan_dense_rectified_linear_
                                                                     unit_total_kata[0][0]']          
                                                                                                      
     lapisan_kombinasi_kata_brand_u  (None, 320)         0           ['dropout[0][0]',                
     rut_total_kata (Concatenate)                                     'lapisan_kombinasi_urut_total_ka
                                                                     ta[0][0]']                       
                                                                                                      
     lapisan_output_final (Dense)   (None, 1)            321         ['lapisan_kombinasi_kata_brand_ur
                                                                     ut_total_kata[0][0]']            
                                                                                                      
    ==================================================================================================
    Total params: 137,025
    Trainable params: 137,025
    Non-trainable params: 0
    __________________________________________________________________________________________________
    


```python
# plot struktur model_3
plot_model(model_3, show_shapes=True)
```




    
![png](ColorSkim_AI_files/ColorSkim_AI_94_0.png)
    



### Eksplorasi Hasil Model 3


```python
# Evaluasi model_3
model_3.evaluate(test_dataset)
```

    355/355 [==============================] - 40s 98ms/step - loss: 0.0214 - accuracy: 0.9944
    [0.021449144929647446, 0.9944498538970947]




```python
# Prediksi probabilitas model_3
model_3_pred_prob = tf.squeeze(model_3.predict(test_dataset))
model_3_pred_prob
```




    <tf.Tensor: shape=(11351,), dtype=float32, numpy=
    array([9.9962115e-01, 5.4790482e-05, 9.9976093e-01, ..., 2.4132943e-08,
           3.0743347e-08, 3.4280158e-05], dtype=float32)>




```python
# Membuat prediksi model_3
model_3_pred = tf.round(tf.round(model_3_pred_prob))
model_3_pred
```




    <tf.Tensor: shape=(11351,), dtype=float32, numpy=array([1., 0., 1., ..., 0., 0., 0.], dtype=float32)>




```python
# Metriks skor model_3
model_3_metrik = hitung_metrik(target=test_target,
                               prediksi=model_3_pred)
model_3_metrik
```




    {'akurasi': 0.9944498282089683,
     'presisi': 0.9944503482062841,
     'recall': 0.9944498282089683,
     'f1-score': 0.9944482769856459}




```python
# Plot residual model_3
residual_plot_logr(test_target=test_target,
                   nama_model=MODEL[3],
                   model_akurasi=model_3_metrik['akurasi'],
                   probabilitas_prediksi_model=tf.squeeze(model_3_pred_prob))
```


    
![png](ColorSkim_AI_files/ColorSkim_AI_100_0.png)
    



```python
# Plot confusion matrix
plot_conf_matrix(target_label=test_target,
                 prediksi_label=model_3_pred,
                 nama_model=MODEL[3],
                 akurasi=model_3_metrik['akurasi'],
                 label_titik_x=['bukan_warna', 'warna'],
                 label_titik_y=['bukan_warna', 'warna'])
plt.figure(facecolor='w')
plt.show()
```


    
![png](ColorSkim_AI_files/ColorSkim_AI_101_0.png)
    



```python
# Dataframe kesalahan prediksi model_3
df_kesalahan_prediksi(label_encoder=label_encoder,
                      test_data=test_data_mnb,
                      prediksi=model_3_pred,
                      probabilitas_prediksi=model_3_pred_prob,
                      order_ulang_header=['brand',
                                          'kata',
                                          'nama_artikel',
                                          'urut_kata',
                                          'total_kata',
                                          'label'])
```

|       | brand   | kata        | nama_artikel                                               |   urut_kata |   total_kata | label       | prediksi    | probabilitas   |
|------:|:--------|:------------|:-----------------------------------------------------------|------------:|-------------:|:------------|:------------|:---------------|
| 21091 | HER     | 600D        | POP QUIZ-600D POLY NAVY/ZIP                                |           3 |            6 | bukan_warna | warna       | 99.94%         |
| 21174 | HER     | RED         | HER-HERITAGE-BOSTON RED SOX-(21L)-BAG-US                   |           4 |            8 | bukan_warna | warna       | 99.77%         |
| 16288 | ADI     | BLK         | CLR BLK CRW 2PP-BLACK                                      |           2 |            5 | bukan_warna | warna       | 99.4%          |
| 17198 | AGL     | YELLOW      | ESAGLXY YELLOW CRICKET LIGHTER -YELLOW                     |           2 |            5 | bukan_warna | warna       | 97.89%         |
| 48075 | PTG     | ORANGE      | POLKA ORANGE-ORANGE                                        |           2 |            3 | bukan_warna | warna       | 97.78%         |
| 48153 | PTG     | DOVE        | MISTY DOVE-GREY                                            |           2 |            3 | bukan_warna | warna       | 94.19%         |
| 21451 | HER     | MEDIUM      | MAMMOTH MEDIUM-ARROWWOOD                                   |           2 |            3 | bukan_warna | warna       | 83.72%         |
|  8735 | ADI     | FULL        | FULL ZIP-CWHITE                                            |           1 |            3 | bukan_warna | warna       | 82.15%         |
| 56226 | WAR     | ORANGE      | SHOELACES ORANGE OVAL LACES-ORANGE                         |           2 |            5 | bukan_warna | warna       | 81.07%         |
|  8962 | ADI     | CORE        | LIN CORE ORG-BLACK                                         |           2 |            4 | bukan_warna | warna       | 78.66%         |
|  8965 | ADI     | CORE        | LIN CORE CROSSB-BLACK                                      |           2 |            4 | bukan_warna | warna       | 78.66%         |
|  8968 | ADI     | CORE        | LIN CORE BP-BLACK                                          |           2 |            4 | bukan_warna | warna       | 78.66%         |
| 17520 | AGL     | BROWN       | BROWN MOUNTAIN 008 - PEACH                                 |           1 |            4 | bukan_warna | warna       | 77.41%         |
| 56083 | WAR     | GLOW        | ROPE GLOW IN THE DARK-WHITE                                |           2 |            6 | bukan_warna | warna       | 67.72%         |
| 56086 | WAR     | GLOW        | FLAT GLOW IN THE DARK-WHITE                                |           2 |            6 | bukan_warna | warna       | 67.72%         |
| 14574 | ADI     | OFF         | PRIDE OFF CENTE-MULTICOLOR                                 |           2 |            4 | bukan_warna | warna       | 66.82%         |
|  9992 | ADI     | TECH        | MARATHON TECH-EASY YELLOW                                  |           2 |            4 | bukan_warna | warna       | 64.85%         |
| 21176 | HER     | ANGELS      | HERITAGE ANGELS (21L)                                      |           2 |            3 | bukan_warna | warna       | 62.74%         |
| 19643 | BEA     | 35          | SERIES 35-MULTI                                            |           2 |            3 | bukan_warna | warna       | 60.07%         |
| 16273 | ADI     | SWEATSHIRT  | OS SWEATSHIRT-BLACK                                        |           2 |            3 | bukan_warna | warna       | 50.28%         |
| 14605 | ADI     | REPEAT      | LINEAR REPEAT-BLACK                                        |           2 |            3 | bukan_warna | warna       | 50.28%         |
| 12202 | ADI     | COM         | UNIFO COM- WHITE                                           |           2 |            3 | bukan_warna | warna       | 50.28%         |
| 12345 | ADI     | FESTIV      | MONOGR FESTIV-MULTICOLOR                                   |           2 |            3 | bukan_warna | warna       | 50.28%         |
|  7224 | ADI     | SPR         | FL_TRH SPR-CARBON                                          |           2 |            3 | bukan_warna | warna       | 50.28%         |
| 15466 | ADI     | SAVANNAH    | OZELIA-SAVANNAH                                            |           2 |            2 | warna       | bukan_warna | 44.58%         |
| 13918 | ADI     | CARDBOARD   | NMD_R1.V2-CARDBOARD                                        |           2 |            2 | warna       | bukan_warna | 44.58%         |
| 13740 | ADI     | MAROON      | ULTRA4D-MAROON                                             |           2 |            2 | warna       | bukan_warna | 44.58%         |
| 10336 | ADI     | MAROON      | RUN60S-MAROON                                              |           2 |            2 | warna       | bukan_warna | 44.58%         |
|  7372 | ADI     | SGREEN      | PROPHERE-SGREEN/CGREEN/CBLACK                              |           2 |            4 | warna       | bukan_warna | 38.93%         |
|  2592 | ADI     | ICEPUR      | GAZELLE-ICEPUR/WHITE/GOLDMT                                |           2 |            4 | warna       | bukan_warna | 38.93%         |
|  4659 | ADI     | BOAQUA      | CAMPUS-BOAQUA/FTWWHT/CWHITE                                |           2 |            4 | warna       | bukan_warna | 38.93%         |
| 52652 | PUM     | GLOW        | SUEDE BLOC IVORY GLOW-PUMA BLACK                           |           4 |            6 | warna       | bukan_warna | 32.14%         |
| 14727 | ADI     | LEGEND      | SHOPPER-LEGEND INK                                         |           2 |            3 | warna       | bukan_warna | 28.49%         |
| 12023 | ADI     | LEGEND      | ASWEERUN-LEGEND INK                                        |           2 |            3 | warna       | bukan_warna | 28.49%         |
| 56661 | WAR     | THE         | 125CM THE BLUES FLAT LACES                                 |           2 |            5 | warna       | bukan_warna | 23.55%         |
|  5964 | ADI     | CLOUD       | FUTUREPACER-CLOUD WHITE                                    |           2 |            3 | warna       | bukan_warna | 19.68%         |
| 55804 | STN     | VOLT        | RAILWAY-VOLT                                               |           2 |            2 | warna       | bukan_warna | 18.83%         |
| 55759 | STN     | RASTA       | VIARTA-RASTA                                               |           2 |            2 | warna       | bukan_warna | 18.83%         |
| 55981 | STN     | OATMEAL     | XYZ-OATMEAL                                                |           2 |            2 | warna       | bukan_warna | 18.83%         |
| 22780 | KIP     | SHADOW      | FS72-SHADOW BROWN-140                                      |           2 |            4 | warna       | bukan_warna | 7.41%          |
| 54953 | SAU     | BRN         | COURAGEOUS-BRN/YEL                                         |           2 |            3 | warna       | bukan_warna | 4.62%          |
| 23355 | NIC     | 7           | NIKE KD FULL COURT 8P-AMBER/BLACK/METALLIC SILVER/BLACK 07 |          11 |           11 | warna       | bukan_warna | 3.41%          |
| 21982 | HER     | FLORAL      | HANSON-FLORAL BLR                                          |           2 |            3 | warna       | bukan_warna | 3.38%          |
| 56444 | WAR     | OREO        | 90CM OREO ROPE                                             |           2 |            3 | warna       | bukan_warna | 1.82%          |
| 17275 | AGL     | 5           | ITALIC 5 PANEL MAROON 005-MAROON                           |           5 |            6 | warna       | bukan_warna | 1.61%          |
| 26752 | NIK     | PEELORANGE  | WMNS KAWA SLIDEPINK PRIME/ORANGE PEELORANGE PEEL           |           6 |            7 | warna       | bukan_warna | 1.37%          |
|  1407 | ADI     | CARGO       | NMD_TS1 PK-NIGHT CARGO                                     |           4 |            4 | warna       | bukan_warna | 1.22%          |
|  1403 | ADI     | F17         | NMD_R1-GREY TWO F17                                        |           4 |            4 | warna       | bukan_warna | 1.2%           |
| 29098 | NIK     | 8ASHEN      | NIKE DOWNSHIFTER 8ASHEN SLATE/OBSIDIANDIFFUSED BLUEBLACK   |           3 |            6 | warna       | bukan_warna | 0.93%          |
| 56746 | WAR     | PAISLEY     | 125CM PAISLEY WHITE FLAT                                   |           2 |            4 | warna       | bukan_warna | 0.88%          |
| 31572 | NIK     | LIGHTCARBON | WMNS NIKE QUEST-LIGHTCARBON/BLACKLASER ORANGE              |           4 |            6 | warna       | bukan_warna | 0.82%          |
| 33814 | NIK     | EXPZ07WHITE | NIKE EXPZ07WHITE/BLACK                                     |           2 |            3 | warna       | bukan_warna | 0.8%           |
| 33831 | NIK     | EXPX14WHITE | NIKE EXPX14WHITE/WOLF GREYBLACK                            |           2 |            4 | warna       | bukan_warna | 0.51%          |
| 31091 | NIK     | VIALEBLACK  | NIKE VIALEBLACK/VOLTSOLAR REDANTHRACITE                    |           2 |            4 | warna       | bukan_warna | 0.51%          |
| 46940 | NIK     | REACTBRIGHT | NK REACTBRIGHT CRIMSON/DARK GREY/PURE PLATINUM             |           2 |            7 | warna       | bukan_warna | 0.32%          |
| 46960 | NIK     | FTR10PURE   | NK FTR10PURE PLATINUM/BRIGHT CRIMSON/DARK GREY             |           2 |            7 | warna       | bukan_warna | 0.32%          |
| 56484 | WAR     | NEON        | 125CM NEON REFLECTIVE ROPE LACES                           |           2 |            5 | warna       | bukan_warna | 0.11%          |
|  3490 | ADI     | SHOCK       | FUTUREPACER-SHOCK RED                                      |           2 |            3 | warna       | bukan_warna | 0.09%          |
| 10573 | ADI     | METAL       | NMD_R1-METAL GREY                                          |           2 |            3 | warna       | bukan_warna | 0.05%          |
| 55259 | STN     | AQUA        | FAMILY FORCE-AQUA                                          |           3 |            3 | warna       | bukan_warna | 0.01%          |
|  1405 | ADI     | PK          | NMD_TS1 PK-NIGHT CARGO                                     |           2 |            4 | warna       | bukan_warna | 0.01%          |
| 50395 | PUM     | PUMA        | RESOLVE PUMA BLACK-PUMA SILVER                             |           2 |            5 | warna       | bukan_warna | 0.01%          |
| 54951 | SAU     | TAN         | COURAGEOUS-TAN/PNK                                         |           2 |            3 | warna       | bukan_warna | 0.0%           |
    


```python
# selesai dengan model 3, bersihkan memori di GPU terkait model_3
del model_3
gc.collect()
```




    157980




```python
tf.config.experimental.get_memory_info('GPU:0')
```




    {'current': 25028096, 'peak': 1074304768}




```python
# Membuat fungsi untuk mengambil bobot masing - masing kata dalam neuron model dengan lapisan embedding
def get_bobot_kata(model_list,
                   lapisan_vektorisasi=lapisan_vektorisasi):
    """
    Fungsi ini akan menerima list model dengan lapisan embedding dan menghasilkan file
    vectors_{model.name}.tsv serta file metadata.tsv untuk diproyeksikan
    dalam bidang 3D tensorboard atau projector.tensorflow.org
    
    Args:
        model_list (list): List model yang akan diekstrak bobot neuronnya dan dipergunakan sebagai input di bidang 3D
        lapisan_vektorisasi (tf.keras.layers.TextVectorization): Lapisan text vektorisasi yang akan diambil vocabularynya untuk file metadata.tsv
    """
    kata_dalam_vektorizer = lapisan_vektorisasi.get_vocabulary()
    
    for model in model_list:
        model = load_model(f'colorskim_checkpoint/{model}')
        bobot_kata_embed = model.get_layer('lapisan_embedding').get_weights()[0]
        file_vektor = io.open(f'vectors_{model.name}.tsv', 'w', encoding='utf-8')
        for indeks, _ in enumerate(kata_dalam_vektorizer):
            if indeks == 0:
                continue
            vektor = bobot_kata_embed[indeks]
            file_vektor.write('\t'.join([str(x) for x in vektor]) + '\n')
        file_vektor.close()
        del model
        gc.collect()
    
    file_metadata = io.open(f'metadata.tsv', 'w', encoding='utf-8')
    for kata in kata_dalam_vektorizer:
        file_metadata.write(kata + '\n')
    file_metadata.close()         
```


```python
# Mengekstrak bobot masing - masing kata dalam neuron dengan model embedding
get_bobot_kata([MODEL[1], MODEL[3]], lapisan_vektorisasi)
```

## Perbandingan Kinerja dari setiap Model

Dari kesemua model yang sudah kita lakukan training, dapat disimpulkan bahwa:

1. Model 0 dan Model 1 memiliki tingkat akurasi yang cukup tinggi dan nilainya hampir serupa, meskipun model ini cukup sederhana.
2. Model 2 memiliki tingkat akurasi yang paling buruk dibandingkan semua model lainnya. Hal ini bisa dikarenakan banyak inkonsistensi dan typo dalam penulisan nama_artikel yang ada pada data dan *universal sentence encoder* sebagai sebuah lapisan ekstraktor fitur yang sudah ditrain menggunakan corpus (internasional) yang baku mungkin mengalami kesulitan dalam mengekstrak fitur dari data.
3. Model 3 merupakan model dengan tingkat akurasi yang paling tinggi diantara kesemua model. Hal ini mungkin dikarenakan kita menambahkan variabel posisi kata dalam kalimat sebagai salah satu variabel independen yang mempengaruhi output dari variabel dependen label `bukan_warna` atau `warna`.

Laporan log untuk ColorSkim dapat diakses di [*Weights & Biases - ColorSkim*](https://wandb.ai/jpao/ColorSkim?workspace=user-jpao)

**Accuracy**

![accuracy_wandb](images/model_accuracy.png)

**Loss**

![loss_wandb](images/model_loss.png)

**Logs**

![performance_wandb](images/model_performance.png)


```python
# Mengkombinasikan hasil model ke dalam dataframe
hasil_semua_model = pd.DataFrame({"model_0_multinomial_naive_bayes": model_0_metrik,
                                  "model_1_Conv1D_vektorisasi_embedding": model_1_metrik,
                                  "model_2_Conv1D_USE_embed": model_2_metrik,
                                  "model_3_quadbrid_embedding": model_3_metrik})
hasil_semua_model = hasil_semua_model.transpose()
print(hasil_semua_model.to_markdown())
```

    |                                      |   akurasi |   presisi |   recall |   f1-score |
    |:-------------------------------------|----------:|----------:|---------:|-----------:|
    | model_0_multinomial_naive_bayes      |  0.992159 |  0.99216  | 0.992159 |   0.992156 |
    | model_1_Conv1D_vektorisasi_embedding |  0.992071 |  0.992072 | 0.992071 |   0.992068 |
    | model_2_Conv1D_USE_embed             |  0.93886  |  0.939021 | 0.93886  |   0.938596 |
    | model_3_quadbrid_embedding           |  0.99445  |  0.99445  | 0.99445  |   0.994448 |
    


```python
# Plot dan perbandingan semua hasil model
hasil_semua_model.plot(kind='bar', figsize=(10, 7)).legend(bbox_to_anchor=(1.0, 1.0))
plt.title('Akurasi Presisi Recall dan F1-score', fontsize=18)
plt.figure(facecolor='w')
plt.show()
```


    
![png](ColorSkim_AI_files/ColorSkim_AI_109_0.png)
    


Dari ringkasan di atas kita dapat melihat bahwa model_3 unggul di semua metrik dibandingkan dengan model lainnya.


```python
# Plot confusion matrix semua model
dua_kolom = int(len(MODEL)/2)
fig, axs = plt.subplots(2, dua_kolom, figsize=(10, 7))
for indeks, model in enumerate(MODEL):
    switcher = {
        0: [model_0_pred, model_0_metrik],
        1: [model_1_pred, model_1_metrik],
        2: [model_2_pred, model_2_metrik],
        3: [model_3_pred, model_3_metrik]
    }
    model_pred = switcher.get(indeks)[0]
    model_metrik = switcher.get(indeks)[1]
    axs[indeks if indeks <= 1 else indeks - 2] = plt.subplot(2, 
                                                             dua_kolom, 
                                                             indeks + 1)
    plot_conf_matrix(target_label=test_target_mnb,
                     prediksi_label=model_pred,
                     nama_model=MODEL[indeks],
                     akurasi=model_metrik['akurasi'],
                     label_titik_x=['bukan_warna', 'warna'],
                     label_titik_y=['bukan_warna', 'warna'])
fig.tight_layout(h_pad=2, w_pad=2)
plt.figure(facecolor='w')
plt.show()
```


    
![png](ColorSkim_AI_files/ColorSkim_AI_111_0.png)
    


Pada *confusion matrix* di atas kita dapat melihat juga bahwa pada model terbaik (model_3), pengujian model terhadap test_data menghasilkan

* 6,795 prediksi `bukan_warna` yang tepat
* 4,493 prediksi `warna` yang tepat
* 39 prediksi `bukan_warna` yang seharusnya adalah `warna`
* 24 prediksi `warna` yang seharusnya adalah `bukan_warna`

Proyeksi kata - kata dalam proses training ke dalam bidang 3 dimensi dapat dilihat di [TensorFlow Projector ColorSkim](http://projector.tensorflow.org/?config=https://gist.githubusercontent.com/johanesPao/4d32ee11e8610a7fc063009e76b08eea/raw/cd227e53ed26694fc050d427083566576ebfa481/template_projector_config.json). Proyeksi kata ke dalam bidang 3 dimensi ini menggunakan model_3 sebagai model dengan tingkat akurasi paling tinggi. Posisi kata dalam bidang 3 dimensi adalah setelah pembobotan dalam training model_3.

*Embedding dalam 3 Dimensi*

![dot_3d](images/dot_3d.png)

*Embedding `white` dalam 3 Dimensi*

![white_3d](images/white_3d.png)

*Embedding `black` dalam 3 Dimensi*

![black_3d](images/black_3d.png)

*Embedding `adidas` dalam 3 Dimensi*

![adidas_3d](images/adidas_3d.png)

*Embedding kata dalam 3 Dimensi*

![kata_3d](images/kata_3d.png)




```python
# Plot nilai residual dari model (kecuali untuk model_0)
fig, axs = plt.subplots(1, 3, figsize=(15, 5), sharey=True)
for indeks, model in enumerate(MODEL[1:]):
    switcher = {
        0: [model_1_metrik, model_1_pred_prob],
        1: [model_2_metrik, model_2_pred_prob],
        2: [model_3_metrik, model_3_pred_prob]
    }
    model_metrik = switcher.get(indeks)[0]
    model_pred_prob = switcher.get(indeks)[1]
    axs[indeks] = plt.subplot(1, len(MODEL[1:]), indeks + 1)
    residual_plot_logr(test_target,
                       MODEL[indeks + 1],
                       model_metrik['akurasi'],
                       model_pred_prob)
plt.tight_layout(h_pad=3, w_pad=3)
plt.figure(facecolor='w')
plt.show()
```


    
![png](ColorSkim_AI_files/ColorSkim_AI_113_0.png)
    


Sama seperti metrik - metrik lainnya dan semua perangkat evaluasi, pada grafik di atas kita dapat melihat bahwa model_3 memiliki residual yang paling minimum (sama dengan akurasi yang tinggi) untuk prediksinya, dimana kita masih dapat melihat sebagian besar residu pada model_3 masih berat di label `warna` dimana model memprediksi `bukan_warna`.
