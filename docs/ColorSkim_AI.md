# ColorSkim Machine Learning AI

Saat ini `item_description` untuk artikel ditulis dalam bentuk/format `nama_artikel + warna` dimana pemisahan `nama_artikel` dan `warna` bervariasi antar brand, beberapa menggunakan spasi, dash, garis miring dsbnya.

Pembelajaran mesin ini merupakan pembelajaran yang akan menerapkan jaringan saraf buatan (neural network) untuk mempelajari pola penulisan artikel yang bercampur dengan warna untuk mengekstrak warna saja dari artikel.

Akan dilakukan beberapa scenario modelling **Natural Language Procesing** untuk permasalahan *sequence to sequence* ini. Pada intinya kita akan membagi kalimat (`item_description`) berdasarkan kata per kata dan mengkategorisasikan masing - masing kata ke dalam satu dari dua kategori warna atau bukan_warna (logistik biner).




```python
# import modul
import tensorflow as tf
from tensorflow.python.client import device_lib
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import wandb as wb
from rahasia import API_KEY_WANDB
tf.config.run_functions_eagerly(True)
tf.data.experimental.enable_debug_mode()
```


```python
# cek ketersediaan GPU untuk modeling
# NVidia GeForce MX250 - office
# NVidia GeForce GTX1060 - home
device_lib.list_local_devices()[1]
```




    name: "/device:GPU:0"
    device_type: "GPU"
    memory_limit: 4852809728
    locality {
      bus_id: 1
      links {
      }
    }
    incarnation: 17804305938031545614
    physical_device_desc: "device: 0, name: NVIDIA GeForce GTX 1060, pci bus id: 0000:01:00.0, compute capability: 6.1"
    xla_global_id: 416903419




```python
# login ke wandb
wb.login(key=API_KEY_WANDB)
```

    Failed to detect the name of this notebook, you can set it manually with the WANDB_NOTEBOOK_NAME environment variable to enable code saving.
    [34m[1mwandb[0m: Currently logged in as: [33mjpao[0m ([33mpri-data[0m). Use [1m`wandb login --relogin`[0m to force relogin
    [34m[1mwandb[0m: [33mWARNING[0m If you're specifying your api key in code, ensure this code is not shared publicly.
    [34m[1mwandb[0m: [33mWARNING[0m Consider setting the WANDB_API_KEY environment variable, or running `wandb login` from the command line.
    [34m[1mwandb[0m: Appending key for api.wandb.ai to your netrc file: C:\Users\jPao/.netrc





    True



## Membaca data


```python
# Membaca data ke dalam DataFrame pandas
data = pd.read_csv('data/setengah_dataset_artikel.csv')
data[:10]
```




<div>
<style scoped>
    .dataframe tbody tr th:only-of-type {
        vertical-align: middle;
    }

    .dataframe tbody tr th {
        vertical-align: top;
    }

    .dataframe thead th {
        text-align: right;
    }
</style>
<table border="1" class="dataframe">
  <thead>
    <tr style="text-align: right;">
      <th></th>
      <th>nama_artikel</th>
      <th>kata</th>
      <th>label</th>
      <th>urut_kata</th>
      <th>total_kata</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>0</th>
      <td>ADISSAGE-BLACK/BLACK/RUNWHT</td>
      <td>ADISSAGE</td>
      <td>bukan_warna</td>
      <td>1</td>
      <td>4</td>
    </tr>
    <tr>
      <th>1</th>
      <td>ADISSAGE-BLACK/BLACK/RUNWHT</td>
      <td>BLACK</td>
      <td>warna</td>
      <td>2</td>
      <td>4</td>
    </tr>
    <tr>
      <th>2</th>
      <td>ADISSAGE-BLACK/BLACK/RUNWHT</td>
      <td>BLACK</td>
      <td>warna</td>
      <td>3</td>
      <td>4</td>
    </tr>
    <tr>
      <th>3</th>
      <td>ADISSAGE-BLACK/BLACK/RUNWHT</td>
      <td>RUNWHT</td>
      <td>warna</td>
      <td>4</td>
      <td>4</td>
    </tr>
    <tr>
      <th>4</th>
      <td>ADISSAGE-N.NAVY/N.NAVY/RUNWHT</td>
      <td>ADISSAGE</td>
      <td>bukan_warna</td>
      <td>1</td>
      <td>4</td>
    </tr>
    <tr>
      <th>5</th>
      <td>ADISSAGE-N.NAVY/N.NAVY/RUNWHT</td>
      <td>N.NAVY</td>
      <td>warna</td>
      <td>2</td>
      <td>4</td>
    </tr>
    <tr>
      <th>6</th>
      <td>ADISSAGE-N.NAVY/N.NAVY/RUNWHT</td>
      <td>N.NAVY</td>
      <td>warna</td>
      <td>3</td>
      <td>4</td>
    </tr>
    <tr>
      <th>7</th>
      <td>ADISSAGE-N.NAVY/N.NAVY/RUNWHT</td>
      <td>RUNWHT</td>
      <td>warna</td>
      <td>4</td>
      <td>4</td>
    </tr>
    <tr>
      <th>8</th>
      <td>3 STRIPE D 29.5-BASKETBALL NATURAL</td>
      <td>3</td>
      <td>bukan_warna</td>
      <td>1</td>
      <td>6</td>
    </tr>
    <tr>
      <th>9</th>
      <td>3 STRIPE D 29.5-BASKETBALL NATURAL</td>
      <td>STRIPE</td>
      <td>bukan_warna</td>
      <td>2</td>
      <td>6</td>
    </tr>
  </tbody>
</table>
</div>



## Eksplorasi data


```python
# distribusi label dalam data
data['label'].value_counts()
```




    bukan_warna    34174
    warna          22577
    Name: label, dtype: int64



## Konversi data ke dalam train dan test


```python
from sklearn.model_selection import train_test_split
train_kata, test_kata, train_label, test_label = train_test_split(data['kata'].to_numpy(), data['label'].to_numpy(), test_size=0.2, random_state=42)
train_kata[:5], test_kata[:5], train_label[:5], test_label[:5]
```




    (array(['INVIS', 'SOLRED', 'JR', 'REACT', 'WHITE'], dtype=object),
     array(['6', 'GA', 'NIKE', 'BLUE', 'BLACK'], dtype=object),
     array(['bukan_warna', 'warna', 'bukan_warna', 'bukan_warna', 'warna'],
           dtype=object),
     array(['bukan_warna', 'bukan_warna', 'bukan_warna', 'warna', 'warna'],
           dtype=object))



## Konversi label ke dalam numerik


```python
from sklearn.preprocessing import LabelEncoder
label_encoder = LabelEncoder()
train_label_encode = label_encoder.fit_transform(train_label)
test_label_encode = label_encoder.transform(test_label)
train_label_encode[:5], test_label_encode[:5]
```




    (array([0, 1, 0, 0, 1]), array([0, 0, 0, 1, 1]))



## Model 0: model dasar


```python
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.naive_bayes import MultinomialNB
from sklearn.pipeline import Pipeline

# Membuat pipeline untuk mengubah kata ke dalam tf-idf
model_0 = Pipeline([
    ("tf-idf", TfidfVectorizer()),
    ("clf", MultinomialNB())
])

# Fit pipeline dengan data training
model_0.fit(X=train_kata, y=train_label_encode)
```




<style>#sk-container-id-1 {color: black;background-color: white;}#sk-container-id-1 pre{padding: 0;}#sk-container-id-1 div.sk-toggleable {background-color: white;}#sk-container-id-1 label.sk-toggleable__label {cursor: pointer;display: block;width: 100%;margin-bottom: 0;padding: 0.3em;box-sizing: border-box;text-align: center;}#sk-container-id-1 label.sk-toggleable__label-arrow:before {content: "▸";float: left;margin-right: 0.25em;color: #696969;}#sk-container-id-1 label.sk-toggleable__label-arrow:hover:before {color: black;}#sk-container-id-1 div.sk-estimator:hover label.sk-toggleable__label-arrow:before {color: black;}#sk-container-id-1 div.sk-toggleable__content {max-height: 0;max-width: 0;overflow: hidden;text-align: left;background-color: #f0f8ff;}#sk-container-id-1 div.sk-toggleable__content pre {margin: 0.2em;color: black;border-radius: 0.25em;background-color: #f0f8ff;}#sk-container-id-1 input.sk-toggleable__control:checked~div.sk-toggleable__content {max-height: 200px;max-width: 100%;overflow: auto;}#sk-container-id-1 input.sk-toggleable__control:checked~label.sk-toggleable__label-arrow:before {content: "▾";}#sk-container-id-1 div.sk-estimator input.sk-toggleable__control:checked~label.sk-toggleable__label {background-color: #d4ebff;}#sk-container-id-1 div.sk-label input.sk-toggleable__control:checked~label.sk-toggleable__label {background-color: #d4ebff;}#sk-container-id-1 input.sk-hidden--visually {border: 0;clip: rect(1px 1px 1px 1px);clip: rect(1px, 1px, 1px, 1px);height: 1px;margin: -1px;overflow: hidden;padding: 0;position: absolute;width: 1px;}#sk-container-id-1 div.sk-estimator {font-family: monospace;background-color: #f0f8ff;border: 1px dotted black;border-radius: 0.25em;box-sizing: border-box;margin-bottom: 0.5em;}#sk-container-id-1 div.sk-estimator:hover {background-color: #d4ebff;}#sk-container-id-1 div.sk-parallel-item::after {content: "";width: 100%;border-bottom: 1px solid gray;flex-grow: 1;}#sk-container-id-1 div.sk-label:hover label.sk-toggleable__label {background-color: #d4ebff;}#sk-container-id-1 div.sk-serial::before {content: "";position: absolute;border-left: 1px solid gray;box-sizing: border-box;top: 0;bottom: 0;left: 50%;z-index: 0;}#sk-container-id-1 div.sk-serial {display: flex;flex-direction: column;align-items: center;background-color: white;padding-right: 0.2em;padding-left: 0.2em;position: relative;}#sk-container-id-1 div.sk-item {position: relative;z-index: 1;}#sk-container-id-1 div.sk-parallel {display: flex;align-items: stretch;justify-content: center;background-color: white;position: relative;}#sk-container-id-1 div.sk-item::before, #sk-container-id-1 div.sk-parallel-item::before {content: "";position: absolute;border-left: 1px solid gray;box-sizing: border-box;top: 0;bottom: 0;left: 50%;z-index: -1;}#sk-container-id-1 div.sk-parallel-item {display: flex;flex-direction: column;z-index: 1;position: relative;background-color: white;}#sk-container-id-1 div.sk-parallel-item:first-child::after {align-self: flex-end;width: 50%;}#sk-container-id-1 div.sk-parallel-item:last-child::after {align-self: flex-start;width: 50%;}#sk-container-id-1 div.sk-parallel-item:only-child::after {width: 0;}#sk-container-id-1 div.sk-dashed-wrapped {border: 1px dashed gray;margin: 0 0.4em 0.5em 0.4em;box-sizing: border-box;padding-bottom: 0.4em;background-color: white;}#sk-container-id-1 div.sk-label label {font-family: monospace;font-weight: bold;display: inline-block;line-height: 1.2em;}#sk-container-id-1 div.sk-label-container {text-align: center;}#sk-container-id-1 div.sk-container {/* jupyter's `normalize.less` sets `[hidden] { display: none; }` but bootstrap.min.css set `[hidden] { display: none !important; }` so we also need the `!important` here to be able to override the default hidden behavior on the sphinx rendered scikit-learn.org. See: https://github.com/scikit-learn/scikit-learn/issues/21755 */display: inline-block !important;position: relative;}#sk-container-id-1 div.sk-text-repr-fallback {display: none;}</style><div id="sk-container-id-1" class="sk-top-container"><div class="sk-text-repr-fallback"><pre>Pipeline(steps=[(&#x27;tf-idf&#x27;, TfidfVectorizer()), (&#x27;clf&#x27;, MultinomialNB())])</pre><b>In a Jupyter environment, please rerun this cell to show the HTML representation or trust the notebook. <br />On GitHub, the HTML representation is unable to render, please try loading this page with nbviewer.org.</b></div><div class="sk-container" hidden><div class="sk-item sk-dashed-wrapped"><div class="sk-label-container"><div class="sk-label sk-toggleable"><input class="sk-toggleable__control sk-hidden--visually" id="sk-estimator-id-1" type="checkbox" ><label for="sk-estimator-id-1" class="sk-toggleable__label sk-toggleable__label-arrow">Pipeline</label><div class="sk-toggleable__content"><pre>Pipeline(steps=[(&#x27;tf-idf&#x27;, TfidfVectorizer()), (&#x27;clf&#x27;, MultinomialNB())])</pre></div></div></div><div class="sk-serial"><div class="sk-item"><div class="sk-estimator sk-toggleable"><input class="sk-toggleable__control sk-hidden--visually" id="sk-estimator-id-2" type="checkbox" ><label for="sk-estimator-id-2" class="sk-toggleable__label sk-toggleable__label-arrow">TfidfVectorizer</label><div class="sk-toggleable__content"><pre>TfidfVectorizer()</pre></div></div></div><div class="sk-item"><div class="sk-estimator sk-toggleable"><input class="sk-toggleable__control sk-hidden--visually" id="sk-estimator-id-3" type="checkbox" ><label for="sk-estimator-id-3" class="sk-toggleable__label sk-toggleable__label-arrow">MultinomialNB</label><div class="sk-toggleable__content"><pre>MultinomialNB()</pre></div></div></div></div></div></div></div>




```python
# Evaluasi model_0 pada data test
model_0.score(X=test_kata, y=test_label_encode)
```




    0.9935688485595983




```python
# Membuat prediksi menggunakan data test
pred_model_0 = model_0.predict(test_kata)
pred_model_0
```




    array([0, 0, 0, ..., 1, 1, 0])




```python
# Membuat fungsi dasar untuk menghitung accuray, precision, recall, f1-score
from sklearn.metrics import accuracy_score, precision_recall_fscore_support
def hitung_metrik(target, prediksi):
    """
    Menghitung accuracy, precision, recall dan f1-score dari model klasifikasi biner
    
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
model_0_metrik = hitung_metrik(target=test_label_encode, 
                               prediksi=pred_model_0)
model_0_metrik
```




    {'akurasi': 0.9935688485595983,
     'presisi': 0.9935690437085363,
     'recall': 0.9935688485595983,
     'f1-score': 0.9935671438217326}



## Menyiapkan data (text) untuk model deep sequence

### Text Vectorizer Layer


```python
# jumlah data (kata) dalam train_data
len(train_kata)
```




    45400




```python
# jumlah data unik (kata unik) dalam train_kata
jumlah_kata_train = len(np.unique(train_kata))
jumlah_kata_train
```




    2940




```python
# Membuat text vectorizer
from tensorflow.keras.layers import TextVectorization
vectorizer_kata = TextVectorization(max_tokens=jumlah_kata_train,
                                    output_sequence_length=1,
                                    standardize='lower')
```


```python
# Mengadaptaasikan text vectorizer ke dalam train_kata
vectorizer_kata.adapt(train_kata)
```


```python
# Test vectorizer kata
import random
target_kata = random.choice(train_kata)
print(f'Kata:\n{target_kata}\n')
print(f'Kata setelah vektorisasi:\n{vectorizer_kata([target_kata])}')
```

    Kata:
    5
    
    Kata setelah vektorisasi:
    [[47]]



```python
vectorizer_kata.get_config()
```




    {'name': 'text_vectorization',
     'trainable': True,
     'batch_input_shape': (None,),
     'dtype': 'string',
     'max_tokens': 2940,
     'standardize': 'lower',
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
# Jumlah vocabulary dalam vectorizer_kata
jumlah_vocab = vectorizer_kata.get_vocabulary()
len(jumlah_vocab)
```




    2938



### Membuat Text Embedding


```python
# Membuat text embedding layer
from tensorflow.keras.layers import Embedding
kata_embed = Embedding(input_dim=len(jumlah_vocab),
                       output_dim=64,
                       mask_zero=True,
                       name='layer_token_embedding')
```


```python
# Contoh vectorizer dan embedding
print(f'Kata sebelum vektorisasi:\n{target_kata}\n')
kata_tervektor = vectorizer_kata([target_kata])
print(f'\nKata sesudah vektorisasi (sebelum embedding):\n{kata_tervektor}\n')
kata_terembed = kata_embed(kata_tervektor)
print(f'\nKata setelah embedding:\n{kata_terembed}\n')
print(f'Shape dari kata setelah embedding:\n{kata_terembed.shape}')
```

    Kata sebelum vektorisasi:
    5
    
    
    Kata sesudah vektorisasi (sebelum embedding):
    [[47]]
    
    
    Kata setelah embedding:
    [[[-0.00258959  0.04990998 -0.04441559 -0.02205954 -0.03956609
       -0.01799051 -0.02189567 -0.01263437  0.04183039 -0.029005
       -0.02622869  0.02765231 -0.04219884  0.00494075 -0.04544568
       -0.02989398  0.0400069  -0.03978921  0.03846495  0.01232683
        0.0007933  -0.02543398  0.00365927 -0.01668191  0.01294837
        0.00481149 -0.02894466 -0.02507013  0.04588108  0.0405197
       -0.01860956 -0.04605418 -0.03735595 -0.00483001 -0.00458407
       -0.01903701  0.03657718  0.00718967  0.01976332 -0.02251861
        0.00589142 -0.00238026 -0.01294807  0.02448275 -0.02560136
        0.033471    0.0314525   0.00829456 -0.03963711  0.03865555
        0.01974393  0.00157628 -0.01058214  0.04987845 -0.01076373
       -0.02952557  0.03197506 -0.01180102 -0.02673483  0.03148599
       -0.04402238  0.01858581 -0.01718934 -0.00017769]]]
    
    Shape dari kata setelah embedding:
    (1, 1, 64)


### Membuat TensorFlow Dataset


```python
# Membuat TensorFlow dataset
train_dataset = tf.data.Dataset.from_tensor_slices((train_kata, train_label_encode))
test_dataset = tf.data.Dataset.from_tensor_slices((test_kata, test_label_encode))

train_dataset
```




    <TensorSliceDataset element_spec=(TensorSpec(shape=(), dtype=tf.string, name=None), TensorSpec(shape=(), dtype=tf.int32, name=None))>




```python
# Membuat TensorSliceDataset menjadi prefetched dataset
train_dataset = train_dataset.batch(32).prefetch(tf.data.AUTOTUNE)
test_dataset = test_dataset.batch(32).prefetch(tf.data.AUTOTUNE)

train_dataset
```




    <BatchDataset element_spec=(TensorSpec(shape=(None,), dtype=tf.string, name=None), TensorSpec(shape=(None,), dtype=tf.int32, name=None))>



## Model 1: Conv1D dengan embedding


```python
# Membuat model_1 dengan layer Conv1D dari kata yang divektorisasi dan di-embed
from tensorflow.keras import layers

inputs = layers.Input(shape=(1,), dtype=tf.string, name='layer_input')
layer_vektor = vectorizer_kata(inputs)
layer_embed = kata_embed(layer_vektor)
x = layers.Conv1D(filters=64, kernel_size=5, padding='same', activation='relu')(layer_embed)
x = layers.GlobalMaxPooling1D(name='layer_max_pool')(x)
outputs = layers.Dense(units=1, activation='sigmoid', name='layer_output')(x)
model_1 = tf.keras.Model(inputs=inputs, outputs=outputs, name='model_1_Conv1D_embed')

# Compile
model_1.compile(loss=tf.keras.losses.BinaryCrossentropy(),
                optimizer=tf.keras.optimizers.Adam(),
                metrics=['accuracy'])
```


```python
# Ringkasa model_1
model_1.summary()
```

    Model: "model_1_Conv1D_embed"
    _________________________________________________________________
     Layer (type)                Output Shape              Param #   
    =================================================================
     layer_input (InputLayer)    [(None, 1)]               0         
                                                                     
     text_vectorization (TextVec  (None, 1)                0         
     torization)                                                     
                                                                     
     layer_token_embedding (Embe  (None, 1, 64)            188032    
     dding)                                                          
                                                                     
     conv1d (Conv1D)             (None, 1, 64)             20544     
                                                                     
     layer_max_pool (GlobalMaxPo  (None, 64)               0         
     oling1D)                                                        
                                                                     
     layer_output (Dense)        (None, 1)                 65        
                                                                     
    =================================================================
    Total params: 208,641
    Trainable params: 208,641
    Non-trainable params: 0
    _________________________________________________________________



```python
# Plot model_1
from tensorflow.keras.utils import plot_model
plot_model(model_1, show_shapes=True)
```

    You must install pydot (`pip install pydot`) and install graphviz (see instructions at https://graphviz.gitlab.io/download/) for plot_model/model_to_dot to work.



```python
# import WandbCallback
from wandb.keras import WandbCallback

# Setup wandb init dan config
wb.init(project='ColorSkim',
        entity='jpao',
        name='model_1_Conv1D_embed',
        config={'epochs': 3,
                'n_layers': len(model_1.layers)})

# Fit model_1
hist_model_1 = model_1.fit(train_dataset,
                           epochs=wb.config.epochs,
                           validation_data=test_dataset,
                           callbacks=[WandbCallback()])
```

    [34m[1mwandb[0m: Currently logged in as: [33mjpao[0m. Use [1m`wandb login --relogin`[0m to force relogin



Tracking run with wandb version 0.12.21



Run data is saved locally in <code>d:\ColorSkim\wandb\run-20220707_112133-15oqfl8s</code>



Syncing run <strong><a href="https://wandb.ai/jpao/ColorSkim/runs/15oqfl8s" target="_blank">model_1_Conv1D_embed</a></strong> to <a href="https://wandb.ai/jpao/ColorSkim" target="_blank">Weights & Biases</a> (<a href="https://wandb.me/run" target="_blank">docs</a>)<br/>


    [34m[1mwandb[0m: [33mWARNING[0m The save_model argument by default saves the model in the HDF5 format that cannot save custom objects like subclassed models and custom layers. This behavior will be deprecated in a future release in favor of the SavedModel format. Meanwhile, the HDF5 model is saved as W&B files and the SavedModel as W&B Artifacts.


    WARNING:tensorflow:Issue encountered when serializing table_initializer.
    Type is unsupported, or the types of the items don't match field type in CollectionDef. Note this is a warning and probably safe to ignore.
    'NoneType' object has no attribute 'name'
    WARNING:tensorflow:From c:\Users\jPao\miniconda3\envs\tf-py39\lib\site-packages\tensorflow\python\profiler\internal\flops_registry.py:138: tensor_shape_from_node_def_name (from tensorflow.python.framework.graph_util_impl) is deprecated and will be removed in a future version.
    Instructions for updating:
    Use `tf.compat.v1.graph_util.tensor_shape_from_node_def_name`
    Epoch 1/3
    1419/1419 [==============================] - ETA: 0s - loss: 0.0729 - accuracy: 0.9841

    [34m[1mwandb[0m: [32m[41mERROR[0m Can't save model in the h5py format. The model will be saved as W&B Artifacts in the SavedModel format.
    WARNING:absl:Found untraced functions such as adapt_step while saving (showing 1 of 1). These functions will not be directly callable after loading.


    INFO:tensorflow:Assets written to: d:\ColorSkim\wandb\run-20220707_112133-15oqfl8s\files\model-best\assets


    INFO:tensorflow:Assets written to: d:\ColorSkim\wandb\run-20220707_112133-15oqfl8s\files\model-best\assets
    [34m[1mwandb[0m: Adding directory to artifact (d:\ColorSkim\wandb\run-20220707_112133-15oqfl8s\files\model-best)... Done. 0.2s


    1419/1419 [==============================] - 104s 71ms/step - loss: 0.0729 - accuracy: 0.9841 - val_loss: 0.0273 - val_accuracy: 0.9930 - _timestamp: 1657167802.0000 - _runtime: 109.0000
    Epoch 2/3
    1419/1419 [==============================] - ETA: 0s - loss: 0.0178 - accuracy: 0.9952

    WARNING:absl:Found untraced functions such as adapt_step while saving (showing 1 of 1). These functions will not be directly callable after loading.


    INFO:tensorflow:Assets written to: d:\ColorSkim\wandb\run-20220707_112133-15oqfl8s\files\model-best\assets


    INFO:tensorflow:Assets written to: d:\ColorSkim\wandb\run-20220707_112133-15oqfl8s\files\model-best\assets
    [34m[1mwandb[0m: Adding directory to artifact (d:\ColorSkim\wandb\run-20220707_112133-15oqfl8s\files\model-best)... Done. 0.1s


    1419/1419 [==============================] - 102s 72ms/step - loss: 0.0178 - accuracy: 0.9952 - val_loss: 0.0264 - val_accuracy: 0.9931 - _timestamp: 1657167905.0000 - _runtime: 212.0000
    Epoch 3/3
    1419/1419 [==============================] - ETA: 0s - loss: 0.0159 - accuracy: 0.9956

    WARNING:absl:Found untraced functions such as adapt_step while saving (showing 1 of 1). These functions will not be directly callable after loading.


    INFO:tensorflow:Assets written to: d:\ColorSkim\wandb\run-20220707_112133-15oqfl8s\files\model-best\assets


    INFO:tensorflow:Assets written to: d:\ColorSkim\wandb\run-20220707_112133-15oqfl8s\files\model-best\assets
    [34m[1mwandb[0m: Adding directory to artifact (d:\ColorSkim\wandb\run-20220707_112133-15oqfl8s\files\model-best)... Done. 0.1s


    1419/1419 [==============================] - 99s 70ms/step - loss: 0.0159 - accuracy: 0.9956 - val_loss: 0.0261 - val_accuracy: 0.9936 - _timestamp: 1657168005.0000 - _runtime: 312.0000



```python
# Evaluasi model_1
model_1.evaluate(test_dataset)
```

    355/355 [==============================] - 14s 39ms/step - loss: 0.0261 - accuracy: 0.9936





    [0.026110293343663216, 0.9935688376426697]




```python
# Membuat prediksi berdasarkan model_1
model_1_pred_prob = model_1.predict(test_dataset)
model_1_pred_prob[:10]
```




    array([[4.4162782e-05],
           [7.1573093e-05],
           [4.8019815e-06],
           [9.9727100e-01],
           [9.9964142e-01],
           [1.0789679e-05],
           [1.4505963e-04],
           [4.8167007e-03],
           [1.1653579e-04],
           [9.9448615e-01]], dtype=float32)




```python
# Mengkonversi model_1_pred_prob ke dalam label
model_1_pred = tf.squeeze(tf.round(model_1_pred_prob))
model_1_pred
```




    <tf.Tensor: shape=(11351,), dtype=float32, numpy=array([0., 0., 0., ..., 1., 1., 0.], dtype=float32)>




```python
# Menghitung metriks dari model_1
model_1_metrik = hitung_metrik(target=test_label_encode,
                              prediksi=model_1_pred)
model_1_metrik
```




    {'akurasi': 0.9935688485595983,
     'presisi': 0.9935703064543551,
     'recall': 0.9935688485595983,
     'f1-score': 0.9935666849784249}



## Model 2: Transfer Learning pretrained feature exraction menggunakan Universal Sentence Encoder (USE)


```python
# Download pretrained USE
import tensorflow_hub as hub
tf_hub_embedding = hub.KerasLayer('https://tfhub.dev/google/universal-sentence-encoder/4',
                                  trainable=False,
                                  name='universal_sentence_encoder')
```


```python
# Melakukan tes pretrained embedding pada contoh kata
kata_acak = random.choice(train_kata)
print(f'Kata acak:\n {kata_acak}')
kata_embed_pretrain = tf_hub_embedding([kata_acak])
print(f'\nKata setelah embed dengan USE:\n{kata_embed_pretrain[0][:30]}\n')
print(f'Panjang dari kata setelah embedding: {len(kata_embed_pretrain[0])}')
```

    Kata acak:
     BLACK
    
    Kata setelah embed dengan USE:
    [-0.05150617 -0.00596834  0.04953347  0.0329339   0.02531563 -0.0103454
      0.05544865  0.01774512 -0.00898479  0.04252742 -0.02970365 -0.03763022
     -0.00209826 -0.05936718 -0.0276504  -0.03143836  0.0275419   0.05507992
     -0.05129641 -0.02967714 -0.00617658  0.0467336  -0.00554184  0.00347361
     -0.04157941  0.01770764 -0.0202241  -0.06021477  0.01153699  0.03923866]
    
    Panjang dari kata setelah embedding: 512



```python
# Membuat model_2 menggunakan USE
inputs = layers.Input(shape=[], dtype=tf.string, name='layer_input')
layer_embed_pretrained = tf_hub_embedding(inputs)
x = layers.Conv1D(filters=64, kernel_size=5, padding='same', activation='relu')(tf.expand_dims(layer_embed_pretrained, axis=-1))
x = layers.GlobalMaxPooling1D(name='layer_max_pooling')(x)
outputs = layers.Dense(1, activation='sigmoid', name='layer_output')(x)
model_2 = tf.keras.Model(inputs=inputs, outputs=outputs, name='model_2_Conv1D_USE_embed')
```


```python
# Ringkasan model_2
model_2.summary()
```

    Model: "model_2_Conv1D_USE_embed"
    _________________________________________________________________
     Layer (type)                Output Shape              Param #   
    =================================================================
     layer_input (InputLayer)    [(None,)]                 0         
                                                                     
     universal_sentence_encoder   (None, 512)              256797824 
     (KerasLayer)                                                    
                                                                     
     tf.expand_dims (TFOpLambda)  (None, 512, 1)           0         
                                                                     
     conv1d (Conv1D)             (None, 512, 64)           384       
                                                                     
     layer_max_pooling (GlobalMa  (None, 64)               0         
     xPooling1D)                                                     
                                                                     
     layer_output (Dense)        (None, 1)                 65        
                                                                     
    =================================================================
    Total params: 256,798,273
    Trainable params: 449
    Non-trainable params: 256,797,824
    _________________________________________________________________



```python
# Plot model_2
plot_model(model_2, show_shapes=True)
```




    
![png](output_47_0.png)
    




```python
# Compile model_2
model_2.compile(loss=tf.keras.losses.BinaryCrossentropy(),
                optimizer=tf.keras.optimizers.Adam(),
                metrics=['accuracy'])
```


```python
# Setup wandb init dan config
wb.init(project='ColorSkim',
        entity='jpao',
        name='model_2_Conv1D_USE_embed',
        config={'epochs': 3,
                'n_layers': len(model_2.layers)})

# Fit model_2
hist_model_2 = model_2.fit(train_dataset,
                           epochs=wb.config.epochs,
                           validation_data=test_dataset,
                           callbacks=[WandbCallback()])
```


Finishing last run (ID:15oqfl8s) before initializing another...



Waiting for W&B process to finish... <strong style="color:green">(success).</strong>



<style>
    table.wandb td:nth-child(1) { padding: 0 10px; text-align: left ; width: auto;} td:nth-child(2) {text-align: left ; width: 100%}
    .wandb-row { display: flex; flex-direction: row; flex-wrap: wrap; justify-content: flex-start; width: 100% }
    .wandb-col { display: flex; flex-direction: column; flex-basis: 100%; flex: 1; padding: 10px; }
    </style>
<div class="wandb-row"><div class="wandb-col"><h3>Run history:</h3><br/><table class="wandb"><tr><td>accuracy</td><td>▁██</td></tr><tr><td>epoch</td><td>▁▅█</td></tr><tr><td>loss</td><td>█▁▁</td></tr><tr><td>val_accuracy</td><td>▁▂█</td></tr><tr><td>val_loss</td><td>█▃▁</td></tr></table><br/></div><div class="wandb-col"><h3>Run summary:</h3><br/><table class="wandb"><tr><td>GFLOPS</td><td>0.0</td></tr><tr><td>accuracy</td><td>0.99562</td></tr><tr><td>best_epoch</td><td>2</td></tr><tr><td>best_val_loss</td><td>0.02611</td></tr><tr><td>epoch</td><td>2</td></tr><tr><td>loss</td><td>0.01587</td></tr><tr><td>val_accuracy</td><td>0.99357</td></tr><tr><td>val_loss</td><td>0.02611</td></tr></table><br/></div></div>



Synced <strong style="color:#cdcd00">model_1_Conv1D_embed</strong>: <a href="https://wandb.ai/jpao/ColorSkim/runs/15oqfl8s" target="_blank">https://wandb.ai/jpao/ColorSkim/runs/15oqfl8s</a><br/>Synced 5 W&B file(s), 1 media file(s), 10 artifact file(s) and 1 other file(s)



Find logs at: <code>.\wandb\run-20220707_112133-15oqfl8s\logs</code>



Successfully finished last run (ID:15oqfl8s). Initializing new run:<br/>



Tracking run with wandb version 0.12.21



Run data is saved locally in <code>d:\ColorSkim\wandb\run-20220707_112718-298uq3pm</code>



Syncing run <strong><a href="https://wandb.ai/jpao/ColorSkim/runs/298uq3pm" target="_blank">model_2_Conv1D_USE_embed</a></strong> to <a href="https://wandb.ai/jpao/ColorSkim" target="_blank">Weights & Biases</a> (<a href="https://wandb.me/run" target="_blank">docs</a>)<br/>


    [34m[1mwandb[0m: [33mWARNING[0m Unable to compute FLOPs for this model.


    Epoch 1/3
    1419/1419 [==============================] - 157s 108ms/step - loss: 0.5851 - accuracy: 0.7125 - val_loss: 0.4620 - val_accuracy: 0.9055 - _timestamp: 1657168206.0000 - _runtime: 162.0000
    Epoch 2/3
    1419/1419 [==============================] - 150s 106ms/step - loss: 0.3688 - accuracy: 0.9020 - val_loss: 0.3056 - val_accuracy: 0.9027 - _timestamp: 1657168358.0000 - _runtime: 314.0000
    Epoch 3/3
    1419/1419 [==============================] - 154s 109ms/step - loss: 0.2764 - accuracy: 0.9056 - val_loss: 0.2576 - val_accuracy: 0.9009 - _timestamp: 1657168511.0000 - _runtime: 467.0000



```python
# Evaluate model_2
model_2.evaluate(test_dataset)
```

    355/355 [==============================] - 17s 47ms/step - loss: 0.2576 - accuracy: 0.9009





    [0.25761494040489197, 0.9008898138999939]




```python
# Membuat prediksi dengan model_2
model_2_pred_prob = model_2.predict(test_dataset)
model_2_pred_prob[:10]
```




    array([[0.10366567],
           [0.07147089],
           [0.0694409 ],
           [0.91198367],
           [0.95933306],
           [0.13955557],
           [0.05977627],
           [0.14464849],
           [0.07072161],
           [0.6322903 ]], dtype=float32)




```python
# Mengkonversi model_2 menjadi label format
model_2_pred = tf.squeeze(tf.round(model_2_pred_prob))
model_2_pred
```




    <tf.Tensor: shape=(11351,), dtype=float32, numpy=array([0., 0., 0., ..., 0., 1., 0.], dtype=float32)>




```python
# Menghitung hasil metrik dari model_2
model_2_hasil = hitung_metrik(target=test_label_encode,
                              prediksi=model_2_pred)
model_2_hasil
```




    {'akurasi': 0.9008897894458638,
     'presisi': 0.9008754342105381,
     'recall': 0.9008897894458638,
     'f1-score': 0.9003678474557234}



## Model 3: Menggunakan positional kata dan custom embed dan concatenate layer


```python
# Test prediksi dengan model_1 (model_1_Conv1D_embed)
class_list = ['bukan_warna', 'warna']
article = 'PUMA XTG WOVEN PANTS PUMA BLACK-PUMA WHITE'
article_list = article.replace("-"," ").split()
model_test = tf.squeeze(tf.round(model_1.predict(article.replace("-"," ").split())))
for i in range(0, len(article_list)):
    print(f'Kata: {article_list[i]}\nPrediksi: {class_list[int(model_test[i])]}\n\n')
```

    Kata: PUMA
    Prediksi: bukan_warna
    
    
    Kata: XTG
    Prediksi: bukan_warna
    
    
    Kata: WOVEN
    Prediksi: bukan_warna
    
    
    Kata: PANTS
    Prediksi: bukan_warna
    
    
    Kata: PUMA
    Prediksi: bukan_warna
    
    
    Kata: BLACK
    Prediksi: warna
    
    
    Kata: PUMA
    Prediksi: bukan_warna
    
    
    Kata: WHITE
    Prediksi: warna
    
    



```python
model_test
```




    array([[1.5349576e-03],
           [7.1742781e-04],
           [1.4221472e-04],
           [2.2269943e-04],
           [1.5349576e-03],
           [9.9965119e-01],
           [1.5349576e-03],
           [9.9896502e-01]], dtype=float32)




```python

```