<a href="https://colab.research.google.com/github/johanesPao/ColorSkim/blob/main/ColorSkim_AI.ipynb" target="_parent"><img src="https://colab.research.google.com/assets/colab-badge.svg" alt="Open In Colab"/></a>

# ColorSkim Machine Learning

Saat ini `item_description` untuk artikel ditulis dalam bentuk/format `nama_artikel + warna` dimana pemisahan `nama_artikel` dan `warna` bervariasi antar brand, beberapa menggunakan spasi, dash, garis miring dsbnya.

Pembelajaran mesin ini merupakan pembelajaran yang akan menerapkan jaringan saraf buatan (neural network) untuk mempelajari pola penulisan artikel yang bercampur dengan warna untuk mengekstrak warna saja dari artikel.

Akan dilakukan beberapa scenario modeling **Natural Language Processing** untuk permasalahan *sequence to sequence* ini. Pada intinya kita akan membagi kalimat (`item_description`) berdasarkan kata per kata dan mengkategorikan masing - masing kata ke dalam satu dari 2 kategori warna atau bukan warna (logistik biner).


```python
# Install wandb (weights and biases)
!pip install wandb

# Import modul
import tensorflow as tf
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import wandb as wb
```

    Looking in indexes: https://pypi.org/simple, https://us-python.pkg.dev/colab-wheels/public/simple/
    Requirement already satisfied: wandb in /usr/local/lib/python3.7/dist-packages (0.12.19)
    Requirement already satisfied: PyYAML in /usr/local/lib/python3.7/dist-packages (from wandb) (3.13)
    Requirement already satisfied: docker-pycreds>=0.4.0 in /usr/local/lib/python3.7/dist-packages (from wandb) (0.4.0)
    Requirement already satisfied: shortuuid>=0.5.0 in /usr/local/lib/python3.7/dist-packages (from wandb) (1.0.9)
    Requirement already satisfied: psutil>=5.0.0 in /usr/local/lib/python3.7/dist-packages (from wandb) (5.4.8)
    Requirement already satisfied: six>=1.13.0 in /usr/local/lib/python3.7/dist-packages (from wandb) (1.15.0)
    Requirement already satisfied: setproctitle in /usr/local/lib/python3.7/dist-packages (from wandb) (1.2.3)
    Requirement already satisfied: setuptools in /usr/local/lib/python3.7/dist-packages (from wandb) (57.4.0)
    Requirement already satisfied: GitPython>=1.0.0 in /usr/local/lib/python3.7/dist-packages (from wandb) (3.1.27)
    Requirement already satisfied: protobuf<4.0dev,>=3.12.0 in /usr/local/lib/python3.7/dist-packages (from wandb) (3.17.3)
    Requirement already satisfied: pathtools in /usr/local/lib/python3.7/dist-packages (from wandb) (0.1.2)
    Requirement already satisfied: requests<3,>=2.0.0 in /usr/local/lib/python3.7/dist-packages (from wandb) (2.23.0)
    Requirement already satisfied: sentry-sdk>=1.0.0 in /usr/local/lib/python3.7/dist-packages (from wandb) (1.6.0)
    Requirement already satisfied: promise<3,>=2.0 in /usr/local/lib/python3.7/dist-packages (from wandb) (2.3)
    Requirement already satisfied: Click!=8.0.0,>=7.0 in /usr/local/lib/python3.7/dist-packages (from wandb) (7.1.2)
    Requirement already satisfied: typing-extensions>=3.7.4.3 in /usr/local/lib/python3.7/dist-packages (from GitPython>=1.0.0->wandb) (4.1.1)
    Requirement already satisfied: gitdb<5,>=4.0.1 in /usr/local/lib/python3.7/dist-packages (from GitPython>=1.0.0->wandb) (4.0.9)
    Requirement already satisfied: smmap<6,>=3.0.1 in /usr/local/lib/python3.7/dist-packages (from gitdb<5,>=4.0.1->GitPython>=1.0.0->wandb) (5.0.0)
    Requirement already satisfied: idna<3,>=2.5 in /usr/local/lib/python3.7/dist-packages (from requests<3,>=2.0.0->wandb) (2.10)
    Requirement already satisfied: urllib3!=1.25.0,!=1.25.1,<1.26,>=1.21.1 in /usr/local/lib/python3.7/dist-packages (from requests<3,>=2.0.0->wandb) (1.24.3)
    Requirement already satisfied: chardet<4,>=3.0.2 in /usr/local/lib/python3.7/dist-packages (from requests<3,>=2.0.0->wandb) (3.0.4)
    Requirement already satisfied: certifi>=2017.4.17 in /usr/local/lib/python3.7/dist-packages (from requests<3,>=2.0.0->wandb) (2022.6.15)
    


```python
# wandb login
wb.login(key='924d78a46727fe1fb5374706bf1b8a158fe73971')
```

    [34m[1mwandb[0m: Currently logged in as: [33mjpao[0m ([33mpri-data[0m). Use [1m`wandb login --relogin`[0m to force relogin
    [34m[1mwandb[0m: [33mWARNING[0m If you're specifying your api key in code, ensure this code is not shared publicly.
    [34m[1mwandb[0m: [33mWARNING[0m Consider setting the WANDB_API_KEY environment variable, or running `wandb login` from the command line.
    [34m[1mwandb[0m: Appending key for api.wandb.ai to your netrc file: /root/.netrc
    




    True



## Membaca data


```python
# Membaca data ke dalam DataFrame pandas
data = pd.read_csv('colorskim_word_dataset.csv')
data[:10]
```





  <div id="df-630f3a48-eb42-4253-8423-c461565d92a8">
    <div class="colab-df-container">
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
      <th>kata</th>
      <th>label</th>
      <th>urutan_kata</th>
      <th>total_kata</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>0</th>
      <td>ADISSAGE</td>
      <td>not_color</td>
      <td>1</td>
      <td>2</td>
    </tr>
    <tr>
      <th>1</th>
      <td>BLACK/BLACK/RUNWHT</td>
      <td>color</td>
      <td>2</td>
      <td>2</td>
    </tr>
    <tr>
      <th>2</th>
      <td>ADISSAGE</td>
      <td>not_color</td>
      <td>1</td>
      <td>2</td>
    </tr>
    <tr>
      <th>3</th>
      <td>N.NAVY/N.NAVY/RUNWHT</td>
      <td>color</td>
      <td>2</td>
      <td>2</td>
    </tr>
    <tr>
      <th>4</th>
      <td>3</td>
      <td>not_color</td>
      <td>1</td>
      <td>6</td>
    </tr>
    <tr>
      <th>5</th>
      <td>STRIPE</td>
      <td>not_color</td>
      <td>2</td>
      <td>6</td>
    </tr>
    <tr>
      <th>6</th>
      <td>D</td>
      <td>not_color</td>
      <td>3</td>
      <td>6</td>
    </tr>
    <tr>
      <th>7</th>
      <td>29.5</td>
      <td>not_color</td>
      <td>4</td>
      <td>6</td>
    </tr>
    <tr>
      <th>8</th>
      <td>BASKETBALL</td>
      <td>color</td>
      <td>5</td>
      <td>6</td>
    </tr>
    <tr>
      <th>9</th>
      <td>NATURAL</td>
      <td>color</td>
      <td>6</td>
      <td>6</td>
    </tr>
  </tbody>
</table>
</div>
      <button class="colab-df-convert" onclick="convertToInteractive('df-630f3a48-eb42-4253-8423-c461565d92a8')"
              title="Convert this dataframe to an interactive table."
              style="display:none;">

  <svg xmlns="http://www.w3.org/2000/svg" height="24px"viewBox="0 0 24 24"
       width="24px">
    <path d="M0 0h24v24H0V0z" fill="none"/>
    <path d="M18.56 5.44l.94 2.06.94-2.06 2.06-.94-2.06-.94-.94-2.06-.94 2.06-2.06.94zm-11 1L8.5 8.5l.94-2.06 2.06-.94-2.06-.94L8.5 2.5l-.94 2.06-2.06.94zm10 10l.94 2.06.94-2.06 2.06-.94-2.06-.94-.94-2.06-.94 2.06-2.06.94z"/><path d="M17.41 7.96l-1.37-1.37c-.4-.4-.92-.59-1.43-.59-.52 0-1.04.2-1.43.59L10.3 9.45l-7.72 7.72c-.78.78-.78 2.05 0 2.83L4 21.41c.39.39.9.59 1.41.59.51 0 1.02-.2 1.41-.59l7.78-7.78 2.81-2.81c.8-.78.8-2.07 0-2.86zM5.41 20L4 18.59l7.72-7.72 1.47 1.35L5.41 20z"/>
  </svg>
      </button>

  <style>
    .colab-df-container {
      display:flex;
      flex-wrap:wrap;
      gap: 12px;
    }

    .colab-df-convert {
      background-color: #E8F0FE;
      border: none;
      border-radius: 50%;
      cursor: pointer;
      display: none;
      fill: #1967D2;
      height: 32px;
      padding: 0 0 0 0;
      width: 32px;
    }

    .colab-df-convert:hover {
      background-color: #E2EBFA;
      box-shadow: 0px 1px 2px rgba(60, 64, 67, 0.3), 0px 1px 3px 1px rgba(60, 64, 67, 0.15);
      fill: #174EA6;
    }

    [theme=dark] .colab-df-convert {
      background-color: #3B4455;
      fill: #D2E3FC;
    }

    [theme=dark] .colab-df-convert:hover {
      background-color: #434B5C;
      box-shadow: 0px 1px 3px 1px rgba(0, 0, 0, 0.15);
      filter: drop-shadow(0px 1px 2px rgba(0, 0, 0, 0.3));
      fill: #FFFFFF;
    }
  </style>

      <script>
        const buttonEl =
          document.querySelector('#df-630f3a48-eb42-4253-8423-c461565d92a8 button.colab-df-convert');
        buttonEl.style.display =
          google.colab.kernel.accessAllowed ? 'block' : 'none';

        async function convertToInteractive(key) {
          const element = document.querySelector('#df-630f3a48-eb42-4253-8423-c461565d92a8');
          const dataTable =
            await google.colab.kernel.invokeFunction('convertToInteractive',
                                                     [key], {});
          if (!dataTable) return;

          const docLinkHtml = 'Like what you see? Visit the ' +
            '<a target="_blank" href=https://colab.research.google.com/notebooks/data_table.ipynb>data table notebook</a>'
            + ' to learn more about interactive tables.';
          element.innerHTML = '';
          dataTable['output_type'] = 'display_data';
          await google.colab.output.renderOutput(dataTable, element);
          const docLink = document.createElement('div');
          docLink.innerHTML = docLinkHtml;
          element.appendChild(docLink);
        }
      </script>
    </div>
  </div>




## Eksplorasi data


```python
# distribusi label dalam data
data['label'].value_counts()[:20]
```




    not_color    439
    color        148
    Name: label, dtype: int64




```python

```

## Konversi data ke dalam train dan test


```python
from sklearn.model_selection import train_test_split
train_kata, test_kata, train_label, test_label = train_test_split(data['kata'].to_numpy(), data['label'].to_numpy(), test_size=0.3)
train_kata[:5], test_kata[:5], train_label[:5], test_label[:5]
```




    (array(['CLIMA', 'ULTRABOOST', 'MYSINK/HIRAQU/CROYAL', 'TANGO', 'MILANO'],
           dtype=object),
     array(['SCARF', 'AC', 'MGREYH/MGREYH/BLACK', '3P', 'LINEAR'], dtype=object),
     array(['not_color', 'not_color', 'color', 'not_color', 'not_color'],
           dtype=object),
     array(['not_color', 'not_color', 'color', 'not_color', 'not_color'],
           dtype=object))



## Konversi label ke dalam numerik


```python
from sklearn.preprocessing import LabelEncoder
label_encoder = LabelEncoder()
train_label_encode = label_encoder.fit_transform(train_label)
test_label_encode = label_encoder.transform(test_label)
train_label_encode[:5], test_label_encode[:5]
```




    (array([1, 1, 0, 1, 1]), array([1, 1, 0, 1, 1]))



## Model 0: model dasar


```python
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.naive_bayes import MultinomialNB
from sklearn.pipeline import Pipeline

# Membuat pipeline
model_0 = Pipeline([
    ("tf-idf", TfidfVectorizer()),
    ("clf", MultinomialNB())
])

# Fit pipeline ke data training
model_0.fit(X=train_kata,
            y=train_label_encode)
```




    Pipeline(steps=[('tf-idf', TfidfVectorizer()), ('clf', MultinomialNB())])




```python
# Evaluasi model_0 pada data test
model_0.score(X=test_kata,
              y=test_label_encode)
```




    0.9661016949152542




```python
# Membuat prediksi menggunakan model_0
pred_model_0 = model_0.predict(test_kata)
pred_model_0
```




    array([1, 1, 0, 1, 1, 1, 1, 1, 0, 1, 1, 1, 1, 1, 1, 1, 0, 1, 1, 1, 0, 1,
           1, 1, 1, 0, 1, 1, 1, 0, 1, 1, 1, 1, 1, 0, 1, 1, 1, 1, 1, 1, 1, 1,
           0, 1, 0, 1, 0, 1, 1, 0, 1, 1, 1, 1, 1, 1, 1, 1, 0, 1, 0, 1, 0, 1,
           1, 1, 0, 1, 1, 1, 1, 0, 0, 1, 1, 1, 1, 1, 1, 0, 1, 1, 0, 1, 1, 0,
           1, 1, 1, 1, 0, 1, 0, 1, 0, 1, 1, 1, 1, 1, 1, 1, 1, 1, 0, 1, 1, 1,
           1, 0, 1, 1, 0, 1, 1, 1, 1, 0, 0, 1, 1, 1, 1, 0, 1, 0, 1, 1, 0, 0,
           1, 1, 1, 1, 1, 1, 1, 1, 1, 0, 1, 1, 0, 1, 1, 1, 1, 1, 1, 1, 1, 1,
           1, 1, 1, 1, 1, 1, 1, 1, 1, 0, 0, 1, 1, 1, 1, 1, 0, 1, 1, 1, 1, 1,
           0])




```python
# Membuat fungsi dasar untuk menghitung accuracy, precision, recall dan f1-score
from sklearn.metrics import accuracy_score, precision_recall_fscore_support
def hitung_hasil(target, prediksi):
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
    # Menghitung precision, recall dan f1-score model menggunakan "weighted average"
    model_presisi, model_recall, model_f1, _ = precision_recall_fscore_support(target, 
                                                                               prediksi, 
                                                                               average='weighted')
    
    hasil_model = {'akurasi': model_akurasi,
                   'presisi': model_presisi,
                   'recall': model_recall,
                   'f1-score': model_f1}
    
    return hasil_model
```


```python
# Menghitung hasil model_0
model_0_hasil = hitung_hasil(target=test_label_encode,
                             prediksi=pred_model_0)
model_0_hasil
```




    {'akurasi': 0.9661016949152542,
     'f1-score': 0.9652353913868151,
     'presisi': 0.9675649311059626,
     'recall': 0.9661016949152542}



## Menyiapkan data (text) untuk model deep sequence

### Text Vectorizer Layer


```python
# jumlah data (kata) dalam train_kata
len(train_kata)
```




    412




```python
# jumlah data unik (kata unik) dalam train_kata
jumlah_kata_train = len(np.unique(train_kata))
jumlah_kata_train
```




    226




```python
# Membuat text vectorizer
from tensorflow.keras.layers import TextVectorization
vectorizer_kata = TextVectorization(max_tokens=jumlah_kata_train,
                                    output_sequence_length=1,
                                    standardize='lower')
```


```python
# Mengadaptasikan text vectorizer ke dalam train_kata
vectorizer_kata.adapt(train_kata)
```


```python
# Test vectorizer_kata
import random
target_kata = random.choice(train_kata)
print(f'Kata:\n{target_kata}\n')
print(f'Kata seteleah vektorisasi:\n{vectorizer_kata([target_kata])}')
```

    Kata:
    EVERLESTO
    
    Kata seteleah vektorisasi:
    [[163]]
    


```python
vectorizer_kata.get_config()
```




    {'batch_input_shape': (None,),
     'dtype': 'string',
     'idf_weights': None,
     'max_tokens': 226,
     'name': 'text_vectorization',
     'ngrams': None,
     'output_mode': 'int',
     'output_sequence_length': 1,
     'pad_to_max_tokens': False,
     'ragged': False,
     'sparse': False,
     'split': 'whitespace',
     'standardize': 'lower',
     'trainable': True,
     'vocabulary': None}




```python
# Jumlah vocabulary dalam vectorizer_kata
jumlah_vocab = vectorizer_kata.get_vocabulary()
len(jumlah_vocab)
```




    226



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
print(f'\nKata seteleh embedding:\n{kata_terembed}\n')
print(f'Shape dari kata setelah embedding: {kata_terembed.shape}')
```

    Kata sebelum vektorisasi:
    EVERLESTO
    
    
    Kata sesudah vektorisasi (sebelum embedding):
    [[163]]
    
    
    Kata seteleh embedding:
    [[[ 0.01631815 -0.04775247 -0.00520905  0.02826596 -0.02610376
       -0.02318859  0.04112567 -0.03131066 -0.0055652   0.02334623
       -0.00561842  0.00632354  0.02209767  0.02569784  0.00146017
       -0.02496719  0.04397715  0.02374946 -0.02793208 -0.02479894
       -0.02689627  0.02449668  0.02413115 -0.00026416 -0.0474188
        0.02375449 -0.03313603  0.01957679  0.01208953  0.02894038
        0.04320562  0.02123917  0.03991547  0.00471902  0.00765711
        0.02515994 -0.04454259 -0.0184782  -0.0466426   0.03179142
        0.00160015  0.03690684 -0.01465347 -0.00856692 -0.04190071
       -0.0354219   0.04571031 -0.04488987 -0.02895441  0.03390359
        0.01122769  0.00747364  0.01801378 -0.02941638 -0.03116806
       -0.04856103 -0.00071516 -0.01321958 -0.0118474   0.03304395
        0.01553127  0.00069499 -0.04673848  0.02494338]]]
    
    Shape dari kata setelah embedding: (1, 1, 64)
    

# Membuat TensorFlow dataset


```python
# Membuat TensorFlow dataset
train_dataset = tf.data.Dataset.from_tensor_slices((train_kata, train_label_encode))
test_dataset = tf.data.Dataset.from_tensor_slices((test_kata, test_label_encode))

train_dataset
```




    <TensorSliceDataset element_spec=(TensorSpec(shape=(), dtype=tf.string, name=None), TensorSpec(shape=(), dtype=tf.int64, name=None))>




```python
# Membuat TensorSliceDataset menjadi prefetced dataset
train_dataset = train_dataset.batch(32).prefetch(tf.data.AUTOTUNE)
test_dataset = test_dataset.batch(32).prefetch(tf.data.AUTOTUNE)

train_dataset
```




    <PrefetchDataset element_spec=(TensorSpec(shape=(None,), dtype=tf.string, name=None), TensorSpec(shape=(None,), dtype=tf.int64, name=None))>



## Model 1: Conv1D dengan embedding


```python
# Membuat model_1 dengan layer Conv1D dari kata yang divektorisasi dan di-embed
from tensorflow.keras import layers

inputs = layers.Input(shape=(1,), dtype=tf.string, name='layer_input')
layer_vektor = vectorizer_kata(inputs)
layer_embed = kata_embed(layer_vektor)
x = layers.Conv1D(64, kernel_size=5, padding='same', activation='relu')(layer_embed)
x = layers.GlobalMaxPooling1D(name='layer_max_pool')(x)
outputs = layers.Dense(1, activation='sigmoid', name='layer_output')(x)
model_1 = tf.keras.Model(inputs, outputs, name='model_1_Conv1D_embed')

# Compile
model_1.compile(loss=tf.keras.losses.BinaryCrossentropy(),
                optimizer=tf.keras.optimizers.Adam(),
                metrics=['accuracy'])
```


```python
# Ringkasan model_1
model_1.summary()
```

    Model: "model_1_Conv1D_embed"
    _________________________________________________________________
     Layer (type)                Output Shape              Param #   
    =================================================================
     layer_input (InputLayer)    [(None, 1)]               0         
                                                                     
     text_vectorization (TextVec  (None, 1)                0         
     torization)                                                     
                                                                     
     layer_token_embedding (Embe  (None, 1, 64)            14464     
     dding)                                                          
                                                                     
     conv1d (Conv1D)             (None, 1, 64)             20544     
                                                                     
     layer_max_pool (GlobalMaxPo  (None, 64)               0         
     oling1D)                                                        
                                                                     
     layer_output (Dense)        (None, 1)                 65        
                                                                     
    =================================================================
    Total params: 35,073
    Trainable params: 35,073
    Non-trainable params: 0
    _________________________________________________________________
    


```python
# Plot model_1
from tensorflow.keras.utils import plot_model
plot_model(model_1, show_shapes=True)
```




    
![png](ColorSkim_AI_files/ColorSkim_AI_37_0.png)
    




```python
# import WandbCallback
from wandb.keras import WandbCallback

# Setup wandb init dan config
wb.init(project='ColorSkim',
        entity='jpao',
        name='model_1_Conv1D_embed',
        config={'epochs': 5,
                'n_layers': len(model_1.layers)})

# Fit model_1
hist_model_1 = model_1.fit(train_dataset,
                           epochs=wb.config.epochs,
                           validation_data=test_dataset,
                           callbacks=[WandbCallback()])
```

    [34m[1mwandb[0m: Currently logged in as: [33mjpao[0m. Use [1m`wandb login --relogin`[0m to force relogin
    


Tracking run with wandb version 0.12.19



Run data is saved locally in <code>/content/wandb/run-20220627_085711-31cyq9st</code>



Syncing run <strong><a href="https://wandb.ai/jpao/ColorSkim/runs/31cyq9st" target="_blank">model_1_Conv1D_embed</a></strong> to <a href="https://wandb.ai/jpao/ColorSkim" target="_blank">Weights & Biases</a> (<a href="https://wandb.me/run" target="_blank">docs</a>)<br/>


    [34m[1mwandb[0m: [33mWARNING[0m The save_model argument by default saves the model in the HDF5 format that cannot save custom objects like subclassed models and custom layers. This behavior will be deprecated in a future release in favor of the SavedModel format. Meanwhile, the HDF5 model is saved as W&B files and the SavedModel as W&B Artifacts.
    

    Epoch 1/5
     1/13 [=>............................] - ETA: 14s - loss: 0.6899 - accuracy: 0.7188

    [34m[1mwandb[0m: [32m[41mERROR[0m Can't save model in the h5py format. The model will be saved as W&B Artifacts in the SavedModel format.
    

    INFO:tensorflow:Assets written to: /content/wandb/run-20220627_085711-31cyq9st/files/model-best/assets
    

    [34m[1mwandb[0m: Adding directory to artifact (/content/wandb/run-20220627_085711-31cyq9st/files/model-best)... Done. 0.1s
    

    13/13 [==============================] - 4s 207ms/step - loss: 0.6785 - accuracy: 0.7354 - val_loss: 0.6626 - val_accuracy: 0.7514 - _timestamp: 1656320235.0000 - _runtime: 4.0000
    Epoch 2/5
    11/13 [========================>.....] - ETA: 0s - loss: 0.6440 - accuracy: 0.7557INFO:tensorflow:Assets written to: /content/wandb/run-20220627_085711-31cyq9st/files/model-best/assets
    

    [34m[1mwandb[0m: Adding directory to artifact (/content/wandb/run-20220627_085711-31cyq9st/files/model-best)... Done. 0.2s
    

    13/13 [==============================] - 3s 211ms/step - loss: 0.6420 - accuracy: 0.7427 - val_loss: 0.6235 - val_accuracy: 0.7514 - _timestamp: 1656320237.0000 - _runtime: 6.0000
    Epoch 3/5
     8/13 [=================>............] - ETA: 0s - loss: 0.5921 - accuracy: 0.7539INFO:tensorflow:Assets written to: /content/wandb/run-20220627_085711-31cyq9st/files/model-best/assets
    

    [34m[1mwandb[0m: Adding directory to artifact (/content/wandb/run-20220627_085711-31cyq9st/files/model-best)... Done. 0.1s
    

    13/13 [==============================] - 3s 231ms/step - loss: 0.5802 - accuracy: 0.7427 - val_loss: 0.5659 - val_accuracy: 0.7514 - _timestamp: 1656320240.0000 - _runtime: 9.0000
    Epoch 4/5
    13/13 [==============================] - ETA: 0s - loss: 0.4837 - accuracy: 0.7670INFO:tensorflow:Assets written to: /content/wandb/run-20220627_085711-31cyq9st/files/model-best/assets
    

    [34m[1mwandb[0m: Adding directory to artifact (/content/wandb/run-20220627_085711-31cyq9st/files/model-best)... Done. 0.2s
    

    13/13 [==============================] - 4s 315ms/step - loss: 0.4837 - accuracy: 0.7670 - val_loss: 0.5023 - val_accuracy: 0.7627 - _timestamp: 1656320245.0000 - _runtime: 14.0000
    Epoch 5/5
     9/13 [===================>..........] - ETA: 0s - loss: 0.3733 - accuracy: 0.8333INFO:tensorflow:Assets written to: /content/wandb/run-20220627_085711-31cyq9st/files/model-best/assets
    

    [34m[1mwandb[0m: Adding directory to artifact (/content/wandb/run-20220627_085711-31cyq9st/files/model-best)... Done. 0.2s
    

    13/13 [==============================] - 3s 276ms/step - loss: 0.3631 - accuracy: 0.8617 - val_loss: 0.4671 - val_accuracy: 0.8192 - _timestamp: 1656320250.0000 - _runtime: 19.0000
    


```python
# Evaluasi model_1
model_1.evaluate(test_dataset)
```

    6/6 [==============================] - 0s 5ms/step - loss: 0.4671 - accuracy: 0.8192
    




    [0.46710270643234253, 0.8192090392112732]




```python
# Membuat prediksi berdasarkan model_1
model_1_pred_prob = model_1.predict(test_dataset)
model_1_pred_prob[:10]
```




    array([[0.79627013],
           [0.856173  ],
           [0.46073976],
           [0.8121166 ],
           [0.79627013],
           [0.8073617 ],
           [0.85300666],
           [0.90371764],
           [0.4854719 ],
           [0.79627013]], dtype=float32)




```python
# Mengkonversi model_1_pred_prob dari probabilitas menjadi label
model_1_pred = tf.squeeze(tf.round(model_1_pred_prob))
model_1_pred
```




    <tf.Tensor: shape=(177,), dtype=float32, numpy=
    array([1., 1., 0., 1., 1., 1., 1., 1., 0., 1., 1., 1., 1., 1., 1., 1., 0.,
           1., 1., 1., 1., 1., 1., 1., 1., 1., 1., 1., 1., 0., 1., 1., 1., 1.,
           1., 1., 1., 1., 1., 1., 1., 1., 1., 1., 1., 1., 1., 1., 1., 1., 1.,
           1., 1., 1., 1., 1., 1., 1., 1., 1., 0., 1., 0., 1., 1., 1., 1., 1.,
           1., 1., 1., 1., 1., 0., 1., 1., 1., 1., 1., 1., 1., 1., 1., 1., 1.,
           1., 1., 1., 1., 1., 1., 1., 1., 1., 1., 1., 1., 1., 1., 1., 1., 1.,
           1., 1., 1., 1., 1., 1., 1., 1., 1., 1., 1., 1., 1., 1., 1., 1., 1.,
           0., 1., 1., 1., 1., 1., 0., 1., 1., 1., 1., 1., 0., 1., 1., 1., 1.,
           1., 1., 1., 1., 1., 1., 1., 1., 1., 1., 1., 1., 1., 1., 1., 1., 0.,
           1., 1., 1., 1., 1., 1., 1., 1., 1., 1., 1., 1., 1., 1., 1., 1., 1.,
           0., 1., 1., 1., 1., 1., 1.], dtype=float32)>




```python
# Menghitung metriks model_1
model_1_hasil = hitung_hasil(target=test_label_encode,
                             prediksi=model_1_pred)
model_1_hasil
```




    {'akurasi': 0.8192090395480226,
     'f1-score': 0.7772613766243615,
     'presisi': 0.8542715288477999,
     'recall': 0.8192090395480226}




```python

```




    array([[0.4478232]], dtype=float32)



## Model 2: Transfer learning pretrained feature extractor menggunakan Universal Sentence Encoder (USE)


```python
# Download pretrained USE
import tensorflow_hub as hub
tf_hub_embedding = hub.KerasLayer('https://tfhub.dev/google/universal-sentence-encoder/4',
                                  trainable=False,
                                  name='universal_sentence_encoder')
```


```python
# Melakukan tes pretrained embedding dalam pada contoh kata
kata_acak = random.choice(train_kata)
print(f'Kata acak:\n {kata_acak}')
kata_embed_pretrain = tf_hub_embedding([kata_acak])
print(f'\nKata setelah embed dengan USE:\n{kata_embed_pretrain[0][:30]}\n')
print(f'Panjang dari kata setelah embedding: {len(kata_embed_pretrain[0])}')
```

    Kata acak:
     CORBLU/CBLACK/FTWWHT
    
    Kata setelah embed dengan USE:
    [ 0.00860817 -0.00689607  0.05629712 -0.02064848 -0.04581999  0.0858016
      0.02770678 -0.04960595 -0.01491964  0.03378397  0.01505858  0.0569918
     -0.02326083  0.00949744 -0.06095064 -0.0286258   0.0223882   0.0515826
      0.00961048 -0.03192639  0.04371056 -0.00939714  0.01711809 -0.01394025
     -0.02168024  0.04030475 -0.01350616 -0.06460485  0.04084557  0.01243608]
    
    Panjang dari kata setelah embedding: 512
    


```python
# Membuat model_2 menggunakan USE
inputs = layers.Input(shape=[], dtype=tf.string, name='layer_input')
layer_embed_pretrained = tf_hub_embedding(inputs)
x = layers.Conv1D(64, kernel_size=5, activation='relu', name='layer_conv1d')(tf.expand_dims(layer_embed_pretrained, axis=-1))
x = layers.GlobalMaxPooling1D(name='layer_max_pooling')(x)
outputs = layers.Dense(1, activation='sigmoid', name='layer_output')(x)
model_2 = tf.keras.Model(inputs, outputs, name='model_2_Conv1D_USE_embed')
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
                                                                     
     layer_conv1d (Conv1D)       (None, 508, 64)           384       
                                                                     
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




    
![png](ColorSkim_AI_files/ColorSkim_AI_49_0.png)
    




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
        config={'epochs': 5,
                'n_layers': len(model_2.layers)})

# Fit model_2
hist_model_2 = model_2.fit(train_dataset,
                           epochs=wb.config.epochs,
                           validation_data=test_dataset,
                           callbacks=[WandbCallback()])
```


Finishing last run (ID:31cyq9st) before initializing another...



Waiting for W&B process to finish... <strong style="color:green">(success).</strong>



    VBox(children=(Label(value='3.186 MB of 3.186 MB uploaded (0.013 MB deduped)\r'), FloatProgress(value=1.0, max…



<style>
    table.wandb td:nth-child(1) { padding: 0 10px; text-align: left ; width: auto;} td:nth-child(2) {text-align: left ; width: 100%}
    .wandb-row { display: flex; flex-direction: row; flex-wrap: wrap; justify-content: flex-start; width: 100% }
    .wandb-col { display: flex; flex-direction: column; flex-basis: 100%; flex: 1; padding: 10px; }
    </style>
<div class="wandb-row"><div class="wandb-col"><h3>Run history:</h3><br/><table class="wandb"><tr><td>accuracy</td><td>▁▁▁▃█</td></tr><tr><td>epoch</td><td>▁▃▅▆█</td></tr><tr><td>loss</td><td>█▇▆▄▁</td></tr><tr><td>val_accuracy</td><td>▁▁▁▂█</td></tr><tr><td>val_loss</td><td>█▇▅▂▁</td></tr></table><br/></div><div class="wandb-col"><h3>Run summary:</h3><br/><table class="wandb"><tr><td>accuracy</td><td>0.86165</td></tr><tr><td>best_epoch</td><td>4</td></tr><tr><td>best_val_loss</td><td>0.4671</td></tr><tr><td>epoch</td><td>4</td></tr><tr><td>loss</td><td>0.36305</td></tr><tr><td>val_accuracy</td><td>0.81921</td></tr><tr><td>val_loss</td><td>0.4671</td></tr></table><br/></div></div>



Synced <strong style="color:#cdcd00">model_1_Conv1D_embed</strong>: <a href="https://wandb.ai/jpao/ColorSkim/runs/31cyq9st" target="_blank">https://wandb.ai/jpao/ColorSkim/runs/31cyq9st</a><br/>Synced 5 W&B file(s), 1 media file(s), 16 artifact file(s) and 1 other file(s)



Find logs at: <code>./wandb/run-20220627_085711-31cyq9st/logs</code>



Successfully finished last run (ID:31cyq9st). Initializing new run:<br/>



Tracking run with wandb version 0.12.19



Run data is saved locally in <code>/content/wandb/run-20220627_093220-3pr9h21l</code>



Syncing run <strong><a href="https://wandb.ai/jpao/ColorSkim/runs/3pr9h21l" target="_blank">model_2_Conv1D_USE_embed</a></strong> to <a href="https://wandb.ai/jpao/ColorSkim" target="_blank">Weights & Biases</a> (<a href="https://wandb.me/run" target="_blank">docs</a>)<br/>


    Epoch 1/5
    13/13 [==============================] - 15s 669ms/step - loss: 0.6708 - accuracy: 0.7427 - val_loss: 0.6512 - val_accuracy: 0.7514 - _timestamp: 1656322355.0000 - _runtime: 11.0000
    Epoch 2/5
    13/13 [==============================] - 6s 471ms/step - loss: 0.6408 - accuracy: 0.7427 - val_loss: 0.6283 - val_accuracy: 0.7514 - _timestamp: 1656322362.0000 - _runtime: 18.0000
    Epoch 3/5
    13/13 [==============================] - 6s 521ms/step - loss: 0.6197 - accuracy: 0.7427 - val_loss: 0.6087 - val_accuracy: 0.7514 - _timestamp: 1656322372.0000 - _runtime: 28.0000
    Epoch 4/5
    13/13 [==============================] - 6s 471ms/step - loss: 0.6012 - accuracy: 0.7427 - val_loss: 0.5919 - val_accuracy: 0.7514 - _timestamp: 1656322378.0000 - _runtime: 34.0000
    Epoch 5/5
    13/13 [==============================] - 6s 466ms/step - loss: 0.5861 - accuracy: 0.7427 - val_loss: 0.5794 - val_accuracy: 0.7514 - _timestamp: 1656322389.0000 - _runtime: 45.0000
    


```python
# Evaluase model_2
model_2.evaluate(test_dataset)
```

    6/6 [==============================] - 0s 19ms/step - loss: 0.5794 - accuracy: 0.7514
    




    [0.5794160962104797, 0.7514124512672424]




```python
# Membuat prediksi dengan model_2
model_2_pred_prob = model_2.predict(test_dataset)
model_2_pred_prob[:10]
```




    array([[0.6672992 ],
           [0.67215574],
           [0.67375624],
           [0.6732286 ],
           [0.6711316 ],
           [0.67131513],
           [0.66906154],
           [0.67712283],
           [0.6757687 ],
           [0.6676929 ]], dtype=float32)




```python
# Mengkonversi model_2 menjadi label format
model_2_pred = tf.squeeze(tf.round(model_2_pred_prob))
model_2_pred
```




    <tf.Tensor: shape=(177,), dtype=float32, numpy=
    array([1., 1., 1., 1., 1., 1., 1., 1., 1., 1., 1., 1., 1., 1., 1., 1., 1.,
           1., 1., 1., 1., 1., 1., 1., 1., 1., 1., 1., 1., 1., 1., 1., 1., 1.,
           1., 1., 1., 1., 1., 1., 1., 1., 1., 1., 1., 1., 1., 1., 1., 1., 1.,
           1., 1., 1., 1., 1., 1., 1., 1., 1., 1., 1., 1., 1., 1., 1., 1., 1.,
           1., 1., 1., 1., 1., 1., 1., 1., 1., 1., 1., 1., 1., 1., 1., 1., 1.,
           1., 1., 1., 1., 1., 1., 1., 1., 1., 1., 1., 1., 1., 1., 1., 1., 1.,
           1., 1., 1., 1., 1., 1., 1., 1., 1., 1., 1., 1., 1., 1., 1., 1., 1.,
           1., 1., 1., 1., 1., 1., 1., 1., 1., 1., 1., 1., 1., 1., 1., 1., 1.,
           1., 1., 1., 1., 1., 1., 1., 1., 1., 1., 1., 1., 1., 1., 1., 1., 1.,
           1., 1., 1., 1., 1., 1., 1., 1., 1., 1., 1., 1., 1., 1., 1., 1., 1.,
           1., 1., 1., 1., 1., 1., 1.], dtype=float32)>




```python
# Menghitung hasil metriks dari model_2
model_2_hasil = hitung_hasil(target=test_label_encode,
                             prediksi=model_2_pred)
model_2_hasil
```

    /usr/local/lib/python3.7/dist-packages/sklearn/metrics/_classification.py:1318: UndefinedMetricWarning: Precision and F-score are ill-defined and being set to 0.0 in labels with no predicted samples. Use `zero_division` parameter to control this behavior.
      _warn_prf(average, modifier, msg_start, len(result))
    




    {'akurasi': 0.751412429378531,
     'f1-score': 0.64476034262803,
     'presisi': 0.5646206390245458,
     'recall': 0.751412429378531}



## Model 3: Menggunakan positional kata dan custom embed dan concatenate layer


```python

```


```python
# Test prediksi dengan model_0 Multinomial Naive-Bayes
class_list = ['warna', 'bukan_warna']
article = 'PUMA XTG WOVEN PANTS PUMA BLACK-PUMA WHITE'
article_list = article.replace("-"," ").split()
model_test = model_0.predict(article.replace("-"," ").split())
for i in range(0, len(article_list)):
    print(f'Kata: {article_list[i]}\nPrediksi: {class_list[model_test[i]]}\n\n')
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

```
