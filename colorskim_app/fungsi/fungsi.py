from nntplib import ArticleInfo
import os
import streamlit as st
import pandas as pd
import numpy as np
from sklearn.preprocessing import OneHotEncoder, LabelEncoder
import tensorflow as tf
import tensorflow.keras.backend as K  # type: ignore

direktori = os.getcwd()
direktori_streamlit = os.path.join(direktori, "colorskim_app")


def preprocessing_input_artikel(brand, artikel):
    """
    Fungsi ini akan memproses input data ke dalam format yang benar untuk
    prediksi model.

    Args:
        brand (str atau list): Brand dalam bentuk string atau list of string
        artikel (str atau list): Artikel dalam bentuk string atau list of string

    Returns:
        dataset_kata (pd.DataFrame): DataFrame dalam format brand, nama_artikel,
            urut_kata dan total_kata
    """
    dataset_kata = pd.DataFrame([])

    df = pd.DataFrame({"brand": brand, "nama_artikel": artikel})
    ganti_karakter = "/-"
    with st.spinner(f"Memproses {len(df)} baris artikel..."):
        progress_bar = st.progress(0)
        for i in range(len(df)):
            nama_brand = df.iloc[i, 0]
            artikel_full = df.iloc[i, 1]
            artikel_untuk_split = artikel_full
            for c in ganti_karakter:
                artikel_untuk_split = artikel_untuk_split.replace(c, " ")
            split_spasi_artikel = artikel_untuk_split.split()
            progress_bar.progress((i + 1) / len(df))
            for i in range(len(split_spasi_artikel)):
                artikel_df = pd.DataFrame(
                    [
                        [
                            nama_brand,
                            artikel_full,
                            split_spasi_artikel[i],
                            i + 1,
                            len(split_spasi_artikel),
                        ]
                    ],
                    columns=[
                        "brand",
                        "nama_artikel",
                        "kata",
                        "urut_kata",
                        "total_kata",
                    ],
                )
                dataset_kata = pd.concat([dataset_kata, artikel_df], ignore_index=True)
        st.info(f"Selesai memproses {len(df)} baris data.")

    return dataset_kata


def preprocessing_input_model(input_dataset, max_kata=15):
    """
    Fungsi ini akan mempersiapkan input data yang sudah diformat untuk
    persiapan input ke dalam model melalui proses encoding, dataset,
    batching dan prefetching

    Args:
        input_dataset (pf.DataFrame): DataFrame dengan format brand, nama_artikel,
            urut_kata dan total_kata
        max_kata (int): Jumlah maksimal kata dalam 1 artikel dari sekumpulan
            artikel yang dilatih pada model. (Nilai mengikuti nilai max_kata
            pada model)

    Returns:
        dataset (tf.data.Dataset): Dataset yang sudah dilakukan batching dan
            prefetching -- ValueError
        list (list): list dari kata, urut_kata_encoded, total_kata_encoded
            dan brand_encoded
        df_encode (pd.DataFrame): DataFrame yang akan ditampilkan setelah
            proses encoding pada brand
    """
    # onehot brand
    # load list brand yang digunakan pada saat training model
    # fit berdasarkan list brand tersebut dan transform input data brand
    brand_pada_model = pd.read_csv(
        os.path.join(direktori_streamlit, "aset/brand_transformer.csv")
    )
    encoder = OneHotEncoder(sparse=False)
    brand_encode = encoder.fit(
        brand_pada_model["brand"].to_numpy().reshape(-1, 1)
    ).transform(input_dataset["brand"].to_numpy().reshape(-1, 1))
    df_brand_encode = pd.DataFrame(
        brand_encode, columns=encoder.get_feature_names_out(["brand"])
    )

    # dataframe sebelum batching dan prefetching
    df_encode = pd.concat([input_dataset, df_brand_encode], axis=1).drop(
        ["brand"], axis=1
    )

    # onehot urut_kata dan total_kata
    urut_kata_encode = tf.one_hot(input_dataset["urut_kata"].to_numpy(), depth=max_kata)
    total_kata_encode = tf.one_hot(
        input_dataset["total_kata"].to_numpy(), depth=max_kata
    )

    # ------------------------ Entah mengapa dataset ValueError dalam prediksi------------------
    # print(
    #     input_dataset.iloc[:, 2],
    #     df_brand_encode,
    #     urut_kata_encode,
    #     total_kata_encode,
    # )

    # print(
    #     input_dataset.iloc[:, 2].shape,
    #     df_brand_encode.shape,
    #     urut_kata_encode.shape,
    #     total_kata_encode.shape,
    # )

    # Dataset, Batching, Prefetching input data
    # dataset = tf.data.Dataset.from_tensor_slices(
    #     (input_dataset.iloc[:, 2], df_brand_encode, urut_kata_encode, total_kata_encode)
    # )
    # dataset = dataset.batch(32).prefetch(tf.data.AUTOTUNE)
    # ----------------------------------------------------------------------------------------------

    return [
        input_dataset.iloc[:, 2],
        df_brand_encode,
        urut_kata_encode,
        total_kata_encode,
    ], df_encode


@st.cache(allow_output_mutation=True)
def memuat_model(direktori_model="model"):
    """
    Fungsi ini digunakan untuk memuat model yang terdapat dalam folder
    ./model

    Args:
        direktori_model (str): Direktori model yang akan dimuat. Default = 'model'

    Returns:
        model (tf.keras.Model): Model yang dimuat untuk prediksi
    """
    model = tf.keras.models.load_model(os.path.join(direktori_streamlit, "model"))
    return model


def prediksi_kata(artikel_full, list_kata, prediksi):
    """
    Fungsi ini digunakan untuk menampilkan hasil prediksi dalam format
    artikel_full, bukan_warna dan warna

    Args:
        artikel_full (str): string nama_artikel secara lengkap
        list_kata (list): list kata yang akan digunakan sebagai referensi pencarian
            kejadian label warna yang pertama
        prediksi (list): list hasil prediksi dari model

    Returns:
        artikel_full (str): string nama_artikel secara lengkap
        bukan_warna (str): string kata yang merupakan bukan warna
        warna (str): string kata yang merupakan warna
    """
    # CARI INDEKS KEJADIAN PERTAMA WARNA
    # jika bukan_warna, masukkan ke dalam list temporer
    list_bukan_warna = []

    # jika tidak mengandung warna maka
    if sum(prediksi) == 0:
        print("ga ada warna")
        bukan_warna = artikel_full
        warna = []
    # jika kata pertama sudah mengandung warna
    # dan fungsi ini berusaha mengembalikan
    # list warna dan tidak ada list bukan_warna
    elif prediksi[0] == 1:
        print("semua warna")
        bukan_warna = []
        warna = artikel_full
    # normalnya...
    else:
        for indeks, i in enumerate(prediksi):
            if i == 0:
                list_bukan_warna.append(list_kata[indeks])
            else:
                # jika warna, cek apakah kata ini sudah ada di dalam
                # list_bukan_warna, dan jika sudah ada, kata keberapa?
                n_kata = list_bukan_warna.count(list_kata[indeks]) + 1
                kata = list_kata[indeks]
                # print(list_bukan_warna, list_kata, kata, n_kata)

                # hentikan for loop
                break

        # Membagi artikel_full menjadi beberapa bagian berdasar jumlah kejadin
        # kata saat kata pertama dengan label warna muncul
        bagian = artikel_full.split(kata, n_kata)
        # Kurangi panjang artikel_full dengan panjang bagian paling akhir
        # dan dikurangi dengan panjang kata untuk mendapatkan indeks
        # terakhir bukan_warna dan indeks pertama warna dalam kalimat
        # artikel_full
        indeks = len(artikel_full) - len(bagian[-1]) - len(kata)

        # Membuat kalimat bukan_warna dan warna berdasar indeks dalam kalimat
        bukan_warna = artikel_full[:indeks]
        warna = artikel_full[indeks:]

        # hapus special char di akhir bukan_warna dan awal warna
        special_char = "/-"
        for char in special_char:
            if char == bukan_warna[-1]:
                bukan_warna = bukan_warna[:-1]
            if char == warna[0]:
                warna = warna[1:]

    return artikel_full, bukan_warna, warna
