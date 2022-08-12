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

    df = pd.DataFrame({"brand": [brand], "nama_artikel": [artikel]})
    ganti_karakter = "/-"
    progress_bar = st.progress(0)
    for i in range(len(df)):
        artikel_full = df.loc[i, "nama_artikel"]
        artikel_untuk_split = artikel_full
        for c in ganti_karakter:
            artikel_untuk_split = artikel_untuk_split.replace(c, " ")
        split_spasi_artikel = artikel_untuk_split.split()
        with st.spinner(f"Memproses {i + 1} dari {len(df)} baris..."):
            progress_bar.progress((i + 1) / len(df))
            for i in range(len(split_spasi_artikel)):
                artikel_df = pd.DataFrame(
                    [
                        [
                            brand,
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
            st.success(f"Selesai memproses {len(df)} baris.")

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
def memuat_model():
    model = tf.keras.models.load_model(os.path.join(direktori_streamlit, "model"))
    return model
