import streamlit as st
import pandas as pd
import numpy as np
from sklearn.preprocessing import OneHotEncoder, LabelEncoder
import tensorflow as tf


def preprocessing_input_artikel(brand, artikel):
    dataset_kata = pd.DataFrame([])
    # Cek metode ekstraksi
    st.write(f"Prediksi akan dilakukan pada: [{brand} - {artikel}]")
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
            st.success(f"Berhasil memproses {len(df)} baris")

    return dataset_kata


def preprocessing_input_model(input_dataset):
    # onehot brand
    encoder = OneHotEncoder(sparse=False)
    brand_encode = encoder.fit_transform(
        input_dataset["brand"].to_numpy().reshape(-1, 1)
    )
    df_brand_encode = pd.DataFrame(
        brand_encode, columns=encoder.get_feature_names_out(["brand"])
    )

    # onehot urut_kata dan total_kata
    max_kata = int(np.max(input_dataset["urut_kata"]))
    urut_kata_encode = tf.one_hot(input_dataset["urut_kata"].to_numpy(), depth=max_kata)
    total_kata_encode = tf.one_hot(
        input_dataset["total_kata"].to_numpy(), depth=max_kata
    )

    # Dataset, Batching, Prefetching input data
    dataset = tf.data.Dataset.from_tensor_slices(
        (input_dataset.iloc[:, 0], df_brand_encode, urut_kata_encode, total_kata_encode)
    )
    dataset = dataset.batch(32).prefetch(tf.data.AUTOTUNE)

    return dataset
