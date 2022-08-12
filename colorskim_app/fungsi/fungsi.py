import streamlit as st
import pandas as pd


def preprocessing_input_artikel(brand, artikel):
    dataset_kata = pd.DataFrame([])
    # Cek metode ekstraksi

    st.write(f"{brand} - {artikel}")
    df = pd.DataFrame({"brand": [brand], "nama_artikel": [artikel]})
    ganti_karakter = "/-"
    for i in range(len(df)):
        artikel_full = df.loc[i, "nama_artikel"]
        artikel_untuk_split = artikel_full
        for c in ganti_karakter:
            artikel_untuk_split = artikel_untuk_split.replace(c, " ")
        split_spasi_artikel = artikel_untuk_split.split()
        print(f"Memproses {i+1} dari {len(df)} baris...")
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
                columns=["brand", "nama_artikel", "kata", "urut_kata", "total_kata"],
            )
            dataset_kata = pd.concat([dataset_kata, artikel_df], ignore_index=True)

    return dataset_kata
