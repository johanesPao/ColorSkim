import os
import streamlit as st
import pandas as pd
from st_aggrid import AgGrid, GridOptionsBuilder
from bahasa.id_ID import *
from fungsi.fungsi import *
from tensorflow.keras.models import load_model  # type: ignore
import tensorflow as tf
from annotated_text import annotated_text

tf.config.run_functions_eagerly(True)
tf.data.experimental.enable_debug_mode()

direktori = os.getcwd()
direktori_streamlit = os.path.join(direktori, "colorskim_app")

st.set_page_config(
    page_title="Pemisahan Warna dari Artikel - PRI",
    page_icon="🤖",
    initial_sidebar_state="collapsed",
    layout="wide",
    menu_items={
        "Get Help": txt_link_help,
        "Report a bug": txt_link_bug,
        "About": txt_tentang,
    },
)

if "single_brand" and "single_artikel" not in st.session_state:
    st.session_state.single_brand = ""
    st.session_state.single_artikel = ""


def prediksi_warna(file):
    st.write(f"Lakukan prediksi warna pada {file.name}")


with st.sidebar:
    st.write(txt_instruksi_metode)
    st.radio(
        "Pilih metode ekstraksi warna:",
        ["One Line Quick Extraction", "Multiple Articles Extraction"],
        index=1,
        key="metode_ekstraksi",
    )

st.write(txt_judul)

if st.session_state.metode_ekstraksi == "Multiple Articles Extraction":
    # METODE 2
    st.write(txt_metode_2_judul)
    file_diupload = st.file_uploader(txt_label_unggah, type="csv", help=txt_help_unggah)
    if file_diupload is not None:
        df_file = pd.read_csv(file_diupload, names=["brand", "nama_artikel"])

        # TAMPILKAN TABEL DATA INPUT
        # konfigurasi dan build AgGrid
        gb_df_file = GridOptionsBuilder.from_dataframe(df_file)
        gb_df_file.configure_pagination(enabled=True)
        opsi_aggrid_df_file = gb_df_file.build()

        # display preprocessed data dalam tabel
        AgGrid(
            df_file,
            gridOptions=opsi_aggrid_df_file,
            height=250,
            theme="dark",
            fit_columns_on_grid_load=True,
        )
        st.button(
            txt_tbl_ekstraksi_warna, on_click=prediksi_warna, args=(file_diupload,)
        )
else:
    # METODE 1
    st.write(txt_metode_1_judul)
    st.text_input(txt_brand, key="single_brand")
    st.text_input(txt_nama_artikel, key="single_artikel")
    tbl_ekstrak_metode_1 = st.button(txt_tbl_ekstraksi_warna)
    if tbl_ekstrak_metode_1:
        # Cek panjang karakter st.session_state.single_brand
        if (
            len(st.session_state.single_brand) == 0
            or len(st.session_state.single_artikel) == 0
        ):
            st.write("Brand/Nama Artikel tidak boleh kosong")
            pass
        else:
            # PREPROCESSING INPUT DATA
            st.write(
                f"Prediksi akan dilakukan untuk: {st.session_state.single_brand} - {st.session_state.single_artikel}"
            )

            # preprocessing input ke dalam format untuk prediksi model
            dataset = preprocessing_input_artikel(
                st.session_state.single_brand, st.session_state.single_artikel
            )

            # konfigurasi dan build AgGrid
            gb_dataset = GridOptionsBuilder.from_dataframe(dataset)
            gb_dataset.configure_pagination(enabled=True)
            opsi_aggrid_dataset = gb_dataset.build()

            # display preprocessed data dalam tabel
            AgGrid(
                dataset,
                gridOptions=opsi_aggrid_dataset,
                height=250,
                theme="dark",
                fit_columns_on_grid_load=True,
            )

            # LOAD MODEL
            with st.spinner("Memuat Model Quadbrid Embedding..."):
                model = memuat_model()
                st.info("Model selesai dimuat.")
            # with st.spinner("Memuat Model Quadbrid Embedding..."):
            #     model = load_model(
            #         os.path.join(
            #             direktori_streamlit, "model/model_3_quadbrid_embedding"
            #         )
            #     )
            #     st.success("Model berhasil dimuat")

            # PREDIKSI LABEL WARNA
            with st.spinner("Mempersiapkan batching dan prefetching input data..."):
                dataset, df_encode = preprocessing_input_model(dataset)
                st.info("Batching dan prefetching input data selesai.")

            # konfigurasi dan build AgGrid
            gb_df_encode = GridOptionsBuilder.from_dataframe(df_encode)
            gb_df_encode.configure_pagination(enabled=True)
            opsi_aggrid_encode = gb_df_encode.build()

            AgGrid(
                df_encode,
                gridOptions=opsi_aggrid_encode,
                height=250,
                theme="dark",
            )

            with st.spinner("Memprediksi warna..."):
                prediksi = tf.squeeze(tf.round(model.predict(dataset)))
                st.success("Prediksi selesai.")

            kata = tf.squeeze(df_encode["kata"])

            st.write(kata)
            st.write(prediksi)

            # buat dictionary untuk menampung hasil formatting prediksi
            prediksi_terformat = {"nama_artikel": [], "bukan_warna": [], "warna": []}

            # buat list untuk menampung kata selama loop
            list_kata = []

            # loop dalam df_encode
            for i in range(len(df_encode)):
                # jika urut_kata tidak sama dengan total_kata,
                # maka tambahkan kata ke dalam list_kata
                if df_encode.iloc[i, 2] != df_encode.iloc[i, 3]:
                    list_kata.append(df_encode.iloc[i, 1])
                # jika urut_kata sama dengan total_kata,
                # maka tambahkan kata ke dalam list_kata,
                # lakukan loop untuk memformat prediksi
                # ke dalam bentuk yg bisa dipahami pengguna
                # dan tambahkan ke dalam prediksi_terformat
                # di akhir bagian bersihkan list_kata = []
                # untuk loop selanjutnya
                else:
                    # tambahkan kata terakhir ke dalam list_kata
                    list_kata.append(df_encode.iloc[i, 1])

                    # format prediksi
                    artikel_full, bukan_warna, warna = prediksi_kata(
                        artikel_full=df_encode.iloc[i, 0],
                        list_kata=list_kata,
                        prediksi=prediksi,
                    )

                    # tambahkan ke dalam prediksi_terformat
                    prediksi_terformat["nama_artikel"].append(artikel_full)
                    prediksi_terformat["bukan_warna"].append(bukan_warna)
                    prediksi_terformat["warna"].append(warna)

                    # bersihkan list_kata
                    list_kata = []

            st.write("### Prediksi dalam format JSON:")
            st.write(prediksi_terformat)

            st.write("### Prediksi dalam tabel:")
            # buat dataframe
            df_prediksi = pd.DataFrame(prediksi_terformat)
            # konfigurasi dan build AgGrid
            gb_df_prediksi = GridOptionsBuilder.from_dataframe(df_prediksi)
            gb_df_prediksi.configure_pagination(enabled=True)
            opsi_aggrid_prediksi = gb_df_prediksi.build()

            AgGrid(
                df_prediksi,
                gridOptions=opsi_aggrid_prediksi,
                height=95,  # set tinggi 50 untuk metode prediksi 1 baris
                theme="dark",
                fit_columns_on_grid_load=True,
            )

st.write(txt_intro)
