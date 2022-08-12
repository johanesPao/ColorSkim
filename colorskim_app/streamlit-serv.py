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
        df = pd.read_csv(file_diupload)
        st.dataframe(df)
        # AgGrid(df)
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
                f"Prediksi akan dilakukan pada: [{st.session_state.single_brand} - {st.session_state.single_artikel}]"
            )

            # preprocessing input ke dalam format untuk prediksi model
            dataset = preprocessing_input_artikel(
                st.session_state.single_brand, st.session_state.single_artikel
            )

            # konfigurasi dan build AgGrid
            gb = GridOptionsBuilder.from_dataframe(dataset)
            gb.configure_pagination(enabled=True)
            opsi_aggrid = gb.build()

            # display preprocessed data dalam tabel
            AgGrid(
                dataset,
                gridOptions=opsi_aggrid,
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

            # Prediksi label warna
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

            bukan_warna = []
            warna = []

            # loop dalam df_encode
            for i in range(len(df_encode)):
                # jika urut_kata = total_kata maka artikel baru
                if df_encode.iloc[i, 2] == df_encode.iloc[i, 3]:
                    print("akhir dari artikel")
                    # ambil nama_artikel
                    # cek label kata terakhir
                else:
                    print("proses dalam 1 artikel")
                    # cek label kata dan berusaha untuk menemukan
                    # kejadian pertama dari label warna
                    # selama looping dalam 1 artikel, simpan
                    # semua kata ke dalam list temporer dan
                    # ketika menemukan label warna pertama,
                    # cek apakah kata tersebut pernah muncul di
                    # list temporer sebelumnya (contoh PUMA SHOES
                    # PUMA WHITE - PUMA BLACK)
                    # gunakan fungsi find seperti:
                    # def findnth(haystack, needle, n):
                    #     parts= haystack.split(needle, n+1)
                    #     if len(parts)<=n+1:
                    #         return -1
                    #     return len(haystack)-len(parts[-1])-len(needle)
                    # kembalikan indeks kata warna pertama ini dan slice
                    # nama_artikel berdasar indeks ini

            # Restrukturisasi output

            # Kembalikan nama_artikel dengan anotasi warna

st.write(txt_intro)

# Cara lain yang mungkin lebih baik dan rapi adalah menggunakan sidebar
# untuk menentukan metode yang ingin dipakai, default adalah metode 2.
# Menampilkan konten ekstraksi sesuai dengan metode yang dipilih.
# st.write(bahasa.metode_2_judul)

# jumlah_input = st.number_input(bahasa.txt_baris_input, 5)

# mode_halaman = st.checkbox("Aktifkan mode halaman tabel", value=False)
# if mode_halaman:
#     st.write("Opsi halaman tabel")
#     halaman_tabel_auto = st.checkbox("Halaman tabel otomatis", value=True)
#     if not halaman_tabel_auto:
#         mode_halaman_tabel = st.number_input(
#             "Data per halaman", value=5, min_value=0, max_value=100
#         )

# submit_jumlah_input = st.button("Buat Baris")
# if submit_jumlah_input:
#     st.session_state.jumlah_input = int(jumlah_input)

# # data = pd.DataFrame({"brand": [], "nama_artikel": []})
# if st.session_state.jumlah_input != 0:
#     list_brand = []
#     list_artikel = []

#     for i in range(st.session_state.jumlah_input):
#         list_brand.append("")
#         list_artikel.append("")

#     data = pd.DataFrame({"brand": list_brand, "nama_artikel": list_artikel})
#     gb = GridOptionsBuilder.from_dataframe(data)
#     gb.configure_default_column(editable=True, groupable=True)

#     if mode_halaman:
#         if halaman_tabel_auto:
#             gb.configure_pagination(paginationAutoPageSize=True)
#         else:
#             gb.configure_pagination(
#                 paginationAutoPageSize=False, paginationPageSize=mode_halaman_tabel
#             )
#     gb.configure_grid_options(domLayout="autoWidth")
#     opsi_grid = gb.build()

#     grid = AgGrid(data, gridOptions=opsi_grid, width="100%")

#     st.write(st.session_state.jumlah_input)

# # data = pd.read_csv("./data/setengah_dataset_artikel.csv")
# # AgGrid(data, theme="dark")
