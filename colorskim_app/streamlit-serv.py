import streamlit as st
import pandas as pd
from st_aggrid import AgGrid, GridOptionsBuilder
from bahasa.id_ID import *

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
if "metode_ekstraksi" not in st.session_state:
    st.session_state.metode_ekstraksi = "Metode 2"


def prediksi_warna(file):
    st.write(f"Lakukan prediksi warna pada {file.name}")


with st.sidebar:
    st.write(txt_instruksi_metode)
    st.selectbox(
        "Pilih metode ekstraksi warna:",
        ["Metode 1", "Metode 2"],
        index=1,
        key="metode_ekstraksi",
    )

st.write(txt_judul)

if st.session_state.metode_ekstraksi == "Metode 2":
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
        st.write("Prediksi warna metode 1")

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
