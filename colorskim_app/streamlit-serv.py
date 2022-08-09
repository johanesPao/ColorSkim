import streamlit as st
import pandas as pd
from st_aggrid import AgGrid, GridOptionsBuilder
import bahasa.id_ID as bahasa

st.set_page_config(layout="wide")

if "input_brand" and "input_artikel" not in st.session_state:
    st.session_state.multiple_brand = []
    st.session_state.multiple_artikel = []

st.write(bahasa.judul)

kolom_penjelasan, kolom_penggunaan = st.columns(2)
with kolom_penjelasan:
    st.write(bahasa.intro)
with kolom_penggunaan:
    with st.form(key="form_metode_1"):
        st.write(bahasa.metode_1_judul)
        st.text_input(bahasa.txt_brand, key="input_brand")
        st.text_input(bahasa.txt_nama_artikel, key="input_artikel")
        metode_1_submit = st.form_submit_button(label="Esktrak")
    with st.form(key="form_metode_2"):
        st.write(bahasa.metode_2_judul)
        st.text_input(bahasa.txt_brand, key="input_brand2")
        st.text_input(bahasa.txt_nama_artikel, key="input_artikel2")
        metode_2_submit = st.form_submit_button(label="Esktrak")
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
