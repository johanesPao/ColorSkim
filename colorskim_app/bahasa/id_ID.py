import streamlit as st

txt_judul = """
        # ColorSkim untuk Artikel
        
        *Aplikasi ini merupakan implementasi dari pemisahan atribut `warna` 
        dari `nama_artikel` menggunakan [**Model Quadbrid Embedding**](https://dagshub.com/johanesPao/ColorSkim/src/44665a6c8887fca566143f03753c756356333377/colorskim_checkpoint/model_3_quadbrid_embedding/saved_model.pb) yang 
        dilatih menggunakan TensorFlow pada training data [`setengah_dataset_artikel.csv`](https://dagshub.com/johanesPao/ColorSkim/src/44665a6c8887fca566143f03753c756356333377/data/setengah_dataset_artikel.csv).*
        
        *Note: untuk detail instruksi penggunaan, lihat di **Menu** kiri atas.*
        
        ----
        """
txt_intro = """
        ----
        
        * Dokumentasi ColorSkim: https://colorskim.jpao.live
        * Notebook ColorSkim: http://nbviewer.org/github/johanesPao/ColorSkim/blob/main/ColorSkim_AI.ipynb
        * Github Repository: https://github.com/johanesPao/ColorSkim
        * Streamlit Container: *in progress*
        * Data Version Control: https://dagshub.com/johanesPao/ColorSkim?filter=model
        
        ### Logika Ekstraksi
        
        Logika ektraksi dari aplikasi ini adalah dengan melakukan prediksi 
        (*Named Entity Recognition*) pada sekumpulan kata atau kalimat dengan
        tahapan sebagai berikut:
        
        1. Satu atau sekumpulan pasangan `brand` `nama_artikel` akan dipecah menjadi `brand`, `nama_artikel`, `kata`, `urut_kata` dan `total_kata`.
        
        2. Prediksi `label` dilakukan terhadap masing - masing `kata` untuk menemukan label `warna` dan `bukan_warna`.
        
        3. Ekstraksi `warna` akan dilakukan untuk semua kata sejak kejadian pertama (*first occurence*) label yang diprediksi sebagai `warna` hingga akhir kalimat. 
        
        > Contoh: Adidas Yeezy Afternoon Orange Dim -> Afternoon adalah kata pertama dengan label warna -> Adidas Yeezy (`nama_artikel`) & Afternoon Orange Dim (`warna`)
        """
txt_garis = "----"
txt_brand = "Brand"
txt_nama_artikel = "Nama Artikel"
txt_metode_1_judul = "### *One Line Quick Extraction*"
txt_metode_2_judul = "### *Multiple Articles Extraction*"
txt_label_unggah = "Unggah file CSV dengan encoding UTF-8 dengan 2 kolom Brand dan Nama Artikel tanpa header pada baris pertama !!"
txt_help_unggah = """
        Pastikan file dalam format CSV dengan encoding 
        UTF-8, memiliki 2 kolom brand dan nama artikel 
        serta tidak memiliki header pada baris pertama.
        """
txt_tbl_ekstraksi_warna = "Ekstrak Warna"
txt_instruksi_metode = """
        Untuk melakukan ekstraksi warna dari artikel 
        silahkan pilih satu diantara dua metode ekstraksi warna yang
        tersedia dalam aplikasi ini.
        
        * ### *One Line Quick Extraction*
        
          1. Input Brand yang terdiri dari 3 karakter (contoh: ADI).
          2. Input Nama Artikel.
          3. Klik tombol `Ekstrak Warna`
        
        * ### *Multiple Articles Extraction*
        
          1. Pastikan file CSV memiliki encoding UTF-8.
          2. Pastikan file CSV hanya memiliki 2 kolom dimana kolom pertama adalah `brand` dan kolom kedua adalah `nama_artikel`.
          3. Pastikan file tidak memiliki header pada baris pertama.
          4. Klik `Browse files` untuk mengunggah file CSV.
          5. Review data pada tabel input.
          6. Klik tombol `Ekstrak Warna`.
          7. Review data pada tabel output.
          8. Download file CSV.
"""
txt_link_help = "https://colorskim.jpao.live"
txt_link_bug = "https://github.com/johanesPao/ColorSkim/issues/new"
txt_tentang = """
        # ColorSkim untuk Artikel
        
        Copyright (c) 2022 Johanes Indra Pradana Pao
"""
