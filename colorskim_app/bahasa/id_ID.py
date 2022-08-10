import streamlit as st

txt_judul = """
        # ColorSkim untuk Artikel
        
        *Aplikasi ini merupakan implementasi dari pemisahan atribut `warna` 
        dari `nama_artikel` menggunakan [**Model Quadbrid Embedding**](https://dagshub.com/johanesPao/ColorSkim/src/44665a6c8887fca566143f03753c756356333377/colorskim_checkpoint/model_3_quadbrid_embedding/saved_model.pb) yang 
        dilatih menggunakan TensorFlow pada training data [`setengah_dataset_artikel.csv`](https://dagshub.com/johanesPao/ColorSkim/src/44665a6c8887fca566143f03753c756356333377/data/setengah_dataset_artikel.csv).*
        
        *Note: untuk detail instruksi penggunaan, lihat di bagian **Logika Ekstraksi** dan **Metode Penggunaan**.*
        
        ----
        """
txt_intro = """
        ----
        
        * Dokumentasi ColorSkim: https://colorskim.jpao.live
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
        
        ### Metode Penggunaan
        
        Terdapat dua alternatif metode pemisahan `warna` dari `nama_artikel` 
        pada aplikasi ini. 
        
        1. *Quick One Line Extraction*
        
        Metode ini dapat anda gunakan saat anda ingin melakukan *quick 
        extraction* `warna` dari satu artikel saja.
        
        2. *Multiple Articles Extraction*
        
        Metode ini dapat anda gunakan ketika anda memiliki banyak artikel 
        yang ingin diekstrak `warna` dari `nama_artikel`-nya.
        """
txt_garis = "----"
txt_brand = "Brand"
txt_nama_artikel = "Nama Artikel"
txt_metode_1_judul = "### Metode 1: *Quick One Line Extraction*"
txt_metode_2_judul = "### Metode 2: *Multiple Articles Extraction*"
txt_label_unggah = "Unggah file CSV dengan encoding UTF-8 dengan 2 kolom Brand dan Nama Artikel tanpa header pada baris pertama !!"
txt_help_unggah = """
        Pastikan file dalam format CSV dengan encoding 
        UTF-8, memiliki 2 kolom brand dan nama artikel 
        serta tidak memiliki header pada baris pertama.
        """
txt_tbl_ekstraksi_warna = "Ekstraksi Warna"
