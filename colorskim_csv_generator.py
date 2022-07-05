"""
File ini digunakan untuk men-generate dataset per kata untuk colorskim artikel
dengan membaca colorskim_articles.csv dan menuliskannya ke csv baru dengan nama
colorskim_dataset.csv yang dikompresi dalam colorskim_dataset.zip
"""

import pandas as pd

# Membaca colorskim_articles.csv
data = pd.read_csv("colorskim_articles.csv", names=["nama_artikel"], header=None)

# Inisialisasi dataframe akhir
dataset_kata = pd.DataFrame([])

for i in range(len(data)):  # loop setiap baris dalam dataframe data
    # mengganti karakter '/' dan '-' dengan spasi ' '
    replace_karakter = "/-"
    artikel_full = data.loc[i, "nama_artikel"]
    artikel_untuk_split = artikel_full
    for c in replace_karakter:
        artikel_untuk_split = artikel_untuk_split.replace(c, " ")
    # split artikel berdasarkan spasi
    split_artikel = artikel_untuk_split.split()
    print(f"Memproses {i+1} dari {len(data)} baris...")
    for i in range(
        len(split_artikel)
    ):  # loop dalam list hasil split_artikel (per kata)
        # bentuk dataframe untuk menampung kata, label, urut_kata dan total_kata
        # edit: menambahkan full article name untuk referensi saat labeling
        artikel_df = pd.DataFrame(
            [[artikel_full, split_artikel[i], "", i + 1, len(split_artikel)]],
            columns=["nama_artikel", "kata", "label", "urut_kata", "total_kata"],
        )
        # menggabungkan dataframe yang dihasilkan ke dalam dataframe akhir
        dataset_kata = pd.concat([dataset_kata, artikel_df], ignore_index=True)

# Set opsi untuk kompresi output
nama_file = "colorskim_dataset"
opsi_kompresi = dict(method="zip", archive_name=nama_file + ".csv")
# Menulis dataframe ke dalam csv yang dikompresi
dataset_kata.to_csv(nama_file + ".zip", index=False, compression=opsi_kompresi)
# Print selesai
print(f"File selesai dituliskan ke dalam {nama_file}.zip")
