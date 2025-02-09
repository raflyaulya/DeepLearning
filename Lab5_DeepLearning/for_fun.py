import requests
from bs4 import BeautifulSoup
import json
import os

def scrape_wikipedia(keyword):
    # Setup URL dan headers
    keyword = keyword.strip().replace(' ', '_').title()
    url = f"https://wikipedia.org/wiki/{keyword}"
    headers = {
        "User-Agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/91.0.4472.124 Safari/537.36"
    }

    try:
        # Request ke Wikipedia
        response = requests.get(url, headers=headers)
        response.raise_for_status()  # Cek error HTTP

        soup = BeautifulSoup(response.text, 'html.parser')

        # Ambil judul artikel
        title = soup.find("h1", {"id": "firstHeading"}).text

        # Ambil semua paragraf (skip yang kosong)
        paragraphs = [p.text.strip() for p in soup.find_all("p") if p.text.strip()]
        
        # Ambil daftar isi (headings)
        sections = [h2.text for h2 in soup.find_all("h2")]

        # Ambil gambar pertama (jika ada)
        image = soup.find("img", {"class": "mw-file-element"})
        image_url = f"https:{image['src']}" if image else "Tidak ada gambar"

        # Simpan hasil dalam dictionary
        data = {
            "title": title,
            "url": url,
            "summary": paragraphs[0] if paragraphs else "Tidak ada ringkasan",
            "content": paragraphs,
            "sections": sections,
            "image_url": image_url
        }

        return data

    except requests.exceptions.HTTPError:
        print(f"Error: Artikel '{keyword}' tidak ditemukan di Wikipedia!")
    except requests.exceptions.ConnectionError:
        print("Error: Koneksi internet bermasalah!")
    except Exception as e:
        print(f"Error: {str(e)}")

def save_to_file(data, format="txt"):
    # Buat folder 'output' jika belum ada
    if not os.path.exists("output"):
        os.makedirs("output")

    filename = f"output/{data['title']}.{format}"

    try:
        if format == "txt":
            with open(filename, "w", encoding="utf-8") as f:
                f.write(f"Judul: {data['title']}\n")
                f.write(f"URL: {data['url']}\n\n")
                f.write("Ringkasan:\n" + data["summary"] + "\n\n")
                f.write("Gambar: " + data["image_url"] + "\n\n")
                f.write("Daftar Isi:\n")
                for section in data["sections"]:
                    f.write(f"- {section}\n")
        elif format == "json":
            with open(filename, "w", encoding="utf-8") as f:
                json.dump(data, f, indent=4, ensure_ascii=False)
        print(f"Data disimpan di: {filename}")
    except Exception as e:
        print(f"Gagal menyimpan file: {str(e)}")

def main():
    keyword = input("Masukkan topik yang ingin di-scrape (contoh: Indonesia, AI): ")
    data = scrape_wikipedia(keyword)

    if data:
        print("\n=== Hasil Scraping ===")
        print(f"Judul: {data['title']}")
        print(f"Ringkasan: {data['summary'][:200]}...")  # Potong teks biar singkat
        print(f"Gambar: {data['image_url']}")

        # Simpan ke file
        pilihan = input("\nSimpan sebagai (txt/json)? ").lower()
        if pilihan in ["txt", "json"]:
            save_to_file(data, pilihan)
        else:
            print("Format tidak valid, data tidak disimpan.")

if __name__ == "__main__":
    main()