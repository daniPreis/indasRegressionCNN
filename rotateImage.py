import csv

new_csv_dir = 'D:/Dokumente/Uni/Master/Indas/BMW - Material-20210512/trained_images.csv'

if __name__ == "__main__":
    with open(new_csv_dir, newline='') as csvfile:
        original_data = list(csv.reader(csvfile))