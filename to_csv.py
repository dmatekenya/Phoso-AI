import pdfplumber
import pandas as pd
import os

# Chemin vers le fichier PDF
pdf_file = '/Users/dunstanmatekenya/Downloads/IPC_Malawi_Acute_Food_Insecurity_May_2024_Mar_2025_Report-1.pdf'

# Chemin où enregistrer les fichiers CSV et Excel
output_path = '/Users/dunstanmatekenya/Downloads/'

# Créer le dossier de sortie s'il n'existe pas
# os.makedirs(output_path, exist_ok=True)

# Ouverture du fichier PDF
with pdfplumber.open(pdf_file) as pdf:
    for i, page in enumerate(pdf.pages):
        # Extraction des tableaux
        tables = page.extract_tables()
        if tables:
            for j, table in enumerate(tables):
                # Convertir le tableau en DataFrame pandas
                df = pd.DataFrame(table[1:], columns=table[0])

                # Enregistrer le DataFrame en CSV si le tableau n'est pas vide
                if not df.empty:
                    if len(df) > 3 and not df.dropna().empty:
                        '''csv_file = os.path.join(output_path, f'tableau_{i}_{j+1}.csv')
                        df.to_csv(csv_file, index=False, header=True)
                        print(f'Tableau {i}_{j+1} enregistré dans {csv_file}')'''

                        # Enregistrer le DataFrame en Excel
                        excel_file = os.path.join(output_path, f'tableau_{i}_{j+1}.xlsx')
                        df.to_excel(excel_file, index=False)
                        print(f'Tableau {i}_{j+1} enregistré dans {excel_file}')
                    else:
                        print(f'Le tableau {i+1}_{j+1} ne contient pas suffisamment de données.')
                else:
                    print(f'Page {i} ne contient pas de tableaux.')
        else:
            print(f'Page {i+1} ne contient pas de tableaux.')
