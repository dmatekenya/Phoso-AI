import os
from pathlib import Path

from pypdf import PdfReader
import pdfplumber
from langchain.docstore.document import Document
from langchain_community.document_loaders import TextLoader
from langchain_community.document_loaders.csv_loader import CSVLoader
from langchain.document_loaders import DirectoryLoader

DIR_DOCS = "./data/docs"
FILTER_PDF = False           # remove tables and figures, keep text only 


def is_table_char(char_bbox, table_bbox):
    """Check if a character bbox is within a table bbox."""
    char_x0, char_y0, char_x1, char_y1 = char_bbox
    table_x0, table_y0, table_x1, table_y1 = table_bbox
    return (
        char_x0 >= table_x0 and char_x1 <= table_x1 and
        char_y0 >= table_y0 and char_y1 <= table_y1
    )

def list_pdf_files(data_dir=DIR_DOCS):
    paths = Path(data_dir).glob('**/*.pdf')
    for path in paths:
        yield str(path)

def list_txt_files(data_dir=DIR_DOCS):
    paths = Path(data_dir).glob('**/*.txt')
    for path in paths:
        yield str(path)


def load_txt_files(data_dir=DIR_DOCS):
    docs = []
    paths = list_txt_files(data_dir)
    for path in paths:
        print(f"Loading {path}")
        loader = TextLoader(path)
        docs.extend(loader.load())
    return docs


def load_csv_files(data_dir=DIR_DOCS):
    docs = []
    paths = Path(data_dir).glob('**/*.csv')
    for path in paths:
        loader = CSVLoader(file_path=str(path))
        docs.extend(loader.load())
    return docs


# Use with result of file_to_summarize = st.file_uploader("Choose a file") or a string.
# or a file like object.
def get_document_text(uploaded_file, title=None):
    docs = []
    fname = uploaded_file.name
    if not title:
        title = os.path.basename(fname)
    if fname.lower().endswith('pdf'):
        pdf_reader = PdfReader(uploaded_file)
        for num, page in enumerate(pdf_reader.pages):
            page = page.extract_text()
            doc = Document(page_content=page, metadata={'title': title, 'page': (num + 1)})
            docs.append(doc)

    else:
        # assume text
        doc_text = uploaded_file.read().decode()
        docs.append(doc_text)

    return docs

def load_filtered_pdf(pdf_file_path):
    extracted_text = ""

    with pdfplumber.open(pdf_file_path) as pdf:
        for pg_num, page in enumerate(pdf.pages, start=1):
            print("Working on page {}".format(pg_num))
            # Extract table bboxes
            table_bboxes = [table.bbox for table in page.find_tables()]

            # Extract all text, then filter out text within tables
            for char in page.chars:
                char_bbox = (char['x0'], char['top'], char['x1'], char['bottom'])
                if not any(is_table_char(char_bbox, bbox) for bbox in table_bboxes):
                    extracted_text += char['text']
            extracted_text += "\n"
    
    return extracted_text


def load_pdf_files(data_dir=DIR_DOCS, filter=FILTER_PDF):
    if filter:
        loader = DirectoryLoader(data_dir, glob="**/*.pdf", loader_kwargs={'loader_fn': load_filtered_pdf})
        docs = loader.load()
        return docs
    docs = []
    paths = Path(data_dir).glob('**/*.pdf')
    for path in paths:
        print(path)
        this_lst = get_document_text(path, title=None)
        docs += this_lst
    return docs

if __name__ == "__main__":
    example_pdf_path = "examples/healthy_meal_10_tips.pdf"
    #docs = get_document_text(open(example_pdf_path, "rb"))
    docs = load_pdf_files()
    for doc in docs:
        print(doc)
    # docs = get_document_text(open("examples/us_army_recipes.txt", "rb"))
    # for doc in docs:
    #     print(doc)
    # txt_docs = load_txt_files("examples")
    # for doc in txt_docs:
    #     print(doc)
    # csv_docs = load_csv_files("examples")
    # for doc in csv_docs:
    #     print(doc)

