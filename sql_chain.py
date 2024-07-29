import json
from decouple import config
from langchain_community.utilities import SQLDatabase
from sqlalchemy import create_engine, MetaData, text
from sqlalchemy.orm import sessionmaker

from langchain.chains import create_sql_query_chain
from langchain_openai import ChatOpenAI
from operator import itemgetter
from langchain_core.output_parsers import StrOutputParser
from langchain_core.prompts import PromptTemplate
from langchain_core.runnables import RunnablePassthrough
from langchain_community.tools.sql_database.tool import QuerySQLDataBaseTool
from langchain_core.prompts import FewShotPromptTemplate, PromptTemplate



OPENAI_API_KEY = config("OPENAI_API_KEY")
DB_USER = config('DB_USER')
DB_PASSWORD = config('DB_PASSWORD')
DB_HOST = 'localhost'
DB_PORT = '5432'
DB_NAME = 'food_security'
DATABASE_URL = f'postgresql://{DB_USER}:{DB_PASSWORD}@{DB_HOST}:{DB_PORT}/{DB_NAME}'
FILE_SQL_EXAMPLES_EN = "./sql_examples_en.txt"
FILE_SQL_EXAMPLES_NY = "./sql_examples_ny.txt"

            

def create_db_object_with_metadata():
    # Create the SQLAlchemy engine
    engine = create_engine(DATABASE_URL)
    metadata_obj = MetaData()
    metadata_obj.reflect(bind=engine)

    # Create a configured "Session" class
    Session = sessionmaker(bind=engine)
    session = Session()

    # Load custom metadata from the table_metadata and column_metadata tables
    try:
        table_metadata = session.execute(text("SELECT * FROM table_metadata")).fetchall()
        column_metadata = session.execute(text("SELECT * FROM column_metadata")).fetchall()

        # Add table metadata
        for row in table_metadata:
            table_name = row.table_name
            description = row.description
            table = metadata_obj.tables.get(table_name)
            table.info['description'] = description

        # Add column metadata
        for row in column_metadata:
            table_name = row.table_name
            if table_name == "commodity_prices":
                continue
            column_name = row.column_name
            description = row.description
            table = metadata_obj.tables.get(table_name)
            column = table.columns[column_name.lower()]
            column.info['description'] = description
    finally:
        session.close()
    
    db = SQLDatabase(engine=engine, metadata=metadata_obj, ignore_tables=['table_metadata', 'column_metadata'])

    return db

def load_sql_examples(file_path):
    with open(file_path, 'r', encoding='utf-8') as file:
        return json.load(file)

def run_sql_chain(user_question, lan="en"):
    # Load examples
    examples = load_sql_examples(file_path=FILE_SQL_EXAMPLES_EN)

    # Create FewShot Prompt
    example_prompt = PromptTemplate.from_template("User input: {input}\nSQL query: {query}")
    sql_prompt = FewShotPromptTemplate(
        examples=examples,
        example_prompt=example_prompt,
        prefix="You are a PostgreSQL expert. Given an input question, create a syntactically correct PostgreSQL query to run. Unless otherwise specificed, do not return more than {top_k} rows.\n\nHere is the relevant table info: {table_info}\n\nBelow are a number of examples of questions and their corresponding SQL queries.",
        suffix="User input: {input}\nSQL query: ",
        input_variables=["input", "top_k", "table_info"],
    )

    # Create SQL Chain and LLM to use
    db = create_db_object_with_metadata()
    llm = ChatOpenAI(model="gpt-3.5-turbo", temperature=0, openai_api_key=OPENAI_API_KEY)
    execute_query = QuerySQLDataBaseTool(db=db)
    write_query = create_sql_query_chain(llm, db, sql_prompt)

    # Create Answer Promopt to help package the response
    answer_prompt = PromptTemplate.from_template(
            """Given the following user question and SQL result, answer the user question.

        Question: {question}
        SQL Result: {result}
        Answer: """
    )

    # Create answer chain
    answer = answer_prompt | llm | StrOutputParser()

    # Put everything together
    chain = (
        RunnablePassthrough.assign(query=write_query).assign(
            result=itemgetter("query") | execute_query
        )
        | answer
    )

    return chain.invoke({"question": "{}".format(user_question)})
    

def main():
    questions_en = ["Where can I find the cheapest maize?",
                    "Which district harvested the most beans?",
                    "How much is Maize in Zomba?",
                    "WHich district produced more Tobacco, Mchinji or Kasungu?",
                    "Where can I get bananas?"]
    for q in questions_en:
        print("\n--- QUESTION: ", q)
        output = run_sql_chain(q)
        print('OUTPUT TYPE==>', output)

if __name__ == '__main__':
    # this is to quite parallel tokenizers warning.
    main()

