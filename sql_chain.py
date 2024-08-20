import json
import psycopg2
from decouple import config
from langchain_community.utilities import SQLDatabase
from sqlalchemy import create_engine, MetaData, text
from sqlalchemy.orm import sessionmaker

from langchain.chains import create_sql_query_chain, LLMChain
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
FILE_SQL_EXAMPLES_EN = "./sql_examples_en.json"
FILE_SQL_EXAMPLES_NY = "./sql_examples_ny.json"
USE_BEST_MATCHING_COLUMNS = False


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
         
def connect_to_database(database_url=DATABASE_URL):
    """Connects to a postgreSQL


    Parameters
    ----------
    database_url : String
        postgreSQL database connection URL, by default DATABASE_URL
    """
    # conn = psycopg2.connect(f"dbname={DB_NAME} user={DB_USER} password={DB_PASSWORD}")
    conn = psycopg2.connect(database_url)

    cur = conn.cursor()

    # Query to get table names and column names
    cur.execute("SELECT table_name, description FROM table_metadata")
    tables = cur.fetchall()

    cur.execute("SELECT table_name, column_name, description FROM column_metadata")
    columns = cur.fetchall()

    cur.close()
    conn.close()

    return tables, columns    

def find_best_table_prompt(user_query, tables, columns, 
                           return_chain=True, llm=None):# Define the template for selecting the best table
    template = """
    You are a database assistant. Given the following tables and columns with their descriptions, select the best table that matches the user's query.

    Tables and Columns:
    {table_info}

    User Query:
    {user_query}

    Provide the output in the following JSON format:
    {{
        "best_matching_table": {{
            "table_name": "<best_table_name>",
            "description": "<best_table_description>"
        }}
    }}
    """
    # Prepare the table_info string including descriptions for each table and its columns
    table_info = ""
    for table in tables:
        table_name, table_description = table
        table_info += f"Table: {table_name} - {table_description}\n"
        table_columns = [col for col in columns if col[0] == table_name]
        for column in table_columns:
            _, column_name, column_description = column
            table_info += f"    Column: {column_name} - {column_description}\n"
        table_info += "\n"

    # Create the PromptTemplate
    prompt_template = PromptTemplate(
        template=template,
        input_variables=["table_info", "user_query"]
    )

    # Format the template 
    formatted_prompt = prompt_template.format(table_info=table_info, user_query=user_query)

    if return_chain:
        # Create the chain using the ChatOpenAI model and the PromptTemplate
        chain = LLMChain(llm=llm,prompt=prompt_template)
        return chain, {"table_info": table_info, "user_query": user_query}

    return formatted_prompt

def get_columns_info(table_name, columns):
    """
    Retrieves formatted information of columns for a specified table.

    This function iterates through a list of column metadata and returns a formatted string 
    containing the names and descriptions of the columns associated with the given table.

    Parameters
    ----------
    table_name : str
        The name of the table for which the column information is needed.
    columns : list of tuples
        A list of tuples containing column metadata. Each tuple includes 'table_name', 
        'column_name', and 'description'.

    Returns
    -------
    columns_info : str
        A formatted string containing the column names and descriptions for the specified table.
    """
    columns_info = ""
    for column in columns:
        table, column_name, column_description = column
        if table == table_name:
            columns_info += f"    Column: {column_name} - {column_description}\n"
    return columns_info

def find_best_columns_prompt(user_query, best_matching_table, columns, return_chain=True, llm=None):
    """
    Generates a formatted prompt to determine the most relevant columns for a given user query 
    based on the best matching table and its columns' descriptions. Optionally, returns an LLMChain 
    object for further processing.

    Parameters
    ----------
    user_query : str
        The user's query that needs to be matched with the most relevant columns in the table.
    best_matching_table : dict
        A dictionary containing the best matching table information with 'table_name' and 'description' keys.
    columns : list of tuples
        A list of tuples containing column metadata. Each tuple includes 'table_name', 'column_name', 
        and 'description'.
    return_chain : bool, optional
        If True, returns a tuple containing an LLMChain object and the input data. If False, returns the 
        formatted prompt as a string. Default is True.
    llm : Optional
        The language model object (e.g., ChatOpenAI) to be used in the LLMChain for generating responses. 
        Default is None.

    Returns
    -------
    result : tuple or str
        - If return_chain is True: Returns a tuple (LLMChain object, dict containing context information).
        - If return_chain is False: Returns the formatted prompt string.
    """
    # Define the template for selecting the relevant columns
    column_template = """
    You are a database assistant. Given the following columns for the table '{table_name}', select the columns that are most relevant to the user's query.

    Table Description: {table_description}

    Columns:
    {columns_info}

    User Query:
    {user_query}

    Relevant Columns:
    """

    columns_info = get_columns_info(best_matching_table["table_name"], columns)

    # Create the PromptTemplate for column selection
    column_prompt_template = PromptTemplate(
        template=column_template,
        input_variables=["table_name", "table_description", "columns_info", "user_query"]
    )

    # Example usage of the template with a user query
    formatted_column_prompt = column_prompt_template.format(
        table_name=best_matching_table["table_name"],
        table_description=best_matching_table["description"],
        columns_info=columns_info,
        user_query=user_query
    )

    # Prepare the context for running the chain
    context = {
        "table_name": best_matching_table["table_name"],
        "table_description": best_matching_table["description"],
        "columns_info": columns_info,
        "user_query": user_query
    }

    if return_chain:
        chain = LLMChain(llm=llm, prompt=column_prompt_template)
        return chain, context

    return formatted_column_prompt

def load_sql_examples(file_path):
    """
    Loads SQL examples from a JSON file.

    This function reads a JSON file from the specified file path and loads the content 
    into a Python dictionary or list, depending on the structure of the JSON file.

    Parameters
    ----------
    file_path : str
        The path to the JSON file containing the SQL examples.

    Returns
    -------
    examples : dict or list
        The content of the JSON file, typically a dictionary or list containing SQL examples.
    """
    with open(file_path, 'r', encoding='utf-8') as file:
        return json.load(file)

def create_sql_prompt(examples, best_matching_table, columns_metadata, 
                      use_best_matching_columns=False):
    """
    Creates a FewShotPromptTemplate for generating SQL queries based on table and column metadata.

    This function generates a prompt template that includes detailed information about the table and its columns.
    The generated prompt instructs a language model (LLM) to create a syntactically correct SQL query based on
    user input. If the table contains a date column and the user does not specify a date, the prompt also instructs
    the LLM to retrieve the most recent data available.

    Parameters
    ----------
    examples : list of dict
        A list of example inputs and corresponding SQL queries. Each example should be a dictionary with 'input' and 'query' keys.
    best_matching_table : dict
        A dictionary containing the best matching table information with 'table_name' and 'description' keys.
    columns_metadata : list of tuples
        A list of tuples containing columns metadata. Each tuple should include 'table_name', 'column_name', and 'description'.
    use_best_matching_columns : bool, optional
        A flag indicating whether to use only the best-matching columns (if True) or all columns in the table (if False). Default is True.

    Returns
    -------
    sql_prompt : FewShotPromptTemplate
        A FewShotPromptTemplate object that can be used with an LLM to generate SQL queries.
    """
    # Prepare table_info string based on the best matching table and columns
    # table_info = f"Table: {best_matching_table['table_name']} - {best_matching_table['description']}\n"
    columns_info = "Columns:\n"
    has_date_column = False

    # Determine which columns to use: best-matching or all columns
    if use_best_matching_columns:
        # If using best_matching_columns, use those provided (filtering columns_metadata based on matching logic)
        columns_to_use = columns_metadata  # Assuming columns_metadata is already filtered
    else:
        # Use all columns for the given table from columns_metadata
        columns_to_use = [col for col in columns_metadata if col[0] == best_matching_table['table_name']]

    # Construct the columns_info string
    for column in columns_to_use:
        table_name, column_name, column_description = column
        columns_info += f"    Column: {column_name} - {column_description}\n"
        if 'date' in column_name.lower():
            has_date_column = True

    # Create FewShot Prompt with instructions for handling most recent data
    example_prompt = PromptTemplate.from_template("User input: {input}\nSQL query: {query}")

    # Add a special instruction if the table has a date column
    recent_data_instruction = (
        "If the user does not specify a date, retrieve the most recent data available by ordering the results "
        "by the date column in descending order."
    ) if has_date_column else ""

    # Combine table_info and columns_info in the prompt
    sql_prompt = FewShotPromptTemplate(
        examples=examples,
        example_prompt=example_prompt,
        prefix=(
            "You are a PostgreSQL expert. Given an input question, create a syntactically correct PostgreSQL query to run. "
            "Unless otherwise specified, do not return more than {top_k} rows.\n\n"
            "Here is the relevant table information:\n{table_info}\n\n"
            f"{recent_data_instruction}\n\n"
            "Below are a number of examples of questions and their corresponding SQL queries."
        ),
        suffix="User input: {input}\nSQL query: ",
        input_variables=["input", "table_info", "top_k"],
    )
    
    
    return sql_prompt

def create_answer_chain(llm):
    """
    Creates a chain for generating answers to user questions based on SQL query results using a language model (LLM).

    This function constructs a prompt that instructs the LLM to generate answers derived strictly from the provided SQL result. 
    The generated answer will include appropriate units, specify the time period if relevant, and clearly indicate if the result 
    does not fully answer the question.

    Parameters
    ----------
    llm : Any
        The language model (e.g., ChatGPT, GPT-3.5) to be used for generating the answers.

    Returns
    -------
    answer_chain : Chain
        A chain object that, when invoked, processes the user question and SQL result to generate a concise and accurate answer.

    Example
    -------
    answer_chain = create_answer_chain(llm)
    response = answer_chain.invoke({"question": "What is the price of Maize?", "result": "Price: 60"})
    print(response)
    """
    answer_prompt = PromptTemplate.from_template(
        """
        You are a knowledgeable assistant. Given the following user question and SQL result, answer the question accurately and concisely.

        IMPORTANT:
        1. Your answer MUST be derived directly from the SQL Result provided. Do not add any extra information, assumptions, or context beyond what is given.
        2. Always include appropriate units in your answer (e.g., Kwacha per kg, liters, etc.).
        3. Specify the time period or date if the question implies or explicitly asks for it.
        4. If the SQL Result does not contain enough information to fully answer the question, clearly state that the answer is based on the available data and provide any relevant context from the result.

        Example:
        - Question: "What's the price of Maize?" SQL Result: "Price: 60" Answer: "The most recent price of Maize is 60 Kwacha per kg."
        - Question: "What's the price of Maize for May 2024?" SQL Result: "Price: 60" Answer: "The price of Maize in May 2024 is 60 Kwacha per kg."

        Question: {question}
        SQL Result: {result}
        Answer:
        """
    )

    return answer_prompt | llm | StrOutputParser()

def run_sql_chain_v2(user_query, best_table_info, columns_info, best_columns=None, llm=None):
    """
    Executes an SQL query generation chain using a language model (LLM) based on the user query, 
    best matching table, and columns information.

    This function loads example SQL queries, creates an SQL prompt tailored to the best matching 
    table and its columns, and then executes a chain that generates and executes an SQL query. 
    The response is returned after processing the generated query.

    Parameters
    ----------
    user_query : str
        The user's query for which an SQL query needs to be generated.
    best_table_info : dict
        A dictionary containing the best matching table information with 'table_name' and 'description' keys.
    columns_info : list of tuples
        A list of tuples containing columns metadata for the table. Each tuple includes 'table_name', 
        'column_name', and 'description'.
    best_columns : list of tuples, optional
        A list of tuples containing the best matching columns metadata, if available. If provided, 
        the SQL prompt will be generated using only these columns. Default is None.
    llm : Any, optional
        The language model (e.g., ChatOpenAI) to be used for generating the SQL query. Default is None.

    Returns
    -------
    response : Any
        The response from the executed SQL query chain, typically containing the results of the SQL query.

    Example
    -------
    response = run_sql_chain(
        user_query="What is the price of maize?",
        best_table_info={"table_name": "maize_prices", "description": "Contains maize price data"},
        columns_info=[("maize_prices", "price", "Price of maize"), ("maize_prices", "date", "Date of the price entry")],
        llm=ChatOpenAI()
    )
    print(response)
    """
    # Load examples and create prompts
    examples = load_sql_examples(file_path=FILE_SQL_EXAMPLES_EN)
    
    # Create SQL Query
    if USE_BEST_MATCHING_COLUMNS and best_columns:
        sql_prompt = create_sql_prompt(
            examples=examples, 
            best_matching_table=best_table_info, 
            columns_metadata=best_columns, 
            use_best_matching_columns=True
        )
    else:
        sql_prompt = create_sql_prompt(
            examples=examples, 
            best_matching_table=best_table_info, 
            columns_metadata=columns_info
        )

    # Initialize LLM and other components
    best_table = best_table_info['table_name']
    engine = create_engine(DATABASE_URL)
    db = SQLDatabase(engine=engine, ignore_tables=['table_metadata', 'column_metadata'])

    execute_query = QuerySQLDataBaseTool(db=db)
    write_query = create_sql_query_chain(llm, db, sql_prompt)

    # Create the answer chain
    answer_chain = create_answer_chain(llm)

    # Put everything together
    master_chain = (
        RunnablePassthrough.assign(query=write_query).assign(
            result=itemgetter("query") | execute_query
        )
        | answer_chain
    )

    # Invoke the master chain and return the response
    response = master_chain.invoke({
        "question": user_query, 
        "top_k": 3,
        "table_info": best_table
    })
    return response

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
 
def process_sql_query(user_question, use_huggingface=False):
    """
    Processes a user's question by generating and executing an SQL query using a language model (LLM). 
    Optionally, uses a Hugging Face model or defaults to OpenAI's GPT-3.5-turbo.

    This function first initializes the appropriate LLM based on the `use_huggingface` flag. It then 
    retrieves metadata information, identifies the best matching table and relevant columns, and 
    executes the SQL query based on the processed information.

    Parameters
    ----------
    user_question : str
        The user's question for which an SQL query needs to be generated and executed.
    use_huggingface : bool, optional
        A flag to determine whether to use a Hugging Face model instead of the default OpenAI model. 
        Default is False.

    Returns
    -------
    output : Any
        The output from the executed SQL query chain, typically containing the results of the SQL query.

    Example
    -------
    output = process_sql_query(
        user_question="What is the price of maize?",
        use_huggingface=False
    )
    print(output)
    """
    # Initialize LLM
    # To Do: add Hugging Face LLM
    if use_huggingface:
        pass  # Hugging Face LLM initialization can be added here
    else:
        openai_llm = ChatOpenAI(model="gpt-3.5-turbo", temperature=0, openai_api_key=OPENAI_API_KEY)
   
    # Retrieve the metadata info (tables and columns)
    tables, columns = connect_to_database()

    # Chain 1: Find the Best Table
    best_table_chain, context = find_best_table_prompt(user_question, tables, columns, llm=openai_llm)
    best_table_output_str = best_table_chain.run(**context)

    # Convert the string output to a dictionary
    try:
        best_table_output = json.loads(best_table_output_str)['best_matching_table']
    except json.JSONDecodeError:
        print("Error: The output is not valid JSON.")
        best_table_output = None

    # Chain 2: Find Relevant Columns
    best_columns_chain, context = find_best_columns_prompt(user_question, best_table_output, columns, llm=openai_llm)
    best_columns_output = best_columns_chain.run(**context)

    # Retrieve result 
    output = run_sql_chain_v2(
        user_query=user_question, 
        best_table_info=best_table_output, 
        columns_info=columns, 
        best_columns=best_columns_output, 
        llm=openai_llm
    )
    
    return output

def main():
    questions_en = ["Where can I find the cheapest maize?",
                    "Which district harvested the most beans?",
                    "How much is Maize in Zomba?",
                    "Which district produced more Tobacco, Mchinji or Kasungu?",
                    "Where can I get bananas?"]
    for q in questions_en:
        print("\n--- QUESTION: ", q)
        output = process_sql_query(q)
        print('OUTPUT TYPE==>', output)

if __name__ == '__main__':
    # this is to quite parallel tokenizers warning.
    main()

