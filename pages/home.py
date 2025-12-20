from statics import chart_funcs_dict as chart_funcs_dict
from statics import colors_dict as colors_dict
from statics import Agg_Funcs as Agg_Funcs
from statics import Options as Options
from statics import Colors as Colors
from statics import Charts as Charts
import streamlit as st
import pandas as pd
import numpy as np
import validators
import openpyxl
import requests
import duckdb
import os
import io
import re
import sys
from pathlib import Path
from google.oauth2 import service_account
from google.cloud import bigquery


# required to import statics from parent directory
dir = Path(__file__).resolve().parent
sys.path.insert(0, str(dir))

# create columns metadata and statistics


def get_column_metadata(df, ncols, scols):
    # save dataframe row and column count for display
    st.session_state.row_count = df.shape[0]
    st.session_state.column_count = df.shape[1]

    # create dictionary for dataframe metadata
    metadata = {
        'Data Type': df.dtypes.astype('str'),
        'Null Count': df.isnull().sum(),
        'Unique Values': df.nunique()
    }

    for col in df.columns:

        if col in st.session_state.column_datatype_dict:
            metadata['Data Type'][col] = (
                st.session_state.column_datatype_dict[col]).lower()

        if col in scols:
            metadata['Empty String Count'] = metadata.get(
                'Empty String Count', pd.Series(dtype=int))
            metadata['Empty String Count'][col] = (
                df[col].astype(str).str.strip() == '').sum()

        if col in ncols:
            metadata['Zero Count'] = metadata.get(
                'Zero Count', pd.Series(dtype=int))
            metadata['Zero Count'][col] = (df[col].astype(float) == 0.0).sum()

            metadata['Mean'] = metadata.get('Mean', pd.Series(dtype=float))
            metadata['Mean'][col] = df[col].mean()

            metadata['Std Dev'] = metadata.get(
                'Std Dev', pd.Series(dtype=float))
            metadata['Std Dev'][col] = df[col].std()

            metadata['Min'] = metadata.get('Min', pd.Series(dtype=float))
            metadata['Min'][col] = df[col].min()

            metadata['Max'] = metadata.get('Max', pd.Series(dtype=float))
            metadata['Max'][col] = df[col].max()

            metadata['Percentile_25'] = metadata.get(
                'Percentile_25', pd.Series(dtype=float))
            metadata['Percentile_25'][col] = df[col].quantile(0.25)

            metadata['Percentile_50'] = metadata.get(
                'Percentile_50', pd.Series(dtype=float))
            metadata['Percentile_50'][col] = df[col].quantile(0.50)

            metadata['Percentile_75'] = metadata.get(
                'Percentile_75', pd.Series(dtype=float))
            metadata['Percentile_75'][col] = df[col].quantile(0.75)

    # Create DataFrame with metrics as index
    metadata_df = pd.DataFrame(metadata)
    return metadata_df


# initialize in-memory duckdb database
@st.cache_resource
def get_duckdb_connection():
    return duckdb.connect(database=':memory:', read_only=False)


def on_selection_change(col):  # function for table and column transformation
    selection = st.session_state[col]
    st.session_state.reset = True
    match selection:
        case Options.rmc:
            st.session_state.df = st.session_state.df.drop(col, axis=1)
        case Options.rnc:
            dialog(col, 'rename', f'Rename {col}', 'Enter new column name:')
        case Options.spc:
            dialog(col, 'split', f'Split {col}', 'Enter delimiter:')
        case Options.cat:
            reset_data()
        case Options.pvt:
            pivot_dialog()
        case Options.sfh:
            row = st.session_state.df.iloc[1]
            is_string_array = all(isinstance(item, str) for item in row)
            if not is_string_array:
                st.error('Invalid operation')
            else:
                h = st.session_state.df.iloc[0]
                st.session_state.df = st.session_state.df[1:]
                st.session_state.df.columns = h
        case Options.rnz:
            st.session_state.df[col].fillna(0, inplace=True)
        case Options.rnm:
            st.session_state.df[col].fillna(
                st.session_state.df[col].mean(), inplace=True)
        case Options.fed | Options.fnd:
            st.session_state.df[col].fillna(method='ffill', inplace=True)
        case Options.feu | Options.fnu:
            st.session_state.df[col].fillna(method='bfill', inplace=True)
        case Options.rnr:
            st.session_state.df.dropna(subset=[col], inplace=True)
        case Options.rer:
            st.session_state.df[col].replace('', np.nan, inplace=True)
            st.session_state.df.dropna(inplace=True)
        case Options.ccs:
            st.session_state.df.astype({col: str}, inplace=True)
        case Options.cci:
            st.session_state.df.astype({col: int}, inplace=True)
        case Options.ccd:
            st.session_state.df.astype({col: float}, inplace=True)
        case Options.trt:
            st.session_state.df = st.session_state.df.T
        case Options.cdc:
            if 'NewDate' not in st.session_state.df.columns:
                if st.session_state.df[col].dtype == 'object':
                    st.session_state.df['NewDate'] = pd.to_datetime(
                        st.session_state.df[col])
                elif st.session_state.df[col].dtype == np.int64:
                    numeric_date_conversion(
                        col, 'Convert numerical date to date type')
        case Options.cts:
            if 'NewDateFromTimestamp' not in st.session_state.df.columns:
                if st.session_state.df[col].dtype == 'object':
                    st.session_state.df['NewDateFromTimestamp'] = pd.to_datetime(
                        st.session_state.df[col])
                elif st.session_state.df[col].dtype == np.int64:
                    st.session_state.df['NewDateFromTimestamp'] = pd.to_datetime(
                        st.session_state.df[col], unit='s')
        case Options.cyc:
            if 'Year' not in st.session_state.df.columns:
                st.session_state.df['Year'] = st.session_state.df[col].dt.year
        case Options.cqc:
            if 'Quarter' not in st.session_state.df.columns:
                st.session_state.df['Quarter'] = st.session_state.df[col].dt.quarter
        case Options.cmc:
            if 'Month' not in st.session_state.df.columns:
                st.session_state.df['Month'] = st.session_state.df[col].dt.month
        case Options.cdy:
            if 'Day' not in st.session_state.df.columns:
                st.session_state.df['Day'] = st.session_state.df[col].dt.date
        case Options.ctc:
            if 'Time' not in st.session_state.df.columns:
                st.session_state.df['Time'] = st.session_state.df[col].dt.time
        case Options.asc:
            st.session_state.df.sort_values(by=col, inplace=True)
        case Options.dsc:
            st.session_state.df.sort_values(
                by=col, ascending=False, inplace=True)
        case Options.gbc:
            group_by()
        case Options.fbc:
            filter_table()


@st.dialog(' ')
# create datetime column from numeric value
def numeric_date_conversion(col, msg):
    st.write(msg)
    dialog_input = st.text_input(
        'Enter date format on column ex. yyyymmdd:', key='date').lower().strip()

    if st.button('Submit') and len(dialog_input) <= 10:

        year = ''
        month = ''
        day = ''
        delimiter = ''

        for c in dialog_input:
            if c == 'y':
                year += c
            elif c == 'm':
                month += c
            elif c == 'd':
                day += c

        if year and month and day:
            dialog_input = dialog_input.replace(
                year, '%Y').replace(month, '%m').replace(day, '%d')
        else:
            st.error('Invalid input')

        try:
            st.session_state.df['NewDate'] = pd.to_datetime(
                st.session_state.df[col], format=dialog_input)
            st.rerun()
        except Exception as e:
            st.error('Invalid input')
            st.stop()


@st.dialog(' ')
def pivot_dialog():  # to pivot table
    st.write('Pivot Table')
    indices = st.multiselect(f'Index:', options=st.session_state.df.columns)
    column = st.selectbox(f'Column:', options=st.session_state.df.columns)
    values = st.selectbox(f'Values:', options=st.session_state.df.columns)
    agg_func = st.selectbox(f'Aggregate function:', options=[
                            e.value for e in Agg_Funcs]).lower()
    if st.button('Submit') and indices:
        if column in indices or values in indices or column == values:
            st.error('Invalid selections')
        else:
            st.session_state.df = st.session_state.df.pivot_table(
                values=values,
                index=indices,
                columns=column,
                aggfunc=agg_func
            )
            st.rerun()


@st.dialog(' ')
def group_by():  # data transformation: group by
    st.write('Group by')
    columns = st.multiselect(f'Select column(s):',
                             options=st.session_state.df.columns)
    values_column = st.selectbox(
        f'Select column to aggregate:', options=st.session_state.df.columns)
    agg_func_array = [e.value for e in Agg_Funcs]
    agg_func = st.selectbox(f'Select aggregation function:',
                            options=agg_func_array).lower()
    if st.button('Submit'):
        if values_column in columns and len(columns) < 1:
            st.error('Invalid operation')
        else:
            st.session_state.df = st.session_state.df.groupby(
                columns)[values_column].agg(agg_func).reset_index()
            st.rerun()


@st.dialog(' ')
def filter_table():  # data transformation: filter
    operators = {
        'equal to': '==',
        'not equal to': '!=',
        'less than': '<',
        'greater than': '>',
        'less than or equal to': '<=',
        'greater than or equal to': '>='
    }

    st.write('Filter')
    column = st.selectbox(
        f'Select column:', options=st.session_state.df.columns)
    operator = st.selectbox(f'Select operator:', options=operators.keys())
    dialog_input = st.text_input('Enter value', key='filter').lower().strip()

    if st.button('Submit') and dialog_input:
        st.session_state.df = st.session_state.df.query(
            f'{column} {operators[operator]} {dialog_input}')
        st.rerun()


@st.dialog(' ')
def dialog(col, kind, msg, prompt):  # to rename or split column
    st.write(msg)
    dialog_input = st.text_input(prompt, key=kind).strip()
    if st.button('Submit'):
        if len(dialog_input) > 50 or len(dialog_input) == 0:
            st.error('Invalid input')
        else:
            match kind:
                case 'rename':
                    st.session_state.df = st.session_state.df.rename(
                        columns={col: dialog_input})
                    st.rerun()
                case 'split':
                    if len(dialog_input) != 1:
                        st.error(
                            'Delimiter must be a space or any single character')
                    else:
                        st.session_state.df[['Column0', 'Column1']] = st.session_state.df[col].str.split(
                            dialog_input, expand=True)
                        st.rerun()


# create duckdb table from pandas dataframe
def create_duckdb_table_from_dataframe(df):
    st.session_state.cxtn.execute('DROP TABLE IF EXISTS duckdb_table;')
    st.session_state.cxtn.from_df(df).create('duckdb_table')
    st.rerun()


def load_dataframe(loaded_file, file_ext):  # load uploaded file

    if file_ext not in ['.xlsx', 'xls', '.parquet', '.csv']:
        st.error(
            f'Invalid file type: {file_ext}. Please upload a .csv, .parquet or Excel file (.xlsx or .xls).')
        st.stop()
    match file_ext:
        case '.csv':
            chunk_size = 10000
            chunks_array = []
            chunks = pd.read_csv(loaded_file, chunksize=chunk_size)
            for chunk in chunks:
                chunks_array.append(chunk)
            st.session_state.df = pd.concat(chunks_array, ignore_index=True)
        case '.parquet':
            st.session_state.df = pd.read_parquet(loaded_file)
        case '.xlsx' | '.xls':
            st.session_state.df = pd.read_excel(loaded_file, engine='openpyxl')
    if not st.session_state.df.empty:
        create_duckdb_table_from_dataframe(st.session_state.df)


def download_file():  # download from cloud storage
    selection = st.session_state.file_type.lower()
    url = st.session_state.url
    if 'drive.google.com' in url and ('sharing' in url or 'drive_link' in url):
        file_id = re.search(r'/d/([a-zA-Z0-9_-]+)', url).group(1)
        url = f'https://drive.google.com/uc?export=download&id={file_id}'
    if 'docs.google.com' in url:
        sheet_name = 'sheet1'
        sheet_id = re.search(r'/d/([a-zA-Z0-9_-]+)', url).group(1)
        url = f'https://docs.google.com/spreadsheets/d/{sheet_id}/gviz/tq?tqx=out:csv&sheet={sheet_name}'
        selection = 'csv'
    if '1drv.ms' in url:
        url = f'{url}&download=1'
    # Download the file
    try:
        response = requests.get(url)
        response.raise_for_status()
        match selection:
            case 'csv':
                chunk_size = 10000
                chunks_array = []
                chunks = pd.read_csv(
                    io.StringIO(response.content.decode('utf-8')), chunksize=chunk_size)
                for chunk in chunks:
                    chunks_array.append(chunk)
                st.session_state.df = pd.concat(
                    chunks_array, ignore_index=True)
            case 'parquet':
                st.session_state.df = pd.read_parquet(
                    io.StringIO(response.content.decode('utf-8')))
            case 'excel':
                st.session_state.df = pd.read_excel(
                    io.BytesIO(response.content), engine='openpyxl')
        if not st.session_state.df.empty:
            create_duckdb_table_from_dataframe(st.session_state.df)

    except Exception as e:
        st.error('Error importing file or invallid file format')
        st.stop()


def on_chart_selection_change(chart_key):  # selection of chart type
    i = chart_key.split()[1]
    match st.session_state[chart_key]:
        case Charts.LCH.value | Charts.ACH.value | Charts.BCH.value | Charts.SCH.value:
            st.session_state[f'pie chart {i}'] = False
            st.session_state[f'histogram chart {i}'] = False
        case Charts.PCH.value | Charts.DCH.value:
            st.session_state[f'pie chart {i}'] = True
            st.session_state[f'histogram chart {i}'] = False
        case Charts.HCH.value:
            st.session_state[f'histogram chart {i}'] = True
            st.session_state[f'pie chart {i}'] = False


# options for string columns
string_options = [Options.stm, Options.cts, Options.cci, Options.ccd, Options.cdc, Options.rer,
                  Options.rnc, Options.spc, Options.asc, Options.dsc, Options.rmc]
# options for numeric columns
numeric_options = [Options.stm, Options.rnz, Options.rnm, Options.cts, Options.rnr, Options.fnd,
                   Options.fnu, Options.ccs, Options.cdc, Options.rnc, Options.asc, Options.dsc,
                   Options.rmc]
# options for date columns
datetime_options = [Options.stm, Options.cyc, Options.cqc, Options.cmc, Options.cdy, Options.ctc,
                    Options.asc, Options.dsc, Options.rmc]


def init_state():
    st.session_state.schema_df = pd.DataFrame()
    st.session_state.df = pd.DataFrame()
    cancel()


def render_chart(i, update=False):  # rendering charts of dashboard page
    match_axis_colors(i)
    st.session_state[f'chart_settings_{i}'][f'chart title {i}'] = st.session_state[f'chart title {i}']
    st.session_state[f'chart_settings_{i}'][f'chart dataframe {i}'] = st.session_state.df
    if st.session_state[f'histogram chart {i}']:
        st.session_state[f'chart_settings_{i}'][f'chart {i}'] = st.session_state[f'chart {i}']
        st.session_state[f'chart_settings_{i}'][f'x-axis {i}'] = st.session_state[f'x-axis {i}']
        st.session_state[f'chart_settings_{i}'][f'y-axis {i}'] = st.session_state[f'y-axis {i}']
        st.session_state[f'chart_settings_{i}'][f'bins {i}'] = st.session_state[f'bins {i}']
    elif st.session_state[f'pie chart {i}']:
        st.session_state[f'chart_settings_{i}'][f'chart {i}'] = st.session_state[f'chart {i}']
        st.session_state[f'chart_settings_{i}'][f'labels {i}'] = st.session_state[f'labels {i}']
        st.session_state[f'chart_settings_{i}'][f'values {i}'] = st.session_state[f'values {i}']
    else:
        st.session_state[f'chart_settings_{i}'][f'chart {i}'] = st.session_state[f'chart {i}']
        st.session_state[f'chart_settings_{i}'][f'x-axis {i}'] = st.session_state[f'x-axis {i}']
        st.session_state[f'chart_settings_{i}'][f'y-axis {i}'] = st.session_state[f'y-axis {i}']
        st.session_state[f'chart_settings_{i}'][f'color {i}'] = st.session_state[f'color {i}']
    if not update:
        st.session_state.charts_array.append(
            st.session_state[f'chart_settings_{i}'])
    else:
        st.session_state.charts_array[i] = st.session_state[f'chart_settings_{i}']

    st.switch_page('pages/dashboard.py')


def validate_input(i):  # validate all required inputs are present
    if st.session_state[f'chart title {i}'] and (st.session_state[f'chart {i}'] and st.session_state[f'x-axis {i}']
                                                 and st.session_state[f'y-axis {i}']) or (st.session_state[f'chart {i}']
                                                                                          and st.session_state[f'labels {i}'] and st.session_state[f'values {i}']) or (st.session_state[f'chart {i}']
                                                                                                                                                                       and st.session_state[f'x-axis {i}'] and st.session_state[f'y-axis {i}'] and st.session_state[f'color {i}']):
        return True
    else:
        return False


def match_axis_colors(i):  # match selected colors with y-axis
    if f'y-axis {i}' in st.session_state and st.session_state[f'color {i}']:
        if len(st.session_state[f'y-axis {i}']) != len(st.session_state[f'color {i}']):
            st.error('y-axis selections must be equal to selected colors')
            st.stop()


def set_warehouse():
    if st.session_state.data_warehouse != '--':
        st.session_state.dw = st.session_state.data_warehouse
        st.session_state.query_warehouse = True
        st.session_state.snowflake_schemas = ['--']
        st.session_state.snowflake_tables = ['--']
        st.session_state.bigquery_tables = ['--']


# Create BigQuery API client.
credentials = service_account.Credentials.from_service_account_info(
    st.secrets["gcp_service_account"]
)
client = bigquery.Client(credentials=credentials)


def get_table_data(ls):

    comma_sep_colnames = ", ".join(ls)

    if st.session_state.dw == 'BIGQUERY':
        table_name = f'`{st.session_state.bigquery_project_id}.{st.session_state.ds}.{st.session_state.bigquery_tbl}`'

    if st.session_state.dw == 'SNOWFLAKE':
        table_name = f'{st.session_state.db}.{st.session_state.sch}.{st.session_state.snowflake_tbl}'

    query = f'''
                SELECT
                    {comma_sep_colnames}
                FROM
                    {table_name}
                LIMIT
                    {st.session_state.no_of_rows};
            '''

    st.session_state.df = run_query(query)
    st.session_state.cxtn.execute('DROP TABLE IF EXISTS duckdb_table;')
    st.session_state.cxtn.from_df(st.session_state.df).create('duckdb_table')

    st.rerun()


def run_query(query):
    try:
        if st.session_state.dw == 'BIGQUERY':
            query = client.query(query)
            return query.to_dataframe()
        if st.session_state.dw == 'SNOWFLAKE':
            cxtn = st.connection("snowflake")
            return cxtn.query(query)
    except Exception as e:
        st.error('Connection error')
        st.stop()


def reset_data():
    st.session_state.df = st.session_state.cxtn.execute(
        'SELECT * FROM duckdb_table;').fetchdf()
    st.session_state.reset = False


def get_snowflake_table_columns():
    if st.session_state.snowflake_table != '--':
        # save table selection
        st.session_state.snowflake_tbl = st.session_state.snowflake_table
        df = st.session_state.snowflake_result_df[[
            'COLUMN_NAME', 'DATA_TYPE', 'IS_NULLABLE', 'TABLE_NAME', 'TABLE_SCHEMA']]
        st.session_state.schema_df = df[(df['TABLE_NAME'] == f'{st.session_state.snowflake_table}') &
                                        (df['TABLE_SCHEMA'] == f'{st.session_state.snowflake_schema}')]


def get_bigquery_table_columns():
    if st.session_state.bigquery_table != '--':
        # save table selection
        st.session_state.bigquery_tbl = st.session_state.bigquery_table
        df = st.session_state.bigquery_result_df[[
            'column_name', 'data_type', 'is_nullable', 'table_name']]
        st.session_state.schema_df = df[(
            df['table_name'] == f'{st.session_state.bigquery_tbl}')]


def get_snowflake_tables():
    if st.session_state.snowflake_schema != '--':
        # save schema selection
        st.session_state.sch = st.session_state.snowflake_schema
        st.session_state.snowflake_tables = ['--']
        st.session_state.snowflake_tables = st.session_state.snowflake_tables + \
            st.session_state.snowflake_result_df['TABLE_NAME'].unique(
            ).tolist()


def get_snowflake_schemas():
    if st.session_state.database != '--':
        # save database selection
        st.session_state.db = st.session_state.database
        query = f'''SELECT
                        table_schema, table_name, column_name, data_type, is_nullable
                    FROM
                        {st.session_state.db}.INFORMATION_SCHEMA.COLUMNS
                    WHERE NOT
                        table_schema = 'INFORMATION_SCHEMA'
                ;'''

        st.session_state.snowflake_result_df = run_query(query)
        st.session_state.snowflake_schemas = ['--']
        st.session_state.snowflake_schemas = st.session_state.snowflake_schemas + \
            st.session_state.snowflake_result_df['TABLE_SCHEMA'].unique(
            ).tolist()


def get_bigquery_tables():
    if st.session_state.dataset != '--':
        # save dataset selection
        st.session_state.ds = st.session_state.dataset
        query = f'''SELECT
                        table_name, column_name, data_type, is_nullable
                    FROM
                        `{st.session_state.bigquery_project_id}.{st.session_state.ds}.INFORMATION_SCHEMA.COLUMNS`;
                '''

        st.session_state.bigquery_result_df = run_query(query)
        st.session_state.bigquery_tables = ['--']
        st.session_state.bigquery_tables = st.session_state.bigquery_tables + \
            st.session_state.bigquery_result_df['table_name'].unique().tolist()


def cancel():
    st.session_state.query_warehouse = False
    st.session_state.snowflake_schemas = ['--']
    st.session_state.snowflake_tables = ['--']
    st.session_state.bigquery_tables = ['--']
    st.rerun()


def main():

    st.set_page_config(layout='wide')

    # set Snowflake database id
    snowflake_databases = st.secrets['connections']['snowflake'].databases

    if 'snowflake_databases' not in st.session_state:
        st.session_state.snowflake_databases = snowflake_databases

    # set Bigquery project id
    bigquery_project_id = st.secrets['gcp_service_account'].project_id

    if 'bigquery_project_id' not in st.session_state:
        st.session_state.bigquery_project_id = bigquery_project_id

    # set Bigquery dataset
    bigquery_datasets = st.secrets['gcp_service_account'].datasets

    if 'databases' not in st.session_state:
        st.session_state.databases = ['--'] + snowflake_databases

    if 'datasets' not in st.session_state:
        st.session_state.datasets = ['--'] + bigquery_datasets

    if 'df' not in st.session_state:
        st.session_state.df = pd.DataFrame()

    if 'schema_df' not in st.session_state:
        st.session_state.schema_df = pd.DataFrame()

    if 'snowflake_tables' not in st.session_state:
        st.session_state.snowflake_tables = ['--']

    if 'snowflake_schemas' not in st.session_state:
        st.session_state.snowflake_schemas = ['--']

    if 'bigquery_tables' not in st.session_state:
        st.session_state.bigquery_tables = ['--']

    if 'metadata_df' not in st.session_state:
        st.session_state.metadata_df = []

    if 'query_warehouse' not in st.session_state:
        st.session_state.query_warehouse = False

    # create duckdb connector
    if 'cxtn' not in st.session_state:
        st.session_state.cxtn = get_duckdb_connection()

    if 'reset' not in st.session_state:
        st.session_state.reset = False

    if 'dw' not in st.session_state:
        st.session_state.dw = False

    if 'column_datatype_dict' not in st.session_state:
        st.session_state.column_datatype_dict = {}

    st.markdown('##### Simple tool for exploratory data analysis')

    tab1, tab2, tab3, tab4 = st.tabs(
        ['Ingest Data', 'Transform Data', 'Data Table', 'Create Charts'])

    # data ingest tab
    with tab1:
        if not st.session_state.df.empty:
            # get column data types
            numeric_cols = st.session_state.df.select_dtypes(
                include=[np.number]).columns
            string_cols = st.session_state.df.select_dtypes(
                include='object').columns
            datetime_cols = st.session_state.df.select_dtypes(
                include='datetime64').columns

            for col in string_cols:
                st.session_state.column_datatype_dict[col] = 'String'

            for col in datetime_cols:
                st.session_state.column_datatype_dict[col] = 'Datetime'

            metadata_df = get_column_metadata(
                st.session_state.df, numeric_cols, string_cols)

            if st.session_state.reset:
                with st.container(horizontal=True, horizontal_alignment='right'):
                    if st.button('Reset data'):
                        reset_data()
                        st.rerun()

            with st.container(horizontal=True):
                st.write(f'Row Count: {st.session_state.row_count}')
                st.write(f'Column Count: {st.session_state.column_count}')

            st.dataframe(
                metadata_df,
                column_config={
                    col: st.column_config.Column(
                        col,
                        width='medium'
                    ) for col in metadata_df.columns
                }
            )
            with st.container():
                if st.button('Clear data'):
                    init_state()
                    st.rerun()
        elif st.session_state.query_warehouse:
            with st.container(border=True):
                if st.session_state.schema_df.empty:
                    if st.session_state.dw == 'BIGQUERY':
                        options = st.session_state.datasets
                        st.selectbox('Select dataset:', options=options,
                                     on_change=get_bigquery_tables, key='dataset')
                        if len(st.session_state.bigquery_tables) > 1:
                            options = st.session_state.bigquery_tables
                            st.selectbox('Select table:', options=options,
                                         on_change=get_bigquery_table_columns, key='bigquery_table')
                        with st.container(horizontal=True):
                            if st.button('Cancel', key='cancel0'):
                                cancel()
                    if st.session_state.dw == 'SNOWFLAKE':
                        options = st.session_state.databases
                        st.selectbox('Select database:', options=options,
                                     on_change=get_snowflake_schemas, key='database')
                        if len(st.session_state.snowflake_schemas) > 1:
                            options = st.session_state.snowflake_schemas
                            st.selectbox('Select schema:', options=options,
                                         on_change=get_snowflake_tables, key='snowflake_schema')
                        if len(st.session_state.snowflake_tables) > 1:
                            options = st.session_state.snowflake_tables
                            st.selectbox('Select table:', options=options,
                                         on_change=get_snowflake_table_columns, key='snowflake_table')
                        with st.container(horizontal=True):
                            if st.button('Cancel', key='cancel1'):
                                cancel()
                else:
                    with st.container():
                        checkbox_selections_dict = {}
                        st.write('Select colomn(s):')

                        column_name = 'column_name'
                        data_type = 'data_type'
                        is_nullable = 'is_nullable'

                        if st.session_state.dw == 'SNOWFLAKE':
                            column_name = column_name.upper()
                            data_type = data_type.upper()
                            is_nullable = is_nullable.upper()

                        for idx, row in st.session_state.schema_df.iterrows():
                            d_type = 'String' if row[data_type] == 'TEXT' else row[data_type].capitalize(
                            )
                            st.checkbox(
                                f'{row[column_name]} {d_type} | {"Nullable" if row[is_nullable] else "Not Nullable"}', key=f'{row}_{idx}')
                            checkbox_selections_dict[row[column_name]
                                                     ] = st.session_state[f'{row}_{idx}']
                            # save column data type in a dictionary
                            st.session_state.column_datatype_dict[column_name] = d_type

                        nstep = 1000
                        nrange = 100
                        options = [i * nstep for i in range(1, nrange + 1)]
                        st.selectbox(f'Select number of rows',
                                     options=options, key='no_of_rows')
                        with st.container(horizontal=True):
                            if st.button('Get data'):
                                columns_list = [
                                    i for i, v in checkbox_selections_dict.items() if v]
                                get_table_data(columns_list)
                            if st.button('Cancel'):
                                st.session_state.query_warehouse = False
                                st.session_state.schema_df = pd.DataFrame()
                                st.rerun()

        else:
            # upload widget container
            with st.container(border=True):
                try:
                    uploaded_file = st.file_uploader('Upload file:',  # upload widget
                                                     accept_multiple_files=False,
                                                     on_change=init_state,
                                                     type=[
                                                         'csv', 'parquet', 'xlsx', 'xls']
                                                     )
                    if uploaded_file:
                        file_name = uploaded_file.name
                        file_ext = os.path.splitext(file_name)[1].lower()
                        load_dataframe(uploaded_file, file_ext)

                except Exception as e:
                    st.error('File upload error')
                    st.stop()

            # import file container
            with st.container(border=True):
                url_input = st.text_input(
                    'Enter file url:', key='url').lower().strip()
                st.selectbox(f'Select file type:', options=[
                    'CSV', 'PARQUET', 'EXCEL'], key='file_type')
                if st.button('Import file') and validators.url(url_input):
                    download_file()

            # import warehouse container
            with st.container(border=True):
                st.selectbox(f'Select data warehouse:', options=[
                    '--', 'BIGQUERY', 'SNOWFLAKE'],
                    on_change=set_warehouse,
                    key='data_warehouse')

    # transform data tab
    with tab2:

        if not st.session_state.df.empty:
            # options for table transform
            options = [Options.stm, Options.sfh, Options.cat,
                       Options.gbc, Options.trt, Options.fbc, Options.pvt]
            st.selectbox(f'Table Transform',
                         options=options,
                         key='table',
                         on_change=on_selection_change,
                         args=['table'])

        for col in st.session_state.df.columns:
            if col in datetime_cols:
                # date column transform options
                options = datetime_options
            elif col in numeric_cols:
                # numeric column transform options
                options = numeric_options
            elif col in string_cols:
                # string column transform options
                options = string_options
            st.selectbox(f'{col} Column Transform',
                         options=options,
                         key=col,
                         on_change=on_selection_change,
                         args=[col])

    # data table tab
    with tab3:
        if not st.session_state.df.empty:
            st.dataframe(st.session_state.df)

    # create chart tab
    with tab4:
        if 'numberof_charts' not in st.session_state:
            st.session_state.numberof_charts = 0

        if 'charts_array' not in st.session_state:
            st.session_state.charts_array = []

        if not st.session_state.numberof_charts:
            # container for add chart
            with st.container(horizontal=True, horizontal_alignment='right'):
                if st.button(':material/add: Add Chart', key=f'add_button_{0}'):
                    st.session_state.numberof_charts += 1
                    st.rerun()
        else:

            for i in range(st.session_state.numberof_charts):

                if f'chart_settings_{i}' not in st.session_state:
                    st.session_state[f'chart_settings_{i}'] = {}

                if f'chart {i}' not in st.session_state:
                    st.session_state[f'chart {i}'] = None

                if f'x-axis {i}' not in st.session_state:
                    st.session_state[f'x-axis {i}'] = None

                if f'y-axis {i}' not in st.session_state:
                    st.session_state[f'y-axis {i}'] = []

                if f'color {i}' not in st.session_state:
                    st.session_state[f'color {i}'] = []

                if f'labels {i}' not in st.session_state:
                    st.session_state[f'labels {i}'] = None

                if f'values {i}' not in st.session_state:
                    st.session_state[f'values {i}'] = None

                if f'pie chart {i}' not in st.session_state:
                    st.session_state[f'pie chart {i}'] = False

                if f'histogram chart {i}' not in st.session_state:
                    st.session_state[f'histogram chart {i}'] = False

                if f'bins {i}' not in st.session_state:
                    st.session_state[f'bins {i}'] = 0

                chart = {}

                # condition to display existing chart update forms
                if len(st.session_state.charts_array) == st.session_state.numberof_charts:
                    chart = st.session_state.charts_array[i]

                # condition to display create chart form
                elif st.session_state.numberof_charts != i + 1:
                    continue

                with st.expander(f'Chart {i + 1}', expanded=True):
                    # create chart form
                    st.text_input('Enter a descriptive title for chart:*',
                                  key=f'chart title {i}')

                    if chart:
                        st.success(chart[f'chart title {i}'])

                    chart_type_array = [e.value for e in Charts]
                    st.selectbox('Select chart type:*', options=chart_type_array,
                                 key=f'chart {i}',
                                 on_change=on_chart_selection_change,
                                 args=[f'chart {i}'])

                    if f'chart {i}' in chart:
                        st.success(chart[f'chart {i}'])

                    if st.session_state[f'pie chart {i}']:

                        st.selectbox('Select labels:*', options=st.session_state.df.columns,
                                     key=f'labels {i}')

                        if f'labels {i}' in chart:
                            st.success(chart[f'labels {i}'])

                        st.selectbox('Select values:*', options=st.session_state.df.columns,
                                     key=f'values {i}')

                        if f'values {i}' in chart:
                            st.success(chart[f'values {i}'])

                    elif st.session_state[f'histogram chart {i}']:

                        st.selectbox('Select x-axis:*', options=st.session_state.df.columns,
                                     key=f'x-axis {i}')

                        if f'x-axis {i}' in chart:
                            st.success(chart[f'x-axis {i}'])

                        st.multiselect('Select y-axis:*', options=st.session_state.df.columns,
                                       key=f'y-axis {i}')

                        if f'x-axis {i}' in chart:
                            st.success(chart[f'y-axis {i}'])

                        st.number_input('Enter bin size:*', key=f'bins {i}')

                        if f'bins {i}' in chart:
                            st.success(chart[f'bins {i}'])
                    else:

                        st.selectbox('Select x-axis:*', options=st.session_state.df.columns,
                                     key=f'x-axis {i}')

                        if f'x-axis {i}' in chart:
                            st.success(chart[f'x-axis {i}'])

                        st.multiselect('Select y-axis:*', options=st.session_state.df.columns,
                                       key=f'y-axis {i}')

                        if f'y-axis {i}' in chart:
                            st.success(chart[f'y-axis {i}'])

                        colors_array = [e.value for e in Colors]
                        st.multiselect('Select color(s):*', options=colors_array,
                                       key=f'color {i}')

                        if f'color {i}' in chart:
                            st.success(chart[f'color {i}'])

                    # container for form buttons
                    with st.container(horizontal=True, horizontal_alignment='right', vertical_alignment='center'):

                        # condition to display buttons for last chart update form
                        if len(st.session_state.charts_array) == st.session_state.numberof_charts\
                                and st.session_state.numberof_charts == i + 1:

                            if st.button(':material/add: Add Chart', key=f'add_button_{i}'):
                                if st.session_state.numberof_charts < 4:
                                    st.session_state.numberof_charts += 1
                                    st.rerun()

                            if st.button(':material/remove: Remoove Chart', key=f'remove_button_{i}'):
                                st.session_state.charts_array.pop()
                                st.session_state.numberof_charts -= 1
                                st.rerun()

                            validate = validate_input(i)
                            if validate:
                                if st.button('Update', key=f'update_button_{i}'):
                                    render_chart(i, True)

                        # condition to display button for create chart form
                        elif len(st.session_state.charts_array) < st.session_state.numberof_charts\
                                and st.session_state.numberof_charts == i + 1:

                            if st.button('Cancel', key=f'cancel_button_{i}'):
                                st.session_state.numberof_charts -= 1
                                st.rerun()

                            validate = validate_input(i)
                            if validate:
                                if st.button('Render Chart', key=f'render_button_{i}'):
                                    render_chart(i)

                        # condition to display buttons chart for any other form
                        else:
                            if st.button('Cancel', key=f'cancel1_button_{i}'):
                                st.switch_page('pages/dashboard.py')

                            validate = validate_input(i)
                            if validate:
                                if st.button('Update', key=f'update_button_{i}'):
                                    render_chart(i, True)


if __name__ == '__main__':
    main()
