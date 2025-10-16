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
        'Data Type': df.dtypes.astype(str),
        'Null Count': df.isnull().sum(),
        'Unique Values': df.nunique()
    }

    for col in df.columns:
        if col in scols:
            metadata['Empty String Count'] = metadata.get(
                'Empty String Count', pd.Series(dtype=object))
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


def on_selection_change(col):  # function for table and column transformation
    selection = st.session_state[col]
    match selection:
        case Options.rmc:
            st.session_state.df = st.session_state.df.drop(col, axis=1)
        case Options.rnc:
            dialog(col, 'rename', f'Rename {col}', 'Enter new column name:')
        case Options.spc:
            dialog(col, 'split', f'Split {col}', 'Enter delimiter:')
        case Options.cat:
            st.session_state.df = st.session_state.dataframe
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
    agg_func = st.selectbox(f'Select aggregation function:', options=[
                            e.value for e in Agg_Funcs]).lower()
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


def load_dataframe(loaded_file, file_ext):  # load uploaded file

    file_ext = file_ext

    if file_ext not in ['.xlsx', 'xls', '.parquet', '.csv']:
        st.error(
            f'Invalid file type: {file_ext}. Please upload a .csv, .parquet or Excel file (.xlsx or .xls).')
        st.stop()

    if file_ext == '.csv':
        st.session_state.dataframe = pd.read_csv(loaded_file)
    elif file_ext == '.parquet':
        st.session_state.dataframe = pd.read_parquet(loaded_file)
    elif file_ext in ['.xlsx', '.xls']:
        st.session_state.dataframe = pd.read_excel(
            loaded_file, engine='openpyxl')

    st.session_state.df = st.session_state.dataframe
    st.rerun()


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
                st.session_state.dataframe = pd.read_csv(
                    io.StringIO(response.content.decode('utf-8')))
            case 'parquet':
                st.session_state.dataframe = pd.read_parquet(
                    io.StringIO(response.content.decode('utf-8')))
            case 'excel':
                st.session_state.dataframe = pd.read_excel(
                    io.BytesIO(response.content), engine='openpyxl')
        if not st.session_state.dataframe.empty:
            st.session_state.df = st.session_state.dataframe
            st.rerun()

    except Exception as e:
        st.error('Error importing file or invallid file format')
        st.stop()

def on_chart_selection_change(settings):  # selection of chart type
    h, i = settings.split()
    match st.session_state[f'chart {i}']:
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
    st.session_state.dataframe = pd.DataFrame()
    st.session_state.schema_df = pd.DataFrame()
    st.session_state.df = pd.DataFrame()
    st.session_state.query_warehouse = False


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
    if st.session_state.data_warehouse is not '--':
        st.session_state.dw = st.session_state.data_warehouse
        st.session_state.query_warehouse = True

# Create BigQuery API client.
credentials = service_account.Credentials.from_service_account_info(
    st.secrets["gcp_service_account"]
)
client = bigquery.Client(credentials=credentials)

def get_table_schema(wh_input_01,  wh_input_02, table_id):
       
    st.session_state.wh_input_01 = wh_input_01 # project or dataset id
    st.session_state.wh_input_02 = wh_input_02 # dataset or schema id
    st.session_state.table_id = table_id
    
    if st.session_state.dw == 'BigQuery':
        query = f"""
                    SELECT
                        column_name,
                        data_type,
                        is_nullable
                    FROM
                        `{st.session_state.wh_input_01}.{st.session_state.wh_input_02}.INFORMATION_SCHEMA.COLUMNS`
                    WHERE
                        table_name = '{st.session_state.table_id}'
                    ORDER BY
                        ordinal_position;
                """
    else:        
        query = f"""
                    SELECT
                        column_name,
                        data_type,
                        is_nullable
                    FROM
                        {st.session_state.wh_input_01}.INFORMATION_SCHEMA.COLUMNS
                    WHERE
                        TABLE_SCHEMA = '{st.session_state.wh_input_02.upper()}' AND TABLE_NAME = '{st.session_state.table_id.upper()}'
                    ORDER BY
                        ordinal_position;
                """
    
    st.session_state.schema_df = run_query(query)
        
    st.rerun()
    
def get_table_data(arr):
    
    comma_sep_colnames =  ", ".join(arr)
    
    if st.session_state.dw == 'BigQuery':
        query = f"""
                    SELECT
                        {comma_sep_colnames}
                    FROM
                        `{st.session_state.wh_input_01}.{st.session_state.wh_input_02}.{st.session_state.table_id}`
                    LIMIT
                        {st.session_state.no_of_rows};
                """
    else:
        query = f"""
                    SELECT
                        {comma_sep_colnames}
                    FROM
                        {st.session_state.wh_input_01}.{st.session_state.wh_input_02}.{st.session_state.table_id}
                    LIMIT
                        {st.session_state.no_of_rows};
                """
        
    st.session_state.dataframe = run_query(query)
    st.session_state.df = st.session_state.dataframe
    
    st.rerun()

@st.cache_data
def run_query(query):
    try:
        if st.session_state.dw == 'BigQuery':
            query = client.query(query)
            return query.to_dataframe()   
        elif st.session_state.dw == 'Snowflake':
            cxtn = st.connection("snowflake")
            return cxtn.query(query) 
    except Exception as e:
        st.error('Connection error.')
        st.stop()

def main():

    st.set_page_config(layout='wide')

    if 'df' not in st.session_state:
        st.session_state.df = pd.DataFrame()
        
    if 'schema_df' not in st.session_state:
        st.session_state.schema_df = pd.DataFrame()
    
    if 'query_warehouse' not in st.session_state:
        st.session_state.query_warehouse = False

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

            metadata_df = get_column_metadata(
                st.session_state.df, numeric_cols, string_cols)

            with st.container(horizontal=True, horizontal_alignment='right'):
                if st.button('Clear data'):
                    init_state()
                    st.rerun()

            st.write(f'Row Count: {st.session_state.row_count}')
            st.write(f'Column Count: {st.session_state.column_count}')
            st.dataframe(
                metadata_df,
                column_config={
                    col: st.column_config.Column(
                        col,
                        help=f'Statistics for {col}',
                        width='medium'
                    ) for col in metadata_df.columns
                }
            )
        elif st.session_state.query_warehouse:
            if st.session_state.dw == 'BigQuery':
                label1 = 'Enter project id:*'
                label2 = 'Enter dataset id:*'
                label3 = 'Enter table id:*'
            elif st.session_state.dw == 'Snowflake':
                label1 = 'Enter database id:*'
                label2 = 'Enter schema id:*'
                label3 = 'Enter table id:*'
                
            with st.container(border=True):
                if st.session_state.schema_df.empty:
                    wh_input_01 = st.text_input(label1).lower().strip()
                    wh_input_02 = st.text_input(label2).lower().strip()
                    table_id = st.text_input(label3).lower().strip()
                    with st.container(horizontal=True):
                        if st.button('Get schema'):
                            if wh_input_01 and wh_input_02 and table_id:
                                get_table_schema(wh_input_01,  wh_input_02, table_id)
                            else:
                                st.error('Missing input(s)')
                        if st.button('Cancel'):
                            st.session_state.query_warehouse = False 
                            st.rerun() 
                else: 
                    with st.container():
                        checkbox_selections_dict = {}
                        st.write('Select colomn(s):')
                        if st.session_state.dw == 'BigQuery':
                            for idx, row in st.session_state.schema_df.iterrows():
                                st.checkbox(f'{row["column_name"]} {row["data_type"]} | {"Nullable" if row["is_nullable"] else "Not Nullable"}', key=f'{row}_{idx}')
                                checkbox_selections_dict[row["column_name"]] = st.session_state[f'{row}_{idx}']
                        else:
                            for idx, row in st.session_state.schema_df.iterrows():
                                st.checkbox(f'{row["COLUMN_NAME"]} {row["DATA_TYPE"]} | {"Nullable" if row["IS_NULLABLE"] else "Not Nullable"}', key=f'{row}_{idx}')
                                checkbox_selections_dict[row["COLUMN_NAME"]] = st.session_state[f'{row}_{idx}']
                                
                        n = 1000  
                        options = [i * n for i in range(1, n + 1)]
                        st.selectbox(f'Select number of rows', options=options, key='no_of_rows')
                        with st.container(horizontal=True):
                            if st.button('Get data'):
                                columns_array = [i for i, v in checkbox_selections_dict.items() if v]
                                get_table_data(columns_array)
                            if st.button('Cancel'):
                                st.session_state.query_warehouse = False 
                                st.session_state.schema_df = pd.DataFrame()
                                st.rerun() 
                    
        else:
            # upload widget container
            with st.container(border=True):
                try:
                    uploaded_file = st.file_uploader('Upload file:', # upload widget
                                                                        accept_multiple_files=False,
                                                                        on_change=init_state,
                                                                        type=[
                                                                            'csv', 'parquet', 'excel']
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
                                '--','BigQuery', 'Snowflake'], 
                                on_change = set_warehouse, 
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

                if len(st.session_state.charts_array) > 0 and len(st.session_state.charts_array) >= st.session_state.numberof_charts:
                    chart = st.session_state.charts_array[i]
                else:
                    chart = {}

                with st.expander(f'Chart {i + 1}'):
                    # create chart form
                    st.text_input('Enter a descriptive title for chart:*',
                                  key=f'chart title {i}')

                    if chart:
                        st.success(chart[f'chart title {i}'])

                    st.selectbox('Select chart type:*', options=[e.value for e in Charts],
                                 key=f'chart {i}',
                                 on_change=on_chart_selection_change,
                                 args=[f'chart {i}'])

                    if chart and f'chart {i}' in chart:
                        st.success(chart[f'chart {i}'])

                    if st.session_state[f'pie chart {i}']:

                        st.selectbox('Select labels:*', options=st.session_state.df.columns,
                                     key=f'labels {i}',
                                     on_change=on_chart_selection_change,
                                     args=[f'labels {i}'])

                        if chart and f'labels {i}' in chart:
                            st.success(chart[f'labels {i}'])

                        st.selectbox('Select values:*', options=st.session_state.df.columns,
                                     key=f'values {i}',
                                     on_change=on_chart_selection_change,
                                     args=[f'values {i}'])

                        if chart and f'values {i}' in chart:
                            st.success(chart[f'values {i}'])

                    elif st.session_state[f'histogram chart {i}']:

                        st.selectbox('Select x-axis:*', options=st.session_state.df.columns,
                                     key=f'x-axis {i}',
                                     on_change=on_chart_selection_change,
                                     args=[f'x-axis {i}'])

                        if chart and f'x-axis {i}' in chart:
                            st.success(chart[f'x-axis {i}'])

                        st.multiselect('Select y-axis:*', options=st.session_state.df.columns,
                                       key=f'y-axis {i}',
                                       on_change=on_chart_selection_change,
                                       args=[f'y-axis {i}'])

                        if chart and f'x-axis {i}' in chart:
                            st.success(chart[f'y-axis {i}'])

                        st.number_input('Enter bin size:*', key=f'bins {i}')

                        if chart and f'bins {i}' in chart:
                            st.success(chart[f'bins {i}'])
                    else:

                        st.selectbox('Select x-axis:*', options=st.session_state.df.columns,
                                     key=f'x-axis {i}',
                                     on_change=on_chart_selection_change,
                                     args=[f'x-axis {i}'])

                        if chart and f'x-axis {i}' in chart:
                            st.success(chart[f'x-axis {i}'])

                        st.multiselect('Select y-axis:*', options=st.session_state.df.columns,
                                       key=f'y-axis {i}',
                                       on_change=on_chart_selection_change,
                                       args=[f'y-axis {i}'])

                        if chart and f'y-axis {i}' in chart:
                            st.success(chart[f'y-axis {i}'])

                        st.multiselect('Select color(s):*', options=[e.value for e in Colors],
                                       key=f'color {i}',
                                       on_change=on_chart_selection_change,
                                       args=[f'color {i}'])

                        if chart and f'color {i}' in chart:
                            st.success(chart[f'color {i}'])

                    # container for form buttons
                    with st.container(horizontal=True, horizontal_alignment='right', vertical_alignment='center'):

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

                        elif len(st.session_state.charts_array) < st.session_state.numberof_charts\
                                and st.session_state.numberof_charts == i + 1:

                            if st.button('Cancel', key=f'cancel_button_{i}'):
                                st.session_state.numberof_charts -= 1
                                st.rerun()

                            validate = validate_input(i)
                            if validate:
                                if st.button('Render Chart', key=f'render_button_{i}'):
                                    render_chart(i)
                        else:
                            validate = validate_input(i)
                            if validate:
                                if st.button('Update', key=f'update_button_{i}'):
                                    render_chart(i, True)


if __name__ == '__main__':
    main()
