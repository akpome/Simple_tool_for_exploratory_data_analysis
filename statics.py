import streamlit as st
from enum import Enum

class Options:
    stm = 'Select Transform'
    rer = 'Remove Empty Rows'
    rnr = 'Remove Non Numeric or Null Rows'
    cci = 'Convert Column To Interger'
    ccd = 'Convert Column To Decimal'
    ccs = 'Convert Column To String'
    cdc = 'Create Datetime Column'
    cts = 'Create Datetime Column From Timestamp'
    cyc = 'Create Year Column'
    cqc = 'Create Quarter Column'
    cmc = 'Create Month Column'
    cdy = 'Create Day Column'
    ctc = 'Create Time Column'
    rmc = 'Remove Column'
    spc = 'Split Column'
    rnc = 'Rename Column'
    sfh = 'Set First Row as Header'
    cat = 'Clear All Transforms'
    rnz = 'Replace Non Numeric or Null Rows With Zero'
    rnm = 'Replace Non Numeric or Null Rows with Mean'
    fnd = 'Fill Non Numeric of Null Rows Down'
    fnu = 'Fill Non Numeric or Null Rows Up'
    fed = 'Fill Empty Rows Down'
    feu = 'Fill Empty Rows Up'
    trt = 'Transpose Table'
    gbc = 'Group By Column'
    fbc = 'Filter Table'
    pvt = 'Pivot Table'
    asc = 'Sort Ascending'
    dsc = 'Sort Descending'

class Charts(Enum):
    LCH = 'Line chart'
    BCH = 'Bar chart'
    SCH = 'Scatter chart'
    ACH = 'Area chart'
    PCH = 'Pie chart'
    DCH = 'Donut chart'
    HCH = 'Histogram chart'

class Agg_Funcs(Enum):
    COUNT= 'Count'
    MEDIAN = 'Median'
    MEAN = 'Mean'
    MAX = 'Max'
    MIN = 'Min'
    SUM = 'Sum'

class Colors(Enum):
    RED = 'Red'
    BLUE = 'Blue'
    GREEN = 'Green'
    YELLOW = 'Yellow'
    ORANGE = 'Orange'
    PURPLE = 'Purple'

chart_funcs_dict = {
    Charts.LCH.value: st.line_chart,
    Charts.BCH.value: st.bar_chart,
    Charts.SCH.value: st.scatter_chart,
    Charts.ACH.value: st.area_chart,
    Charts.PCH.value: st.plotly_chart,
    Charts.DCH.value: st.plotly_chart,
    Charts.HCH.value: st.plotly_chart
}

colors_dict = {
    Colors.BLUE.value: "#0000ff",
    Colors.RED.value: '#ff0000',
    Colors.GREEN.value: '#00ff00',
    Colors.YELLOW.value: '#ffff00',
    Colors.ORANGE.value: '#ffaa00',
    Colors.PURPLE.value: '#ff00ff'
}