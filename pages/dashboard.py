from statics import colors_dict as colors_dict
from statics import chart_funcs_dict as chart_funcs_dict
from statics import Charts as Charts
import streamlit as st
import plotly.express as px
from math import ceil
import sys
from pathlib import Path

# required to import statics from parent directory
dir = Path(__file__).resolve().parent
sys.path.insert(0, str(dir))


def chart_row(n, j):
    # anchor first row of charts to a max of 2
    if j == 0 and n > 2:
        n = 2
    # second row of charts
    if j > 0:
        n -= 2
    for i, col in enumerate(st.columns(n)):
        if j > 0:
            i += 2
        with col:
            chart = st.session_state.charts_array[i]
            with st.container():
                title = chart[f'chart title {i}']
                st.markdown(
                    f'<div style="text-align: center; color: grey;">{title}</div>', unsafe_allow_html=True)
            chart_func = chart_funcs_dict[chart[f'chart {i}']]
            match chart[f'chart {i}']:  # line, area, bar and scatter charts
                case Charts.LCH.value | Charts.ACH.value | Charts.BCH.value | Charts.SCH.value:
                    chart_func(chart[f'chart dataframe {i}'],
                               x=chart[f'x-axis {i}'], y=chart[f'y-axis {i}'],
                               color=[colors_dict[i] for i in chart[f'color {i}']])
                case Charts.PCH.value:  # pie chart
                    chart_func(px.pie(chart[f'chart dataframe {i}'], values=chart[f'values {i}'],
                                      names=chart[f'labels {i}']).update_layout(width=500, height=350),
                               key=f'pie {i}')
                case Charts.DCH.value:  # donut chart
                    chart_func(px.pie(chart[f'chart dataframe {i}'], values=chart[f'values {i}'],
                                      names=chart[f'labels {i}'], hole=0.4).update_layout(width=500, height=350),
                               key=f'donut {i}')
                case Charts.HCH.value:  # histogram chart
                    if chart[f'bins {i}']:
                        bin_size = int(chart[f'bins {i}'])
                    else:
                        bin_size = 0
                    chart_func(px.histogram(chart[f'chart dataframe {i}'], x=chart[f'x-axis {i}'],
                                            y=chart[f'y-axis {i}'], nbins=bin_size).update_layout(bargap=0.01,
                                                                                                  width=500, height=350),
                               key=f'histogram {i}')


def main():

    if 'charts_array' in st.session_state and st.session_state.charts_array:
        with st.container():
            num_of_charts = len(st.session_state.charts_array)
            num_of_charts_per_row = 2
            try:
                x = ceil(num_of_charts/num_of_charts_per_row)
                for i in range(x):
                    chart_row(num_of_charts, i)

            except Exception as e:
                st.error(e)


if __name__ == '__main__':
    main()
