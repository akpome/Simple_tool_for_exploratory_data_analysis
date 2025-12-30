from statics import colors_dict as colors_dict
from statics import Charts as Charts
import plotly.express as px
from pathlib import Path
import streamlit as st
from math import ceil
import sys

# required to import statics from parent directory
dir = Path(__file__).resolve().parent
sys.path.insert(0, str(dir))


def render_chart(fig):
    fig.update_layout(yaxis_title=None)
    fig.update_layout(legend_title_text="")
    st.plotly_chart(fig)


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
                    f'<div style="font-weight: bold;">{title}</div>', unsafe_allow_html=True)
            match chart[f'chart {i}']:
                case Charts.ACH.value: # area chart
                    fig = px.area(chart[f'chart dataframe {i}'], x=chart[f'x-axis {i}'], y=chart[f'y-axis {i}'], 
                                  color_discrete_sequence=[colors_dict[i] for i in chart[f'color {i}']])
                    render_chart(fig)
                case Charts.SCH.value: # scatter chart
                    fig = px.scatter(chart[f'chart dataframe {i}'], x=chart[f'x-axis {i}'], y=chart[f'y-axis {i}'], 
                                     color_discrete_sequence=[colors_dict[i] for i in chart[f'color {i}']])
                    render_chart(fig)
                case Charts.LCH.value: # line chart
                    fig = px.line(chart[f'chart dataframe {i}'], x=chart[f'x-axis {i}'], y=chart[f'y-axis {i}'], 
                                  color_discrete_sequence=[colors_dict[i] for i in chart[f'color {i}']])
                    render_chart(fig)
                case Charts.BUC.value: # bubble chart
                    fig = px.scatter(chart[f'chart dataframe {i}'], x=chart[f'x-axis {i}'], y=chart[f'y-axis {i}'], size=chart[f'size {i}'], 
                                     size_max=40, color_discrete_sequence=[colors_dict[i] for i in chart[f'color {i}']])
                    render_chart(fig)
                case Charts.BCH.value: # bar chart
                    fig = px.bar(chart[f'chart dataframe {i}'], x=chart[f'x-axis {i}'], y=chart[f'y-axis {i}'], barmode='group', 
                                 color_discrete_sequence=[colors_dict[i] for i in chart[f'color {i}']])
                    render_chart(fig)
                case Charts.HCH.value:  # histogram chart
                    bin_size = int(chart[f'bins {i}'])
                    fig = px.histogram(chart[f'chart dataframe {i}'], x=chart[f'x-axis {i}'], y=chart[f'y-axis {i}'], nbins=bin_size, 
                                       color_discrete_sequence=[colors_dict[i] for i in chart[f'color {i}']])
                    render_chart(fig)
                case Charts.PCH.value:  # pie chart
                    fig = px.pie(chart[f'chart dataframe {i}'], values=chart[f'values {i}'],names=chart[f'labels {i}'])
                    st.plotly_chart(fig)
                case Charts.DCH.value:  # donut chart
                    fig = px.pie(chart[f'chart dataframe {i}'], values=chart[f'values {i}'], names=chart[f'labels {i}'], hole=0.4)
                    st.plotly_chart(fig)


def main():

    if 'charts_array' in st.session_state and st.session_state.charts_array:
        with st.container():
            st.markdown(
                f'<div style="text-align: center; font-size: 24px; font-weight: bold; margin-bottom: 20px;">{st.session_state.dashboard_title}</div>', unsafe_allow_html=True)
        with st.container():
            num_of_charts = len(st.session_state.charts_array)
            num_of_charts_per_row = 2            
            x = ceil(num_of_charts/num_of_charts_per_row)
            for i in range(x):
                try:
                    chart_row(num_of_charts, i)
                except Exception as e:
                    st.error("Error rendering chart. Please go back to the Home page and re-generate the chart")

if __name__ == '__main__':
    main()
