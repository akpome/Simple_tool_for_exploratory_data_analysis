import streamlit as st
import plotly.express as px
import sys
from pathlib import Path

dir = Path(__file__).resolve().parent 
sys.path.insert(0, str(dir))

from statics import Charts as Charts
from statics import chart_funcs_dict as chart_funcs_dict
from statics import colors_dict as colors_dict

def chart_row(n, second_row=False):
    if not second_row and n > 2:
        n = 2
    if second_row:
        n -= 2
    for i, col in enumerate(st.columns(n)):
        if second_row:
            i += 2
        with col:
            chart = st.session_state.charts_array[i]
            with st.container():
                title = chart[f'chart title {i}']
                st.markdown(f"<div style='text-align: center; color: grey;'>{title}</div>", unsafe_allow_html=True)
            match chart[f'chart {i}']:
                case Charts.LCH.value | Charts.ACH.value | Charts.BCH.value | Charts.SCH.value:
                    chart_funcs_dict[chart[f'chart {i}']](chart[f'chart dataframe {i}'], 
                                            x = chart[f'x-axis {i}'], 
                                            y = chart[f'y-axis {i}'], 
                                            color = [colors_dict[i] for i in chart[f'color {i}']])
                case Charts.PCH.value:
                    chart_funcs_dict[chart[f'chart {i}']](px.pie(chart[f'chart dataframe {i}'], 
                                           values=chart[f'values {i}'], 
                                           names=chart[f'labels {i}']).update_layout(width=500, height=350))
                case Charts.DCH.value:
                    chart_funcs_dict[chart[f'chart {i}']](px.pie(chart[f'chart dataframe {i}'], 
                                           values=chart[f'values {i}'], 
                                           names=chart[f'labels {i}'], hole=0.4).update_layout(width=500, height=350))                    
                case Charts.HCH.value:
                    if chart[f'bins {i}']:
                        bin_size = int(chart[f'bins {i}'])
                    else:
                        bin_size = 0
                    chart_funcs_dict[chart[f'chart {i}']](px.histogram(chart[f'chart dataframe {i}'], 
                                           x=chart[f'x-axis {i}'], 
                                           y = chart[f'y-axis {i}'],
                                           nbins=bin_size).update_layout(bargap=0.01,width=500, height=350))
                    
def main():
    
    if 'charts_array' in st.session_state and st.session_state.charts_array:
        with st.container():
            n = len(st.session_state.charts_array)
            try:
                if n > 0 and n <=2:
                    chart_row(n)
                if n > 2:
                    chart_row(n)
                    chart_row(n, True)

            except Exception as e:
                    st.error('Error rendering chart. Please review your settings and try again')

if __name__ == "__main__":
    main()