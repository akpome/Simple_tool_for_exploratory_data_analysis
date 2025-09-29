import streamlit as st

st.set_page_config(
    page_title='Simple tool for exploratory data analysis',
    page_icon=':material/bar_chart:',
)

pages = [
    st.Page('pages/home.py', title='Home', icon=':material/home:'),
    st.Page('pages/dashboard.py', title='Dashboard', icon=':material/bar_chart:')
]

pg = st.navigation(pages)

pg.run()