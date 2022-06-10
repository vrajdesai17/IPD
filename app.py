import streamlit as st
from multiapp import MultiApp
from apps import app1,app2,app3 # import your app modules here
app = MultiApp()

st.markdown("""
# ML + SQL PLAYGROUND + DATASET EXPLORER

LETS explore.

""")

# Add all your application here
app.add_app("Home", app3.app)
app.add_app("Data", app2.app)
app.add_app("Model", app1.app)
# The main app
app.run()
