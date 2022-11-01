import streamlit as st
from multiapp import MultiApp
from apps import app1,app2,app3 # import your app modules here
app = MultiApp()

st.markdown("""
# Auto ML dataset explorer
""")

# Add all your application here
app.add_app("Home", app3.app)
app.add_app("AutoML of Regression models", app2.app)
app.add_app("AutoML of Classification models", app1.app)
# The main app
app.run()
