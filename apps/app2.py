import os
import streamlit as st
import pandas as pd

import sqlite3 

def app():
    conn = sqlite3.connect('./apps/data/world.sqlite')
    C = conn.cursor()
    def sql_executor(raw_code):
        C.execute(raw_code)
        data = C.fetchall()
        return data

    city = ['ID','Name','CountryCode','District','Population']
    country = ['Code','Name','Continent','Region','SurfaceArea','IndepYear','Population',"LifeExpectancy","GNP","GNPOld","LocalName","GovernmentForm","HeadOfState","Capital","Code2"]
    countrylanguage = ["CountryCode","Language","IsOfficial","Percentage"]
    
    st.title("SQLPlayground")

    # menu = ["Home","About"]
    # choice  = st.sidebar.selectbox("Menu",menu)

    # if choice == "Home":
    st.subheader("HomePage")
    col1,col2 = st.columns(2) 
    with col1:
        with st.form(key="query_form"): # with statement in Python is used in exception handling to make the code cleaner and much more readable. 
            raw_code = st.text_area("SQL code here")
            submit_code = st.form_submit_button("Execute")

        with st.expander("Table Info"):
            t_info = {'city':city,'country':country,'countrylanguage':countrylanguage}
            st.json(t_info) # Display object or string as a pretty-printed JSON string.

# Result layout
    with col2:
        if submit_code:
            st.info("Query Submitted")
            st.code(raw_code) # .code(): represents your query with a cop to clipboard prop.
             
            # Result
            query_results = sql_executor(raw_code)
            with st.expander("Results"):
                st.write(query_results)

            with st.expander("Pretty Table"):
                query_df = pd.DataFrame(query_results)
                st.dataframe(query_df)

    # else:
    #     st.subheader("About")

    

# if __name__ == '__main__':
#     main()