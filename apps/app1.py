# Libraries
import streamlit as st
import pandas as pd
import lazypredict
from lazypredict.Supervised import LazyClassifier
from lazypredict.Supervised import LazyRegressor
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_squared_error, r2_score
import matplotlib.pyplot as plt
import seaborn as sns
import base64
import io
import numpy as np
from streamlit_option_menu import option_menu
import streamlit.components.v1 as components
def app():
    # Page expands to full width
    # st.set_page_config(page_title='Auto Ml Regression and Classification App',layout='wide')
        def build_model(df):
            X = df.iloc[:,:-1] # Features variables
            Y = df.iloc[:,-1]  # Target variable

            # Dimensions of dataset
            st.markdown('**1.2. Dataset dimension**')
            st.write('X')
            st.info(X.shape)
            st.write('Y')
            st.info(Y.shape)

            # Variable details
            st.markdown('**1.3. Variable details**:')
            st.write('X variable (first 20 are shown)')
            st.info(list(X.columns[:20]))
            st.write('Y variable')
            st.info(Y.name)

            # Building lazy model
            X_train, X_test, Y_train, Y_test = train_test_split(X, Y,test_size = split_size,random_state = seed_number)
            # reg = LazyRegressor(verbose=0,ignore_warnings=False, custom_metric=None)
            clf = LazyClassifier( verbose=0, ignore_warnings=True, custom_metric=None)
            models,predictions = clf.fit(X_train, X_test, Y_train, Y_test)
            # verbose is the choice that how you want to see the output of your algorithm while it's training
            models_train,predictions_train = clf.fit(X_train, X_train, Y_train, Y_train)
            models_test,predictions_test = clf.fit(X_train, X_test, Y_train, Y_test)
            
            st.subheader('2. Table of Model Performance')

            st.write('Training set')
            st.write(predictions_train)
            st.markdown(filedownload(predictions_train,'training.csv'), unsafe_allow_html=True)

            st.write('Test set')
            st.write(predictions_test)
            st.markdown(filedownload(predictions_test,'test.csv'), unsafe_allow_html=True)

            st.subheader('3. Plot of Model Performance (Test set)')

            #Accuracy
            with st.markdown('**Accuracy**'):
                # Tall
                predictions_test["Accuracy"] = [0 if i < 0 else i for i in predictions_test["Accuracy"] ]
                # if value(i) is less than 0 i.e R-squared predicted value of test set is 0 , it will return 0
                # else it will return the R-squared value predicted
                plt.figure(figsize=(3, 9))
                sns.set_theme(style="whitegrid")
                ax1 = sns.barplot(y=predictions_test.index, x="Accuracy", data=predictions_test)
                ax1.set(xlim=(0, 1))
                # xlim is basically the x axis value sepration range
            st.markdown(imagedownload(plt,'plot-r2-tall.pdf'), unsafe_allow_html=True)
                # Wide
            plt.figure(figsize=(9, 3))
            sns.set_theme(style="whitegrid")
            ax1 = sns.barplot(x=predictions_test.index, y="Accuracy", data=predictions_test)
            ax1.set(ylim=(0, 1))
            # y axis plot download
            plt.xticks(rotation=90)
            st.pyplot(plt)
            st.markdown(imagedownload(plt,'plot-r2-wide.pdf'), unsafe_allow_html=True)

            # Calculation plot
            with st.markdown('**Calculation time**'):
                # Tall
                predictions_test["Time Taken"] = [0 if i < 0 else i for i in predictions_test["Time Taken"] ]
                plt.figure(figsize=(3, 9))
                sns.set_theme(style="whitegrid")
                ax3 = sns.barplot(y=predictions_test.index, x="Time Taken", data=predictions_test)
            st.markdown(imagedownload(plt,'plot-calculation-time-tall.pdf'), unsafe_allow_html=True)
                # Wide
            plt.figure(figsize=(9, 3))
            sns.set_theme(style="whitegrid")
            ax3 = sns.barplot(x=predictions_test.index, y="Time Taken", data=predictions_test)
            plt.xticks(rotation=90)
            st.pyplot(plt)
            st.markdown(imagedownload(plt,'plot-calculation-time-wide.pdf'), unsafe_allow_html=True)

        def filedownload(df, filename):
            csv = df.to_csv(index=False)
            b64 = base64.b64encode(csv.encode()).decode()  # strings <-> bytes conversions
            href = f'<a href="data:file/csv;base64,{b64}" download={filename}>Download {filename} File</a>'
            return href

        def imagedownload(plt, filename):
            s = io.BytesIO()
            plt.savefig(s, format='pdf', bbox_inches='tight')
            plt.close()
            b64 = base64.b64encode(s.getvalue()).decode()  # strings <-> bytes conversions
            href = f'<a href="data:image/png;base64,{b64}" download={filename}>Download {filename} File</a>'
            return href

        st.write("""
        # Auto Ml Classification App 
        """)

        # Sidebar - Collects user input features into dataframe
        with st.sidebar.header('1. Upload your CSV data'):
            uploaded_file = st.sidebar.file_uploader(" do Upload your input CSV file", type=["csv"])
            st.sidebar.markdown("""
        [Example CSV input file](https://raw.githubusercontent.com/dataprofessor/data/master/delaney_solubility_with_descriptors.csv)
    """)

        # Sidebar - Specify parameter settings
        with st.sidebar.header('2. Set Parameters'):
            split_size = st.sidebar.slider('Data split ratio is(% for Training Set)', 10, 90, 80, 5)
            seed_number = st.sidebar.slider('Set the random seed number', 1, 100, 43, 1)


        # Main Panel
    # Displays the dataset
        st.subheader('1. Dataset')

        if uploaded_file is not None:
            df = pd.read_csv(uploaded_file)
            st.markdown('**1.1. Glimpse of dataset**')
            st.write(df)
            build_model(df)
        else:
            st.info('Awaiting for CSV file to be uploaded.')
     
    # TAB 1 end 









    # # TAB 2 start
    # with tab2:
    #     def build_model(df):
    #         X = df.iloc[:,:-1] # Features variables
    #         Y = df.iloc[:,-1]  # Target variable

    #         # Dimensions of dataset
    #         st.markdown('**1.2. Dataset dimension**')
    #         st.write('X')
    #         st.info(X.shape)
    #         st.write('Y')
    #         st.info(Y.shape)

    #         # Variable details
    #         st.markdown('**1.3. Variable details**:')
    #         st.write('X variable (first 20 are shown)')
    #         st.info(list(X.columns[:20]))
    #         st.write('Y variable')
    #         st.info(Y.name)

    #         # Building lazy model
    #         X_train, X_test, Y_train, Y_test = train_test_split(X, Y,test_size = split_size,random_state = seed_number)
    #         # reg = LazyRegressor(verbose=0,ignore_warnings=False, custom_metric=None)
    #         clf = LazyClassifier( verbose=0, ignore_warnings=True, custom_metric=None)
    #         models,predictions = clf.fit(X_train, X_test, Y_train, Y_test)
    #         # verbose is the choice that how you want to see the output of your algorithm while it's training
    #         models_train,predictions_train = clf.fit(X_train, X_train, Y_train, Y_train)
    #         models_test,predictions_test = clf.fit(X_train, X_test, Y_train, Y_test)
            
    #         st.subheader('2. Table of Model Performance')

    #         st.write('Training set')
    #         st.write(predictions_train)
    #         st.markdown(filedownload(predictions_train,'training.csv'), unsafe_allow_html=True)

    #         st.write('Test set')
    #         st.write(predictions_test)
    #         st.markdown(filedownload(predictions_test,'test.csv'), unsafe_allow_html=True)

    #         st.subheader('3. Plot of Model Performance (Test set)')

    #         #Accuracy
    #         with st.markdown('**Accuracy**'):
    #             # Tall
    #             predictions_test["Accuracy"] = [0 if i < 0 else i for i in predictions_test["Accuracy"] ]
    #             # if value(i) is less than 0 i.e R-squared predicted value of test set is 0 , it will return 0
    #             # else it will return the R-squared value predicted
    #             plt.figure(figsize=(3, 9))
    #             sns.set_theme(style="whitegrid")
    #             ax1 = sns.barplot(y=predictions_test.index, x="Accuracy", data=predictions_test)
    #             ax1.set(xlim=(0, 1))
    #             # xlim is basically the x axis value sepration range
    #         st.markdown(imagedownload(plt,'plot-r2-tall.pdf'), unsafe_allow_html=True)
    #             # Wide
    #         plt.figure(figsize=(9, 3))
    #         sns.set_theme(style="whitegrid")
    #         ax1 = sns.barplot(x=predictions_test.index, y="Accuracy", data=predictions_test)
    #         ax1.set(ylim=(0, 1))
    #         # y axis plot download
    #         plt.xticks(rotation=90)
    #         st.pyplot(plt)
    #         st.markdown(imagedownload(plt,'plot-r2-wide.pdf'), unsafe_allow_html=True)

    #         # Calculation plot
    #         with st.markdown('**Calculation time**'):
    #             # Tall
    #             predictions_test["Time Taken"] = [0 if i < 0 else i for i in predictions_test["Time Taken"] ]
    #             plt.figure(figsize=(3, 9))
    #             sns.set_theme(style="whitegrid")
    #             ax3 = sns.barplot(y=predictions_test.index, x="Time Taken", data=predictions_test)
    #         st.markdown(imagedownload(plt,'plot-calculation-time-tall.pdf'), unsafe_allow_html=True)
    #             # Wide
    #         plt.figure(figsize=(9, 3))
    #         sns.set_theme(style="whitegrid")
    #         ax3 = sns.barplot(x=predictions_test.index, y="Time Taken", data=predictions_test)
    #         plt.xticks(rotation=90)
    #         st.pyplot(plt)
    #         st.markdown(imagedownload(plt,'plot-calculation-time-wide.pdf'), unsafe_allow_html=True)

    #     def filedownload(df, filename):
    #         csv = df.to_csv(index=False)
    #         b64 = base64.b64encode(csv.encode()).decode()  # strings <-> bytes conversions
    #         href = f'<a href="data:file/csv;base64,{b64}" download={filename}>Download {filename} File</a>'
    #         return href

    #     def imagedownload(plt, filename):
    #         s = io.BytesIO()
    #         plt.savefig(s, format='pdf', bbox_inches='tight')
    #         plt.close()
    #         b64 = base64.b64encode(s.getvalue()).decode()  # strings <-> bytes conversions
    #         href = f'<a href="data:image/png;base64,{b64}" download={filename}>Download {filename} File</a>'
    #         return href

    #     st.write("""
    #     # Auto Ml Regression and Classification App 
    #     """)

    #     # Sidebar - Collects user input features into dataframe
    #     with st.sidebar.header('1. Upload your CSV data'):
    #         uploaded_file = st.sidebar.file_uploader("Upload your input CSV file", type=["csv"])
    #         st.sidebar.markdown("""
    #     [Example CSV input file](https://raw.githubusercontent.com/dataprofessor/data/master/delaney_solubility_with_descriptors.csv)
    # """)

    #     # Sidebar - Specify parameter settings
    #     with st.sidebar.header('2. Set Parameters'):
    #         split_size = st.sidebar.slider('Data split ratio (% for Training Set)', 10, 90, 80, 5)
    #         seed_number = st.sidebar.slider('Set the random seed number', 1, 100, 42, 1)


    #     # Main Panel
    # # Displays the dataset
    #     st.subheader('1. Dataset')

    #     if uploaded_file is not None:
    #         df = pd.read_csv(uploaded_file)
    #         st.markdown('**1.1. Glimpse of dataset**')
    #         st.write(df)
    #         build_model(df)
    #     else:
    #         st.info('Awaiting for CSV file to be uploaded.')