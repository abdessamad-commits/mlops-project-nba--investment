import streamlit as st
import plotly.express as px
st.set_option('deprecation.showPyplotGlobalUse', False)



def app(df, features, target):

    st.title('NBA Players Career Length APP')
    #st.write('Univariate Data Exploration')



    agree1 = st.checkbox("Plot the dataset", value=True)
    if agree1:
        st.dataframe(data=df, width=None, height=None)
        st.write(df.shape)

    agree2 = st.checkbox("Plot summary statistics", value=False)
    if agree2:
        st.dataframe(features.describe())    

    agree3 = st.checkbox("Plot class distribution", value=False)
    if agree3:
        dist = px.histogram(df, x="Outcome Career Length", color="Outcome Career Length")
        st.plotly_chart(dist)
              

    option = st.selectbox(
     'Please select one Numerical variable to vizualize',
     list(features))

    choice = st.selectbox(
     'Please select the chosen plot',
     ("Box Plot","Histogram"))
    
   # col1, col2 = st.columns(2)

    if choice == "Histogram":
        with st.container():    
        #st.write('First column')
            fig1 = px.histogram(df, x=option, title='Distribution of '+option, width=800)
            st.plotly_chart(fig1)
    elif choice == "Box Plot":
        with st.container():    
        #st.write('First column')
            fig1 = px.box(df, y=option, title='Distribution of '+option, width=1000)
            st.plotly_chart(fig1)
        



    if choice == "Histogram":
        with st.container():    
        #st.write('First column')
            fig2 = px.histogram(df, x=option, color="Outcome Career Length", width=800, title='Distribution of '+option+' grouped by the career length larger then five years')
            st.plotly_chart(fig2)
    elif choice == "Box Plot":
        with st.container():    
        #st.write('First column')
            fig2 = px.box(df, y=option, color="Outcome Career Length", width=1000, title='Distribution of '+option+' grouped by the career length being larger than five years')
            st.plotly_chart(fig2)
        
    