import streamlit as st
import plotly.express as px


def app(df, features, target):
    st.title('NBA Players Career Length APP')
    #st.write('Multivariate Data Exploration')


    
    option_1 = st.selectbox(
        'Please select the variable to vizualize on the x-axis',
        list(features)
    )

    option_2 = st.selectbox(
        'Please select the variable to vizualize on the y-axis',
        list(features),
        index=1
    )

    option_3 = st.selectbox(
        'Please select the variable representing the data points size',
        list(features)
    )

    choice3 = st.selectbox('Please select the marginal distribution plot', ("box", "violin", "histogram"))
    choice2 = st.selectbox('Labeled Data', ("Yes", "No"))

    df["label"] = df["Outcome Career Length"].astype(str)

    if choice2 == "Yes":
        with st.container():    
            fig3 = px.scatter(df, x=option_1, y=option_2, color="label", marginal_x=choice3, marginal_y=choice3, width=1200, height=900,
                size=option_3, title="Scatter Plot", trendline="ols")
            st.plotly_chart(fig3)
    else:  
        with st.container():    
            fig3 = px.scatter(df, x=option_1, y=option_2, marginal_x=choice3, marginal_y=choice3, width=1200, height=900,
                size=option_3, title="Log-transformed fit on linear axes", trendline="ols")
            st.plotly_chart(fig3)
    
    agree = st.checkbox("Plot Correlation matrix")
    if agree:
        with st.container():    
                fig4 = px.imshow(features.corr(),width=1200,height=900,text_auto=True)
                st.plotly_chart(fig4)