import streamlit as st
import joblib
import pandas as pd

# Load the model and scaler
Model = joblib.load('models/model.h5')
scaler = joblib.load('models/scaler.h5')

Inputs = ['Depth', 'WOB', 'RPM', 'Q', 'T', 'SPP', 'ROP', 'M. Wt', 'M. T. in', 'M. T. out']


def predict(Depth, WOB, RPM, Q, T, SPP, ROP, M_Wt, M_T_in, M_T_out):
    test_df = pd.DataFrame(columns=Inputs, index=[0])
    test_df.at[0, "Depth"] = Depth
    test_df.at[0, "WOB"] = WOB
    test_df.at[0, "RPM"] = RPM
    test_df.at[0, "Q"] = Q
    test_df.at[0, "T"] = T
    test_df.at[0, "SPP"] = SPP
    test_df.at[0, "ROP"] = ROP
    test_df.at[0, "M. Wt"] = M_Wt
    test_df.at[0, "M. T. in"] = M_T_in
    test_df.at[0, "M. T. out"] = M_T_out

    result = Model.predict(scaler.transform(test_df))[0]
    return result


def header(url):
    st.markdown(f'<p style="background-color:#0066cc;color:#33ff33;font-size:24px;border-radius:2%;">{url}</p>',
                unsafe_allow_html=True)


def main():
    page_element="""
        <style>
        [data-testid="stAppViewContainer"]{
          background-image: url("https://www.fieldequip.com/wp-content/uploads/2019/02/AdobeStock_139370673.jpeg");
          background-size: cover;
        }
        </style>
        """
    st.markdown(page_element, unsafe_allow_html=True) 
        
    # Front-end elements of the web page 
    st.markdown("""
    <style>
        body {
            background-image: url('https://www.fieldequip.com/wp-content/uploads/2019/02/AdobeStock_139370673.jpeg');
            background-size: cover;
        }
        .sidebar .sidebar-content {
        background-color: #0066cc; /* Change sidebar background color to blue */
        }
        .header {
            background-color:yellow;
            padding: 20px;
            color: black;
            text-align: center;
            border-top-left-radius: 10px;
            border-top-right-radius: 10px;
        }
        .container {
            padding: 20px;
            background-color: rgba(255, 255, 255, 0.8); /* Adjust the opacity as needed */
            border-radius: 10px;
        }
        .card {
            background-color: rgba(51, 170, 51, 0.1); /* Adjust the color and opacity as needed */
            padding: 20px;
            border-radius: 10px;
            color: black; /* Adjust the text color as needed */
        }
    </style>
    <div class="header">
        <h1>Vibrations Detection System ML Prediction App</h1>
    </div>
    """, unsafe_allow_html=True)

    st.sidebar.title("Choose your Features")
    # Add descriptions for each feature
    Depth = st.sidebar.slider("Depth (ft)", min_value=0, max_value=100000, value=0, step=1)
    WOB = st.sidebar.slider("WOB (lbs)", min_value=0, max_value=100000, value=0, step=1)
    RPM = st.sidebar.slider("RPM (rev/min)", min_value=0, max_value=100000, value=0, step=1)
    Q = st.sidebar.slider("Q (gpm)", min_value=0, max_value=100000, value=0, step=1)
    T = st.sidebar.slider("T (°F)", min_value=0, max_value=100000, value=0, step=1)
    SPP = st.sidebar.slider("SPP (psi)", min_value=0, max_value=100000, value=0, step=1)
    ROP = st.sidebar.slider("ROP (ft/hr)", min_value=0, max_value=100000, value=0, step=1)
    M_Wt = st.sidebar.slider("M. Wt (ppg)", min_value=0, max_value=100000, value=0, step=1)
    M_T_in = st.sidebar.slider("M. T. in (°F)", min_value=0, max_value=100000, value=0, step=1)
    M_T_out = st.sidebar.slider("M. T. out (°F)", min_value=0, max_value=100000, value=0, step=1)
    result = ""

    if st.sidebar.button("Predict"):
        result = predict(Depth, WOB, RPM, Q, T, SPP, ROP, M_Wt, M_T_in, M_T_out)

        #st.markdown(f'<h1 style="color:#33ff33;font-size:40px;text-align:center;border-style: solid;border-width:5px;border-color:#fbff00;">{result}</h1>',unsafe_allow_html=True)
        
        target_columns=['Vib. Count', 'Lateral Vib.', 'Axial Vib.', 'SS%']
        for i, value in enumerate(result):
           # Add beauty features for the result presentation
            st.markdown(f'<h1 style="color:blue;font-size:40px;text-align:left">{target_columns[i]}: {value:.2f}</h1>', unsafe_allow_html=True)
    #st.image('R.jfif')

if __name__ == '__main__':
    main()
