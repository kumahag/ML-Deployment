# ML-Deployment
ML to App Project
This project deploys a regression ML on Corporation Favorita Sales into a Web App using streamlit.

SetUp
import streamlit as st
import pandas as pd
import os, pickle

# display the front end aspect
st.set_page_config(layout="centered")

# loading the dataframe
@st.cache
def load_data(relative_path):
    train_data = pd.read_csv(relative_path, index_col=0)
    train_data["year"] = pd.to_datetime(train_data["date"]).dt.year
    train_data["date"] = pd.to_datetime(train_data["date"]).dt.date
    train_data.rename(columns={"type_x": "store_type"}, inplace=True)
    train_data.rename(columns={"type_y": "holiday_type"}, inplace=True)
    return train_data

def change():
  print(st.session_state.train_data)
state =st.checkbox("checkbox", value=True,on_change= change, key ="train_data")
if state:
 st.write("Preview Data")
else:
   pass
# dataframe base
relative_path = r"streamlit_project\train_data.csv"
train_data = load_data(relative_path)
st.write(train_data.head())
st.markdown("------")
st.markdown("------")

# interface
st.title("Favorita Sales Prediction App")
st.text("This App  shows sales prediction for Favorita Stores based on User Inputs")

#Getting date features
def getDateFeatures(input_data, date):
        input_data['date']= pd.to_datetime(input_data['date'])
        input_data['month'] =input_data['date'].dt.month
        input_data['day_of_month'] =input_data['date'].dt.day
        input_data['day_of_year'] =input_data['date'].dt.dayofyear
        input_data['week_of_year'] =input_data['date'].dt.isocalendar().week
        input_data['day_of_week'] =input_data['date'].dt.dayofweek
        input_data['year'] =input_data['date'].dt.year
        #input_data["is weekend"] = np.where(input_data['day_of_week']>5,1,0)
        input_data['is_month_start'] =input_data['date'].dt.is_month_start.astype(int)
        input_data['is_month_end'] =input_data['date'].dt.is_month_end.astype(int)
        input_data['quarter'] =input_data['date'].dt.quarter
        input_data['is_quarter_start'] =input_data['date'].dt.is_quarter_start.astype(int)
        input_data['is_quarter_end'] =input_data['date'].dt.is_quarter_end.astype(int)
        input_data['is_year_start'] =input_data['date'].dt.is_year_start.astype(int)
        input_data['is_year_end'] =input_data['date'].dt.is_year_end.astype(int)
        return input_data
   

@st.cache(allow_output_mutation=True)
def load_ml_items():
    "Load ML items to use" 
    current_dir = os.path.dirname(__file__)
    with open(os.path.join(current_dir, "ML_items"), "rb") as file:
        loaded_object = pickle.load(file)
    return loaded_object


#loading Ml_items
if "results" not in st.session_state:
    st.session_state["results"] =[]

# config
#instantiating Ml items
loaded_object = load_ml_items()
scaler =loaded_object["scaler"]
model = loaded_object["model"]
encoder=loaded_object["encoder"]

cols_to_scale= ["onpromotion"]
categoricals=["family","city","state","store_type","holiday_type","locale"]

# Structure of form for input and output section
st.markdown("<h1 style:= 'text_align:center;'>Inputs</h1>", unsafe_allow_html=True)
st.markdown("**Capturing Inputs for Prediction**")
left_col, right_col = st.columns(2)

# following lines create boxes for form to retrieve inputs
sales_date = left_col.date_input("select a date:", min_value=train_data["date"].min())
city = right_col.selectbox("City:", options=set(train_data["city"]))
family = left_col.selectbox(
    "Product family:", options=list(train_data["family"].unique())
)
store_nbr = right_col.selectbox("Store number:", options=set(train_data["store_nbr"]))
state = right_col.selectbox("State:", options=set(train_data["state"]))
onpromotion = left_col.number_input(
    "Number of Products on Promotion:",
    min_value=train_data["onpromotion"].min(),
    max_value=train_data["onpromotion"].max(),
    value=train_data["onpromotion"].min(),
)
oil_price = left_col.number_input(
    "Oil Price:",
    min_value=train_data["dcoilwtico"].min(),
    max_value=train_data["dcoilwtico"].max(),
    value=train_data["dcoilwtico"].min(),
)
store_type = right_col.radio(
    "Store type:", options=sorted(set(train_data["store_type"])), horizontal=True
)
cluster = right_col.select_slider("Store cluster:", options=set(train_data["cluster"]))


if left_col.checkbox("Is it a holiday?"):
    holiday_type = left_col.selectbox(
        "Holiday type:", options=set(train_data["holiday_type"])
    )
    locale = left_col.selectbox("locale:", options=set(train_data["locale"]))
    transferred = left_col.radio(
        "Is the holiday transferred?",
        options=set(train_data["transferred"]),
        horizontal=True,
    )
else:
    holiday_type = "Work Day"
    locale = "National"
    transferred = "False"
     
form = st.form(key="information", clear_on_submit=True)
    # submit section
submitted = form.form_submit_button(label ="submit")
if submitted:
    st.success("Done")
#formatting inputs
    input_dict = {"sales date": [sales_date],
        "family": [family],
        "store nbr": [store_nbr],
        "state": [state],
        "onpromotion": [onpromotion],
        "city": [city],
        "oil price": [oil_price],
        "store type": [store_type],
        "cluster": [cluster],
        "holiday_type":[holiday_type],
        "locale":[locale],}

# converting input into a dataframe
    input_data = pd.DataFrame.from_dict(input_dict)
    input_df =input_data.copy()
# converting datatype to datetime
    input_data["sales date"] = pd.to_datetime(input_data["sales date"]).dt.date
# Data Processing
    input_data.drop(columns =['sales date'], inplace =True)

    #Scaling the columns
    input_data[cols_to_scale] = scaler.transform(input_data[cols_to_scale])
    input_data[cols_to_scale] =input_data[cols_to_scale].apply(int)

    #Encoding the categoricals

    encoded_categoricals = encoder.transform(input_data[categoricals])
    encoded_categoricals = pd.DataFrame(encoded_categoricals, columns = encoder.get_feature_names_out().tolist())
    input_data = input_data.join(encoded_categoricals)
    input_data.drop(columns= categoricals, inplace= True)
  

    #Making Predictions
    gbr_pred = model.predict(input_data)
    input_data["sales"] =gbr_pred
    input_df["sales"] =gbr_pred
    result = gbr_pred[0]

    # Adding the predictions to previous predictions
    st.session_state["results"].append(input_df)
    result = pd.concat(st.session_state["results"])

    # resulting prediction results
    st.success(f"**Predicted sales**: {result}")

    # Expander to result previous predictions
    previous_output = st.expander("**Review previous predictions**")
    
    Execution
    Run the streamlit app at
    streamlit run streamlit_project\Favorita_app.py
    previous_output.dataframe(result, use_container_width= True)
