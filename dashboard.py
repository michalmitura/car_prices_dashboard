import streamlit as st
import pandas as pd
import plotly.express as px
import numpy as np
import pickle
from sklearn.base import BaseEstimator, TransformerMixin
from sklearn.pipeline import FeatureUnion

class ColumnSelector(BaseEstimator, TransformerMixin):
    def __init__(self, columns):
        self.columns = columns
    
    def fit(self, X, y=None):
        return self
    
    def transform(self, X):
        return X[self.columns]

st.set_page_config(layout='wide')

header = st.container()
dataset = st.container()
model_overview = st.container()
ml_models = st.container()




@st.cache
def load_data():
    # df = pd.read_csv('/Users/michalmitura/Documents/python projects/Car price prediction project/data/data_preprocessed_clean.csv')
    df = pd.read_csv('data/data_preprocessed_clean.csv')
    df.drop('Unnamed: 0', axis=1, inplace=True)
    return df

@st.cache
def load_model_data():
    df = pd.read_csv('data/model_results.csv')
    df.drop('Unnamed: 0', axis=1, inplace=True)
    return df

@st.cache
def load_model(model_name):
    file = open(f'models/{model}.pickle', 'rb')
    model = pickle.load(file)
    return model

@st.cache
def load_feature_data():
    df = pd.read_csv('data/feature_importances_comparison.csv')
    df.drop('Unnamed: 0', axis=1, inplace=True)
    return df





data = load_data()
brands = list(data['brand'].unique())
brands.append('All')

data_ml_models = load_model_data()
feature_data = load_feature_data()

def generate_prediction(input_data, model):
    model = pickle.load(open(f'models/{chosen_model}_model.pickle', 'rb'))
    predicted_value = model.predict(input_data)
    return predicted_value




with header:
    st.title('Car prices dashboard')

with dataset:
    # Choose brand
    st.header('Dataset')
    selected_brand = st.selectbox('Brand', options = brands)
    if selected_brand == 'All':
        selected_data_brand = data
    else:
        selected_data_brand = data[data['brand'] == selected_brand]
    
    # Choose model
    brand_models = list(selected_data_brand['model'].unique())
    brand_models.sort()
    brand_models.append('All')
    selected_model = st.selectbox('Model', options=brand_models)
    if selected_model == 'All':
        selected_data_model = selected_data_brand
        selected_data_model =  selected_data_model.sort_values(by='year')
    else:
        selected_data_model = selected_data_brand[selected_data_brand['model'] == selected_model]
        selected_data_model =  selected_data_model.sort_values(by='year')

    # Choose year
    years = selected_data_model['year'].unique()
    years = years.astype(int)
    years = years.tolist()

    selected_years = st.slider('Year', min_value=min(years), max_value=max(years), value=(min(years), max(years)))
    selected_data_years = selected_data_model[(selected_data_model['year'] >= selected_years[0]) & (selected_data_model['year'] <= selected_years[1])]

    # Choose power range
    powers = selected_data_years['power'].unique()
    powers = powers.astype(int)
    powers = powers.tolist()

    selected_power_range = st.slider('Power', min_value=min(powers), max_value=max(powers), value=(min(powers), max(powers)))
    selected_data_power = selected_data_years[(selected_data_years['power'] >= selected_power_range[0]) & (selected_data_model['power'] <= selected_power_range[1])]

    st.subheader('Chosen vechicles')
    st.write(selected_data_power)

    # Aggregations for plots
    aggregated_by_year = selected_data_power.groupby(by='year').aggregate('mean', numeric_only=True).reset_index()
    aggregated_by_mileage = selected_data_power.groupby(by='mileage').aggregate('mean', numeric_only=True).reset_index()
    aggregated_by_fuel_type  = selected_data_power.groupby(by='fuel_type').aggregate('mean', numeric_only=True).reset_index()
    aggregated_by_origin_country = selected_data_power.groupby(by='origin_country').aggregate('mean', numeric_only=True).reset_index().sort_values(by='price')

    col1, col2 = st.columns(2)
    
    with col1:
        st.subheader('Fuel type')
        st.plotly_chart(figure_or_data=px.pie(data_frame=selected_data_power, names='fuel_type', 
    color_discrete_sequence=px.colors.qualitative.D3), use_container_width=True)

        st.subheader('Price vs. production year')
        fig1_2 = px.line(data_frame=aggregated_by_year, x='year', y='price',
        color_discrete_sequence=px.colors.qualitative.D3)
        fig1_2.update_layout({'plot_bgcolor': 'rgba(0, 0, 0, 0)', 'paper_bgcolor': 'rgba(0, 0, 0, 0)'},
        xaxis=dict(showgrid=False), yaxis=dict(showgrid=False))
        st.plotly_chart(figure_or_data=fig1_2, use_container_width=True)

        st.subheader('Price vs. fuel type')
        fig1_3 = px.bar(data_frame=aggregated_by_fuel_type, x='fuel_type', y='price', 
        color_discrete_sequence=px.colors.qualitative.D3)
        fig1_3.update_layout({'plot_bgcolor': 'rgba(0, 0, 0, 0)', 'paper_bgcolor': 'rgba(0, 0, 0, 0)'},
        xaxis=dict(showgrid=False), yaxis=dict(showgrid=False))
        st.plotly_chart(figure_or_data=fig1_3, use_container_width=True)


    with col2:
        st.subheader('Transmission')
        st.plotly_chart(figure_or_data=px.pie(data_frame=selected_data_power, names='transmission', 
    color_discrete_sequence=px.colors.qualitative.D3), use_container_width=True)

        st.subheader('Price vs. mileage')
        fig2_2 = px.line(data_frame=aggregated_by_mileage, x='mileage', y='price',
        color_discrete_sequence=px.colors.qualitative.D3)
        fig2_2.update_layout({'plot_bgcolor': 'rgba(0, 0, 0, 0)', 'paper_bgcolor': 'rgba(0, 0, 0, 0)'},
        xaxis=dict(showgrid=False), yaxis=dict(showgrid=False))
        st.plotly_chart(figure_or_data=fig2_2, use_container_width=True)
        
        st.subheader('Price vs. country of origin')
        fig2_3 = px.bar(data_frame=aggregated_by_origin_country, x='origin_country', y='price', 
        color_discrete_sequence=px.colors.qualitative.D3)
        fig2_3.update_layout({'plot_bgcolor': 'rgba(0, 0, 0, 0)', 'paper_bgcolor': 'rgba(0, 0, 0, 0)'},
        xaxis=dict(showgrid=False), yaxis=dict(showgrid=False))
        st.plotly_chart(figure_or_data=fig2_3, use_container_width=True)

with model_overview:
    with col1:
        st.header('Model overview')
        st.write(data_ml_models)
    with col2:
        st.header('Feature importances')
        st.dataframe(feature_data, use_container_width=True, height=211)


with ml_models:
    with col1:

        st.header('Predictions')
        chosen_model_input = st.selectbox('Choose ML model', options=['Gradient Boosting Regressor', 'Decision Tree Regressor'])
        if chosen_model_input == 'Gradient Boosting Regressor':
            chosen_model = 'xgb'
        elif chosen_model_input == 'Decision Tree Regressor':
            chosen_model = 'dtr'

        brand_ml = st.selectbox('Choose brand', options=brands)
        model_ml = st.selectbox('Choose model', options=data[data['brand']==brand_ml]['model'].unique())
        category_ml = 'Osobowe'
        registration_ml_chosen = st.selectbox('Choose if registrated', options=['Tak', 'Nie'])
        if registration_ml_chosen == 'Tak':
            registration_ml = 'Tak'
        else:
            registration_ml = 'No'
        seller_ml_chosen = st.selectbox('Choose seller', options=['Osoba prywatna', 'Firma'])
        if seller_ml_chosen == 'Osoba prywatna':
            seller_ml = 'Osoby prywatnej'
        else:
            seller_ml = 'Firmy'
        chosen_model_data = data[data['model'] == model_ml]
        years_ml = chosen_model_data['year'].astype(int).tolist()
        if min(years_ml) < max(years_ml):
            year_ml = st.slider('Choose year', min_value=min(years_ml), max_value=max(years_ml))
        else:
            year_ml = max(years_ml)
            st.text(str(year_ml))
        mileage_ml = st.text_input('Input mileage')
        if mileage_ml == '':
            mileage_ml = 0
        else:
            mileage_ml = float(mileage_ml)       
        engine_sizes_ml = sorted(chosen_model_data['engine_size'].unique().tolist())
        engine_size_ml = float(st.selectbox('Choose engine size [cm3]', options=engine_sizes_ml))
        fuel_type_ml = st.selectbox('Choose fuel type', options=data[data['model'] == model_ml]['fuel_type'].unique())
        power_ml = st.slider('Choose power', min_value=min(chosen_model_data['power'].unique().tolist()), max_value=max(chosen_model_data['power'].unique().tolist()), step=1.0)
        transmission_ml = st.selectbox('Choose transmission type', options = ['Automatyczna', 'Manualna'])
    with col2:
        st.header('')
        st.subheader('')
        drive_train_ml = st.selectbox('Choose drive train', options=data[data['model'] == model_ml]['drive_train'].unique())
        body_type_ml = st.selectbox('Choose body type', options=data[data['model'] == model_ml]['body_type'].unique())
        colour_ml = st.selectbox('Choose colour', options=data[data['model'] == model_ml]['colour'].unique())
        colour_type_ml = st.selectbox('Choose colour type', options=data[data['model'] == model_ml]['colour_type'].unique())
        origin_country_ml = st.selectbox('Choose origin country', options=data[data['model'] == model_ml]['origin_country'].unique())
        condition_ml = st.selectbox('Choose condition', options=data[data['model'] == model_ml]['condition'].unique())
        accident_free_ml = st.selectbox('Choose if accident free', options=data[data['model'] == model_ml]['accident_free'].unique())
        aso_serviced_ml = st.selectbox('Choose if aso serviced', options=data[data['model'] == model_ml]['aso_serviced'].unique())
        tuning_ml = st.selectbox('Choose if tuned', options=data[data['model'] == model_ml]['tuning'].unique())
        first_owner_ml = st.selectbox('Choose if first owner', options=data[data['model'] == model_ml]['first_owner'].unique())
        dpf_ml = st.selectbox('Choose if has DPF', options=data[data['model'] == model_ml]['dpf'].unique())
        damaged_ml = st.selectbox('Choose if damaged', options=data[data['model'] == model_ml]['damaged'].unique())
        registrated_as_vintage_ml = st.selectbox('Choose if registrated as vintage', options=data[data['model'] == model_ml]['registrated_as_vintage'].unique())
        right_side_wheel_ml = st.selectbox('Choose if has right side weel', options=data[data['model'] == model_ml]['right_side_wheel'].unique())
        registrated_in_poland_ml = st.selectbox('Choose if registrated in Poland', options=data[data['model'] == model_ml]['right_side_wheel'].unique())
        
        feature_values = [seller_ml, category_ml, registration_ml, brand_ml, model_ml, year_ml,mileage_ml,
            engine_size_ml, fuel_type_ml, power_ml, transmission_ml,
            drive_train_ml, body_type_ml, colour_ml, colour_type_ml, origin_country_ml,
            condition_ml, accident_free_ml, aso_serviced_ml, tuning_ml, first_owner_ml,
            dpf_ml, damaged_ml, registrated_as_vintage_ml, right_side_wheel_ml, registrated_in_poland_ml]
        feature_keys = ['seller', 'category', 'registration_no', 'brand', 'model',
            'year', 'mileage', 'engine_size', 'fuel_type', 'power', 'transmission',
            'drive_train', 'body_type', 'colour', 'colour_type', 'origin_country',
            'condition', 'accident_free', 'aso_serviced', 'tuning', 'first_owner',
            'dpf', 'damaged', 'registrated_as_vintage', 'right_side_wheel',
            'registrated_in_poland']

        input_ml_dict = dict(zip(feature_keys, feature_values))
        input_data_ml = pd.DataFrame(input_ml_dict, index=[0])

    with col1:
        prediction = generate_prediction(input_data=input_data_ml, model=chosen_model)[0]
        st.text(f'Predicted price is {np.round(prediction, 2)} zł')
