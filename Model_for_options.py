#!/usr/bin/env python
# coding: utf-8

# In[1]:


import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.linear_model import LinearRegression, BayesianRidge, Lasso
from sklearn.ensemble import RandomForestRegressor
from sklearn.tree import DecisionTreeRegressor
from sklearn.pipeline import Pipeline
import xgboost as xgb
import lightgbm as lgb
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split
from sklearn.metrics import r2_score, mean_squared_error
from scipy import stats
import warnings
warnings.filterwarnings('ignore')


# In[2]:


def options_model(company_name, file_name, start_test):
    path = file_name
    df_options = pd.read_csv(path)
    df_options['Symbol'] = company_name
    df_options['Date'] = pd.to_datetime(df_options['Date'])

    df_options.replace({'Option_type':{'call':0,'put':1,'Call':0,'Put':1}},inplace=True)

    corr=df_options.corrwith(df_options['future_option_one_day'])
    corr = corr.squeeze()
    # Create the heatmap
    plt.figure(figsize=(5, 20))
    sns.heatmap(corr.to_frame(), cmap="coolwarm", annot=True, fmt='.3g')
    plt.title('Correlation predicted')
    plt.show()
    
#     date_str = '01/01/2006'
#     date_obj = datetime.strptime(date_str, '%m/%d/%Y')
#     final_date = pd.Timestamp(date_obj)
#     df_options['Date'] = pd.to_datetime(df_options['Date'], errors='coerce')
#     comparison_result = df_options['Date'].iloc[0] <= final_date
#     if (comparison_result):
#         date_obj = datetime.strptime(start_test, '%m/%d/%Y')
#         final_date = pd.Timestamp(date_obj)
#         df_test = df_options[df_options['Date'] >= final_date]
#         df_options = df_options[df_options['Date'] < final_date]
#     else:
#         df_options['Date'] = pd.to_datetime(df_options['Date'])
#         Date_counts = df_options['Date'].value_counts().to_frame()
#         Date_counts.rename(columns={'Date': 'value_counts'}, inplace=True)
#         Date_counts.index.name = 'Date'
#         Date_counts_sorted = Date_counts.sort_index()
#         midpoint = len(Date_counts_sorted) // 2
#         first_half_dates = Date_counts_sorted.iloc[:midpoint].index
#         second_half_dates = Date_counts_sorted.iloc[midpoint:].index
#         df_first_half = df_options[df_options['Date'].isin(first_half_dates)]
#         df_second_half = df_options[df_options['Date'].isin(second_half_dates)]
#         df_options = df_first_half.reset_index(drop=True)
#         df_test = df_second_half.reset_index(drop=True)

    features = ["Strike_Price", "Stock_price",'Option_price', "Rate",
                "Rolling_Std", "implied_volatility",'theta',
                "Vega", "delta", "gamma", "rho",'sharpe_ratio']
    for i in features:
        pearson_coef, p_value = stats.pearsonr(df_options[i], df_options["future_option_one_day"])
        print(f"The Pearson Correlation Coefficient of {i} is", pearson_coef, " with a P-value of P =", p_value)

    numerical_cols = ["Strike_Price", "Stock_price",'Option_price', "Rate",
                      "Rolling_Std",'Option_type', "implied_volatility",
                      "Vega", "delta", "gamma", 'sharpe_ratio']
    X = df_options[numerical_cols]
    y = df_options["future_option_one_day"]
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=42)

    # Define and fit the linear regression model
    columns = ['Model','Training R-squared','Test R-squared','Test MSE']
    df_models = pd.DataFrame(columns=columns)

    # Define and fit the linear regression model
    lm = LinearRegression()
    lm.fit(X_train, y_train)

    # Calculate R-squared score
    train_r2_LinearRegression = r2_score(y_train, lm.predict(X_train))

    test_r2_LinearRegression = r2_score(y_test, lm.predict(X_test))
    
    # Calculate Mean Squared Error (MSE) for the test set
    test_mse_LinearRegression = mean_squared_error(y_test, lm.predict(X_test))
    
    # Create a new row as a dictionary
    new_row = {"Symbol" : company_name,
        'Model':'linear regression',
          'Training R-squared': train_r2_LinearRegression,
               'Test R-squared': test_r2_LinearRegression,
               'Test MSE': test_mse_LinearRegression}
    df_models = pd.concat([df_models, pd.DataFrame([new_row])], ignore_index=True)


    bayesian_regressor = BayesianRidge()
    # Train the model
    bayesian_regressor.fit(X_train, y_train)

    train_r2_bayesian = r2_score(y_train,bayesian_regressor.predict(X_train))

    # Calculate R-squared score for the test set
    test_r2_bayesian = r2_score(y_test, bayesian_regressor.predict(X_test))

    # Calculate Mean Squared Error (MSE) as a performance metric
    mse_bayesian = mean_squared_error(y_test, bayesian_regressor.predict(X_test))
    new_row = {"Symbol" : company_name,
        'Model':'Bayesian Ridge',
        'Training R-squared': train_r2_bayesian,
               'Test R-squared': test_r2_bayesian,
               'Test MSE': mse_bayesian}

    df_models = pd.concat([df_models, pd.DataFrame([new_row])], ignore_index=True)


    # lasso_regressor = Lasso(alpha=0.1)  # Adjust alpha for regularization strength

    # Train the model
    lasso_pipeline = Pipeline([
        ('scaler', StandardScaler()),
        ('lasso', Lasso(alpha=0.1))  # Adjust alpha as needed
    ])
    lasso_pipeline.fit(X_train, y_train)
    # lasso_regressor.fit(X_train, y_train)

    # Calculate R-squared score for the training set
    train_r2_lasso_pipeline = r2_score(y_train, lasso_pipeline.predict(X_train))

    # Calculate R-squared score for the test set
    test_r2_lasso_pipeline = r2_score(y_test, lasso_pipeline.predict(X_test))

    # Calculate Mean Squared Error (MSE) as a performance metric
    mse_lasso_pipeline = mean_squared_error(y_test, lasso_pipeline.predict(X_test))
    new_row = {"Symbol" : company_name,
        'Model':'Lasso',
        'Training R-squared': train_r2_lasso_pipeline,
               'Test R-squared': test_r2_lasso_pipeline,
               'Test MSE': mse_lasso_pipeline}

    df_models = pd.concat([df_models, pd.DataFrame([new_row])], ignore_index=True)
                          
    tree_regressor = DecisionTreeRegressor(max_depth=3, random_state=42)  # Adjust max_depth for tree depth

    # Train the model
    tree_regressor.fit(X_train, y_train)

    # Calculate R-squared score for the training set
    train_r2_DecisionTree = r2_score(y_train, tree_regressor.predict(X_train))

    # Calculate R-squared score for the test set
    test_r2_DecisionTree = r2_score(y_test, tree_regressor.predict(X_test))

    # Calculate Mean Squared Error (MSE) as a performance metric
    mse_DecisionTree = mean_squared_error(y_test, tree_regressor.predict(X_test))
    new_row = {"Symbol" : company_name,
               'Model':'Decision Tree',
               'Training R-squared': train_r2_DecisionTree,
               'Test R-squared': test_r2_DecisionTree,
               'Test MSE': mse_DecisionTree}

    df_models = pd.concat([df_models, pd.DataFrame([new_row])], ignore_index=True)


    # random forest
    rf_regressor = RandomForestRegressor(n_estimators=100, random_state=42)
    # Train the model
    rf_regressor.fit(X_train, y_train)

    train_r2_RandomForest = rf_regressor.score(X_train, y_train)

    test_r2_RandomForest = rf_regressor.score(X_test, y_test)

    mse_RandomForest = mean_squared_error(y_test, rf_regressor.predict(X_test))
    new_row = {"Symbol" : company_name,
               'Model':'random forest',
               'Training R-squared': train_r2_RandomForest,
               'Test R-squared': test_r2_RandomForest,
               'Test MSE': mse_RandomForest}

    df_models = pd.concat([df_models, pd.DataFrame([new_row])], ignore_index=True)
    df_models = df_models.round(4)
 
    
    X = df_test[["Strike_Price", "Stock_price",'Option_price', "Rate",
                      "Rolling_Std",'Option_type', "implied_volatility",
                      "Vega", "delta", "gamma", 'sharpe_ratio']]

    for col in X.columns:
        X[col] = X[col].astype(str).str.replace(',', '').astype(float)
    
    df_test['predict_future_price_random_forest'] = rf_regressor.predict(X)
    df_test['predict_future_price_decision_tree'] = tree_regressor.predict(X)
    df_test['predict_future_price_linear_regression'] = lm.predict(X)
    df_test['predict_future_price_bayesian_regressor'] = bayesian_regressor.predict(X)
    df_test['predict_future_price_lasso_pipeline'] = lasso_pipeline.predict(X)
    
    for col in ['future_option_one_day', 'predict_future_price_random_forest','predict_future_price_decision_tree',
                'predict_future_price_linear_regression','predict_future_price_bayesian_regressor','predict_future_price_lasso_pipeline']:  # Replace with your column names
        df_test[col] = pd.to_numeric(df_test[col], errors='coerce')
    df_test = df_test.dropna()
    plt.figure(figsize=(10, 6))  # Set figure size

    # Plotting each line separately
    sns.lineplot(x="Date", y="future_option_one_day", data=df_test, label='Actual')
    sns.lineplot(x="Date", y="predict_future_price_random_forest", data=df_test, label='Random Forest', color='green', linestyle='--')

    # Add title and labels
    plt.title('Future Price Prediction Comparison')
    plt.xlabel('Date')
    plt.ylabel('Price')

    # Rotate x-axis labels for better visibility
    plt.xticks(rotation=45)

    # Add legend
    plt.legend()

    # Adjust layout to prevent overlapping labels
    plt.tight_layout()

    # Show the plot
    plt.show()

    # Repeat the process for other predictions
    plt.figure(figsize=(10, 6))  # Set figure size

    sns.lineplot(x="Date", y="future_option_one_day", data=df_test, label='Actual')
    sns.lineplot(x="Date", y="predict_future_price_decision_tree", data=df_test, label='Decision Tree', color='green', linestyle='-.')
    plt.title('Future Price Prediction Comparison')
    plt.xlabel('Date')
    plt.ylabel('Price')
    plt.xticks(rotation=45)
    plt.legend()
    plt.tight_layout()
    plt.show()

    # Repeat for other prediction models
    # ...

    # Repeat the process for each prediction model
    # ...

    plt.figure(figsize=(10, 6))  # Set figure size

    sns.lineplot(x="Date", y="future_option_one_day", data=df_test, label='Actual')
    sns.lineplot(x="Date", y="predict_future_price_linear_regression", data=df_test, label='Linear Regression', color='red', linestyle=':')
    plt.title('Future Price Prediction Comparison')
    plt.xlabel('Date')
    plt.ylabel('Price')
    plt.xticks(rotation=45)
    plt.legend()
    plt.tight_layout()
    plt.show()

    plt.figure(figsize=(10, 6))  # Set figure size

    sns.lineplot(x="Date", y="future_option_one_day", data=df_test, label='Actual')
    sns.lineplot(x="Date", y="predict_future_price_bayesian_regressor", data=df_test, label='Bayesian Regressor', color='purple', linestyle='-')
    plt.title('Future Price Prediction Comparison')
    plt.xlabel('Date')
    plt.ylabel('Price')
    plt.xticks(rotation=45)
    plt.legend()
    plt.tight_layout()
    plt.show()

    plt.figure(figsize=(10, 6))  # Set figure size

    sns.lineplot(x="Date", y="future_option_one_day", data=df_test, label='Actual')
    sns.lineplot(x="Date", y="predict_future_price_lasso_pipeline", data=df_test, label='Lasso Pipeline', color='orange', linestyle='-')
    plt.title('Future Price Prediction Comparison')
    plt.xlabel('Date')
    plt.ylabel('Price')
    plt.xticks(rotation=45)
    plt.legend()
    plt.tight_layout()
    plt.show()

#     df_test['predict_future_price_random_forest'] = rf_regressor.predict(X)
#     df_test['predict_future_price_decision_tree'] = tree_regressor.predict(X)
#     df_test['predict_future_price_linear_regression'] = lm.predict(X)
#     df_test['predict_future_price_bayesian_regressor'] = bayesian_regressor.predict(X)
#     df_test['predict_future_price_lasso_pipeline'] = lasso_pipeline.predict(X)
#     plt.figure(figsize=(10, 6))

#     # Plotting each line with customized details
#     sns.lineplot(x="Date", y="Future_One_day", data=df_test, label='Actual')
#     sns.lineplot(x="Date", y="predict_future_price_random_forest", data=df_test, label='Random Forest', color='blue', linestyle='--')
#     sns.lineplot(x="Date", y="predict_future_price_decision_tree", data=df_test, label='Decision Tree', color='green', linestyle='-.')
#     sns.lineplot(x="Date", y="predict_future_price_linear_regression", data=df_test, label='Linear Regression', color='red', linestyle=':')
#     sns.lineplot(x="Date", y="predict_future_price_bayesian_regressor", data=df_test, label='Bayesian Regressor', color='purple', linestyle='-')
#     sns.lineplot(x="Date", y="predict_future_price_lasso_pipeline", data=df_test, label='Lasso Pipeline', color='orange', linestyle='-')

#     # Add title and labels
#     plt.title('Future Price Prediction Comparison')
#     plt.xlabel('Date')
#     plt.ylabel('Price')

#     # Rotate x-axis labels for better visibility
#     plt.xticks(rotation=45)

#     # Add legend
#     plt.legend()

#     # Adjust layout to prevent overlapping labels
#     plt.tight_layout()

#     # Show the plot
#     plt.show()
    return df_models,df_test


# In[3]:


companies = [
    {'name': 'MSFT'},
    {'name': 'NVDA'},
    {'name': 'TSLA'},
    {'name': 'SMCI'},
    {'name': 'PANW'},
    {'name': 'AMZN'},
    {'name': 'META'},
    {'name': 'AAPL'},
    {'name': 'GOOGL'},
    {'name': 'ADBE'},
    {'name': 'AMD'},
    {'name': 'CRWD'},
    {'name': 'FTNT'},
    {'name': 'TSM'},
    {'name': 'OXY'},
    {'name': 'CRM'},
    {'name': 'NIO'},
    {'name': 'SPCE'},
    {'name': 'RIVN'},
    {'name': 'LCID'},
    {'name': 'U'},
    {'name': 'V'},
    {'name': 'UNH'},
    {'name': 'UBER'},
    {'name': 'VRT'},
    {'name': 'VKTX'},
    {'name': 'WDAY'},
    {'name': 'ZM'},
    {'name': 'XLE_ETF'},
    {'name': 'XLF_ETF'},
    {'name': 'SMH_ETF'},
    {'name': 'SOUN'},
    {'name': 'ROKU'},
    {'name': 'ROIV'},
    {'name': 'RBLX'},
    {'name': 'TWLO'},
    {'name': 'TEAM'},
    {'name': 'SYM'},
    {'name': 'SQ'},
    {'name': 'AEYE'},
    {'name': 'ANET'},
    {'name': 'ARKG'},
    {'name': 'ARKK_ETF'},
    {'name': 'ASML'},
    {'name': 'BILL'},
    {'name': 'CELH'},
    {'name': 'CMG'},
    {'name': 'COIN'},
    {'name': 'COST'},
    {'name': 'CYBR'},
    {'name': 'DDOG'},
    {'name': 'DKNG'},
    {'name': 'DT'},
    {'name': 'ELF'},
    {'name': 'FTI'},
    {'name': 'GTEK_ETF'},
    {'name': 'HUBS'},
    {'name': 'INTC'},
    {'name': 'KLAC'},
    {'name': 'LLY'},
    {'name': 'LPLA'},
    {'name': 'MA'},
    {'name': 'MELI'},
    {'name': 'MLTX'},
    {'name': 'MRVL'},
    {'name': 'MSI'},
    {'name': 'ORCL'},
    {'name': 'PATH'}
]


# In[4]:


def process_company(company,start_date):
    print('________________________________________________________________________________________________________________')
    print(company['name'])
    company_model = f"D:/Quantum/Codes/Option_data_from_historical/options_data/companies_train/{company['name']}_final_data_for_options.csv"
    model, test_data = options_model(company['name'], company_model, start_date)
    display(model)
    display(test_data)
    test_data.to_csv(f"D:/Quantum/Codes/Option_data_from_historical/options_data/portfolio_test/done3/test_data_{company['name']}.csv", index=False)
    


# In[5]:


start_date = '06/13/2014'
for company in companies:
    process_company(company,start_date)


# In[6]:


def process_company_tests(company):
    print('________________________________________________________________________________________________________________')
    print(company['name'])
    df_options_path = f"D:/Quantum/Codes/Option_data_from_historical/options_data/portfolio_test/done3/test_data_{company['name']}.csv"
    df_options = pd.read_csv(df_options_path)
    df_options['Date'] = pd.to_datetime(df_options['Date'])
    Date_counts = df_options['Date'].value_counts().to_frame()
    Date_counts.rename(columns={'Date': 'value_counts'}, inplace=True)
    Date_counts.index.name = 'Date'
    Date_counts_sorted = Date_counts.sort_index()
    Date_counts_sorted.index = pd.to_datetime(Date_counts_sorted.index)
    final_date = Date_counts_sorted.index[-1]
    final_date_minus_30 = final_date - pd.Timedelta(days=61)
    final_date_str = final_date.strftime('%Y-%m-%d')
    final_date_minus_30_str = final_date_minus_30.strftime('%Y-%m-%d')
    split_date = final_date_minus_30_str
    df_test2 = df_options[df_options['Date'] >= split_date]
    df_test1 = df_options[df_options['Date'] < split_date]
    df_test1.to_csv(f"D:/Quantum/Codes/Option_data_from_historical/options_data/portfolio_test/period1/test_data_{company['name']}.csv", index=False)
    df_test2.to_csv(f"D:/Quantum/Codes/Option_data_from_historical/options_data/portfolio_test/period2/test_data_{company['name']}.csv", index=False)


# In[7]:


for company in companies:
    process_company_tests(company)


# In[ ]:




