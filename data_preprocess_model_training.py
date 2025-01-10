import opendatasets as od
import matplotlib.pyplot as plt
import seaborn as sns
import pandas as pd
import numpy as np
import matplotlib
import os
import plotly.express as px
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import MinMaxScaler, OneHotEncoder
from sklearn.tree import DecisionTreeClassifier, plot_tree, export_text
from sklearn.metrics import accuracy_score, confusion_matrix
from sklearn.ensemble import RandomForestClassifier
import joblib

pd.set_option('display.max_columns', None)
pd.set_option('display.max_rows', 150)
sns.set_style('darkgrid')
matplotlib.rcParams['font.size'] = 14
matplotlib.rcParams['figure.figsize'] = (10, 6)
matplotlib.rcParams['figure.facecolor'] = '#00000000'

od.download('https://www.kaggle.com/jsphyg/weather-dataset-rattle-package')
os.listdir(".\weather-dataset-rattle-package")
raw_df = pd.read_csv(os.path.join(".\weather-dataset-rattle-package","weatherAUS.csv"))
print(raw_df.info())

raw_df.dropna(subset=["RainTomorrow"], inplace=True)
print(raw_df.info())

plt.title("No. of Rows per Year")
sns.countplot(x=pd.to_datetime(raw_df.Date).dt.year)
plt.show()

year = pd.to_datetime(raw_df.Date).dt.year

train_df = raw_df[year<2015]
val_df = raw_df[year==2015]
test_df = raw_df[year>2015]
print(train_df.shape, val_df.shape, test_df.shape)

input_cols = list(train_df.columns)[1:-1]
target_col = "RainTomorrow"

train_inputs = train_df[input_cols].copy()
train_targets = train_df[target_col].copy()

val_inputs = val_df[input_cols].copy()
val_targets = val_df[target_col].copy()

test_inputs = test_df[input_cols].copy()
test_targets = test_df[target_col].copy()

numeric_cols = train_inputs.select_dtypes(include=np.number).columns.tolist()
categorical_cols = train_inputs.select_dtypes('object').columns.tolist()


print(train_inputs[numeric_cols].isna().sum().sort_values(ascending=False))

imputer = SimpleImputer(strategy="mean").fit(raw_df[numeric_cols])
train_inputs[numeric_cols] = imputer.transform(train_inputs[numeric_cols])
val_inputs[numeric_cols] = imputer.transform(val_inputs[numeric_cols])
test_inputs[numeric_cols] = imputer.transform(test_inputs[numeric_cols])

scaler = MinMaxScaler().fit(raw_df[numeric_cols])
train_inputs[numeric_cols] = scaler.transform(train_inputs[numeric_cols])
val_inputs[numeric_cols] = scaler.transform(val_inputs[numeric_cols])
test_inputs[numeric_cols] = scaler.transform(test_inputs[numeric_cols])

print(train_inputs.describe().loc[["min", "max"]])

encoder = OneHotEncoder(sparse_output=False, handle_unknown='ignore').fit(raw_df[categorical_cols])
encoded_cols = list(encoder.get_feature_names_out(categorical_cols))
train_inputs[encoded_cols] = encoder.transform(train_inputs[categorical_cols])
val_inputs[encoded_cols] = encoder.transform(val_inputs[categorical_cols])
test_inputs[encoded_cols] = encoder.transform(test_inputs[categorical_cols])

#print(test_inputs)

X_train = train_inputs[numeric_cols + encoded_cols]
X_val = val_inputs[numeric_cols + encoded_cols]
X_test = test_inputs[numeric_cols + encoded_cols]


##### n_jobs allows the random forest to use mutiple parallel workers to train decision trees,
#  and random_state=42 ensures that the we get the same results for each execution

# no max depth and max leaf nodes
base_model = RandomForestClassifier(n_jobs=-1, random_state=42)
base_model.fit(X_train, train_targets)
base_acc_train = base_model.score(X_train, train_targets)
base_acc_val = base_model.score(X_val, val_targets)
print("Base Train Accuracy:", base_acc_train, "Base Validation Accuracy:",base_acc_val)

train_prob = base_model.predict_proba(X_train)
print(train_prob)

## We can can access individual decision trees using `model.estimators_`
print(len(base_model.estimators_), base_model.estimators_[0])

plt.figure(figsize=(50, 25))
plot_tree(base_model.estimators_[0], max_depth=3, feature_names=X_train.columns, filled=True, rounded=True, class_names=base_model.classes_)
plt.show()

plt.figure(figsize=(50, 25))
plot_tree(base_model.estimators_[17], max_depth=3, feature_names=X_train.columns, filled=True, rounded=True, class_names=base_model.classes_)
plt.show()

importance_df = pd.DataFrame({
    "feature": X_train.columns,
    "importance": base_model.feature_importances_
}).sort_values("importance", ascending=False)

print(importance_df.head(10))
plt.title("Feature Importance")
sns.barplot(data=importance_df.head(10), x="importance", y="feature")
plt.show()


'''
def n_estimator_error(ns):
    ### if get error remove n_jobs
    model = RandomForestClassifier(n_jobs=-1, random_state=42, n_estimators=ns)
    model.fit(X_train, train_targets)
    train_error = 1 - model.score(X_train, train_targets)
    val_error = 1 - model.score(X_val, val_targets)
    return {'n_estimator': ns, 'Training Error': train_error, 'Validation Error': val_error}

errors_df = pd.DataFrame([n_estimator_error(ns) for ns in range(10, 100, 10)])
print(errors_df)

plt.figure()
plt.plot(errors_df['n_estimator'], errors_df['Training Error'])
plt.plot(errors_df['n_estimator'], errors_df['Validation Error'])
plt.title('Training vs. Validation Error')
plt.xticks(range(10, 100, 10))
plt.xlabel('n_estimator')
plt.ylabel('Prediction Error (1 - Accuracy)')
plt.legend(['Training', 'Validation'])
plt.show()
'''

def test_params(**params):
    model = RandomForestClassifier(random_state=42,  **params).fit(X_train, train_targets)
    return model.score(X_train, train_targets), model.score(X_val, val_targets)
'''
print(test_params(max_depth=26))
print(test_params(max_leaf_nodes=2**5))
print(test_params(max_features='log2'))
print(test_params(max_features=3))
print(test_params(min_samples_split=100, min_samples_leaf=60))
print(test_params(min_impurity_decrease=1e-2))
'''
#70 percent of data
print(test_params(max_samples=0.6))

print(train_targets.value_counts()/ len(train_targets))
print(test_params(class_weight='balanced'))
print(test_params(class_weight={'No':1, 'Yes':3}))

model_final = RandomForestClassifier(
                               random_state=42, 
                               n_estimators=100,
                               max_features=20,
                               max_depth=30, 
                               class_weight={'No': 1, 'Yes': 1.5}).fit(X_train, train_targets)
accuracy_final_train = model_final.score(X_train, train_targets)
accuracy_final_val = model_final.score(X_val, val_targets)
accuracy_final_test = model_final.score(X_test, test_targets)
print(accuracy_final_train, accuracy_final_val, accuracy_final_test)


def predict_input(model, single_input):
    input_df = pd.DataFrame([single_input])
    input_df[numeric_cols] = imputer.transform(input_df[numeric_cols])
    input_df[numeric_cols] = scaler.transform(input_df[numeric_cols])
    input_df[encoded_cols] = encoder.transform(input_df[categorical_cols])
    X_input = input_df[numeric_cols + encoded_cols]
    pred = model.predict(X_input)[0]
    prob = model.predict_proba(X_input)[0][list(model.classes_).index(pred)]
    return pred, prob


new_input = {'Date': '2021-06-19',
             'Location': 'Launceston',
             'MinTemp': 23.2,
             'MaxTemp': 33.2,
             'Rainfall': 10.2,
             'Evaporation': 4.2,
             'Sunshine': np.nan,
             'WindGustDir': 'NNW',
             'WindGustSpeed': 52.0,
             'WindDir9am': 'NW',
             'WindDir3pm': 'NNE',
             'WindSpeed9am': 13.0,
             'WindSpeed3pm': 20.0,
             'Humidity9am': 89.0,
             'Humidity3pm': 58.0,
             'Pressure9am': 1004.8,
             'Pressure3pm': 1001.5,
             'Cloud9am': 8.0,
             'Cloud3pm': 5.0,
             'Temp9am': 25.7,
             'Temp3pm': 33.0,
             'RainToday': 'Yes'}


print(predict_input(model_final, new_input))

aussie_rain = {
    'model': model_final,
    'imputer': imputer,
    'scaler': scaler,
    'encoder': encoder,
    'input_cols': input_cols,
    'target_col': target_col,
    'numeric_cols': numeric_cols,
    'categorical_cols': categorical_cols,
    'encoded_cols': encoded_cols
}

joblib.dump(aussie_rain, 'aussie_rain.joblib')
aussie_rain2 = joblib.load('aussie_rain.joblib')
test_preds2 = aussie_rain2['model'].predict(X_test)
print(accuracy_score(test_targets, test_preds2))