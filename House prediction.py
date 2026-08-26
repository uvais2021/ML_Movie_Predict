from sklearn.linear_model import LinearRegression , Ridge , Lasso 
from sklearn.model_selection import train_test_split
from sklearn.metrics import (
    mean_absolute_error,
    mean_squared_error,
    r2_score
)
import pandas as pd
import numpy as np

movie_data = pd.read_csv("recent_movies_2024_2026.csv")
movie_data = movie_data.drop([
    "title",
    "synopsis",
    "wikipedia_url",
    "poster_url",
    "director",
    "screenwriter",
    "cast"
], axis=1)


movie_data["release_day"] = movie_data["release_day"].fillna(
    movie_data["release_day"].median()
)



# print(movie_data.head(10))


X = movie_data.drop("rating_out_of_10", axis=1)
y = movie_data["rating_out_of_10"]

numerical_cols = X.select_dtypes(
    include=["int64", "float64"]
).columns

categorical_cols = X.select_dtypes(
    include=["object"]
).columns


from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.compose import ColumnTransformer

preprocessor = ColumnTransformer(
    transformers=[
        (
            "num",
            StandardScaler(),
            numerical_cols
        ),
        (
            "cat",
            OneHotEncoder(
                handle_unknown="ignore"
            ),
            categorical_cols
        )
    ]
)


X_train, X_test, y_train, y_test = train_test_split(X,y, test_size=0.2, random_state=42)

from sklearn.pipeline import Pipeline

LR_model = Pipeline([
    ("preprocessor", preprocessor),
    ("model", LinearRegression())
])

Ridge_model = Pipeline([
    ("preprocessor", preprocessor),
    ("model", Ridge(alpha=1.0))
])

Lasso_model = Pipeline([
    ("preprocessor", preprocessor),
    ("model", Lasso(alpha=0.1))
])

for model in [LR_model, Ridge_model, Lasso_model]:

    model.fit(X_train, y_train)

    y_pred = model.predict(X_test)

    mae = mean_absolute_error(y_test, y_pred)
    mse = mean_squared_error(y_test, y_pred)
    rmse = np.sqrt(mse)
    r2 = r2_score(y_test, y_pred)

    print(f"Model: {model['model'].__class__.__name__}")
    print("MAE :", mae)
    print("MSE :", mse)
    print("RMSE:", rmse)
    print("R2  :", r2)
    print("-" * 40)

# ------------------------------------------

# housing = fetch_california_housing(as_frame=True)

# movie_data = housing.frame



# X = movie_data.drop("MedHouseVal", axis=1)
# y = movie_data["MedHouseVal"]

# X_train, X_test, y_train, y_test = train_test_split(
#     X,
#     y,
#     test_size=0.2,
#     random_state=42
# )



# model = LinearRegression()

# model.fit(X_train, y_train)
# y_pred = model.predict(X_test)

# print(y_pred[:10])

# from sklearn.metrics import (
#     mean_absolute_error,
#     mean_squared_error,
#     r2_score
# )
# import numpy as np

# mae = mean_absolute_error(y_test, y_pred)

# mse = mean_squared_error(y_test, y_pred)

# rmse = np.sqrt(mse)

# r2 = r2_score(y_test, y_pred)

# print("MAE :", mae)
# print("MSE :", mse)
# print("RMSE:", rmse)
# print("R2  :", r2)

# from sklearn.ensemble import RandomForestRegressor

# rf_model = RandomForestRegressor(
#     n_estimators=100,
#     random_state=42,
#     n_jobs=-1
# )

# rf_model.fit(X_train, y_train)

# rf_pred = rf_model.predict(X_test)

# print("MAE:", mean_absolute_error(y_test, rf_pred))
# print("RMSE:", np.sqrt(mean_squared_error(y_test, rf_pred)))
# print("R2:", r2_score(y_test, rf_pred))

# new_house = [[
#     5.0,       # MedInc
#     20.0,      # HouseAge
#     6.0,       # AveRooms
#     1.0,       # AveBedrms
#     1000.0,    # Population
#     3.0,       # AveOccup
#     34.0,      # Latitude
#     -118.0     # Longitude
# ]]

# prediction = rf_model.predict(new_house)

# print("Predicted value:", prediction[0])
# print("Approx price: $", prediction[0] * 100000)