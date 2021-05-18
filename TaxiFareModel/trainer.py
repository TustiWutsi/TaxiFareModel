from sklearn.preprocessing import OneHotEncoder
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.pipeline import Pipeline
from sklearn.compose import ColumnTransformer
from TaxiFareModel.encoders import TimeFeaturesEncoder, DistanceTransformer 
from TaxiFareModel.utils import compute_rmse
from TaxiFareModel.data import get_data, clean_data

class Trainer():
    def __init__(self, X, y):
        """
            X: pandas DataFrame
            y: pandas Series
        """
        self.pipeline = None
        self.X = X
        self.y = y

    def set_pipeline(self):
        '''returns a pipelined model'''
        dist_pipe = Pipeline([
            ('dist_trans', DistanceTransformer()),
            ('stdscaler', StandardScaler())
        ])
        time_pipe = Pipeline([
            ('time_enc', TimeFeaturesEncoder('pickup_datetime')),
            ('ohe', OneHotEncoder(handle_unknown='ignore'))
        ])
        preproc_pipe = ColumnTransformer([
            ('distance', dist_pipe, ["pickup_latitude", "pickup_longitude", 'dropoff_latitude', 'dropoff_longitude']),
            ('time', time_pipe, ['pickup_datetime'])
        ], remainder="drop")
        pipe = Pipeline([
            ('preproc', preproc_pipe),
            ('linear_model', LinearRegression())
        ])
        self.pipeline = pipe

    def run(self):
        '''returns a trained pipelined model'''
        self.set_pipeline()
        self.pipeline.fit(self.X, self.y)

    def evaluate(self, X_test, y_test):
        '''returns the value of the RMSE'''
        y_pred = self.pipeline.predict(X_test)
        rmse = compute_rmse(y_pred, y_test)
        return rmse


if __name__ == "__main__":
    df = get_data(nrows=10_000)
    df = clean_data(df, test=False)
    y = df["fare_amount"]
    X = df.drop("fare_amount", axis=1)
    X_train, X_val, y_train, y_val = train_test_split(X, y, test_size=0.15)
    trainer = Trainer(X_train,y_train)
    trainer.run()
    rmse = trainer.evaluate(X_val, y_val)
    print(rmse)