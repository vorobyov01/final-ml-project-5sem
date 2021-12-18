import io
import pandas as pd
import numpy as np

from ensembles import RandomForestMSE, GradientBoostingMSE

from sklearn.ensemble import RandomForestRegressor, GradientBoostingRegressor # <- needs to be replaced
from sklearn.metrics import mean_squared_error # <- needs to be replaced

from collections import namedtuple
from flask_wtf import FlaskForm
from flask_bootstrap import Bootstrap
from flask import Flask, request, url_for
from flask import render_template, redirect, send_file

from flask_wtf.file import FileAllowed
from wtforms.validators import DataRequired
from wtforms import StringField, SubmitField, FileField, SelectField

app = Flask(__name__, template_folder='templates')
app.config['BOOTSTRAP_SERVE_LOCAL'] = True
app.config['SECRET_KEY'] = 'sergei'
Bootstrap(app)

class UploadForm(FlaskForm):
    file_path = FileField('Choose dataset', validators=[
        DataRequired('Specify file'),
        FileAllowed(['csv'], 'CSV only!')
    ])
    submit = SubmitField('Upload Data')

class ParamsForm(FlaskForm):
    model_list = ['Random Forest MSE', 'Gradient Boosting MSE']
    model = SelectField('Select model', choices=model_list)
    p1 = StringField('n_estimators', validators=[DataRequired()], default='100')
    p2 = StringField('feature_subsample_size', validators=[DataRequired()], default='0.7')
    p3 = StringField('max_depth', validators=[DataRequired()], default='6')
    p4 = StringField('learning_rate (will be ignored for Random Forest)', validators=[DataRequired()], default='0.05')
    submit = SubmitField('Go to train page')

class ValidateForm(FlaskForm):
    file_path = FileField('Choose dataset', validators=[
        DataRequired('Specify file'),
        FileAllowed(['csv'], 'CSV only!')
    ])
    submit = SubmitField('Upload Data & Validate')
    # score = StringField('Score', validators=[DataRequired()])
    # sentiment = StringField('Sentiment', validators=[DataRequired()])
    # submit = SubmitField('Try Again')

class TrainForm(FlaskForm):
    submit = SubmitField('Train')

class Dataset():
    df = None
    X = None
    y = None
    val = None
df = Dataset()

class Model():
    clf = None
    trained = False
    params = {'n_estimators': 200, 'feature_subsample_size': 0.7, 'max_depth': 4, 'learning_rate': 0.05}
clf = Model()

@app.route('/', methods=['POST', 'GET'])
def get_index():
    file_form = UploadForm()
    params_form = ParamsForm()
    model_output = ''
    val_form = ValidateForm()

    if request.method == 'POST' and file_form.validate_on_submit():
        stream = io.StringIO(file_form.file_path.data.stream.read().decode("UTF8"), newline=None)
        df.df = pd.read_csv(stream)
        df.X = df.df.drop(['price'], axis='columns')
        df.y = df.df['price']
        df.X['date'] = pd.to_datetime(df.X['date']).dt.dayofyear
        df.X = df.X.to_numpy()
        df.y = df.y.to_numpy()
        print(f'Uploaded {df.df.shape} lines')
        return redirect(url_for('get_index'))

    if request.method == 'POST' and params_form.validate_on_submit():
        try:
            print('Set params...')
            print(params_form.model.data)
            n_estimators = int(params_form.p1.data)
            feature_subsample_size = float(params_form.p2.data)
            max_depth = int(params_form.p3.data)
            learning_rate = float(params_form.p4.data)
            if params_form.model == 'Random Forest MSE':
                clf.clf = RandomForestMSE(n_estimators=n_estimators,
                                            feature_subsample_size=feature_subsample_size,
                                            max_depth=max_depth,
                                            )
            else:
                clf.clf = GradientBoostingMSE(n_estimators=n_estimators,
                                            feature_subsample_size=feature_subsample_size,
                                            max_depth=max_depth,
                                            learning_rate=learning_rate)
            return redirect(url_for('get_train'))
            # return render_template('train.html', model_output='')
        except Exception as exc:
            app.logger.info('Exception: {0}'.format(exc))
            print('Exception')
    return render_template('index.html', file_form=file_form, params_form=params_form, val_form=val_form)

@app.route('/train', methods=['POST', 'GET'])
def get_train():
    train_form = TrainForm()
    model_output = ''
    validate_is_available = ''
    if request.method == 'POST' and train_form.validate_on_submit():
        print('Ready for train')
        clf.clf.fit(df.X, df.y)
        print('Trained.')
        clf.trained = True
        y_pred = clf.clf.predict(df.X)
        print('Validated.')
        model_output = mean_squared_error(y_pred, df.y, squared=False)
    if clf.trained:
        validate_is_available = 'Go to validate page'
    else:
        validate_is_available = ''
    return render_template('train.html', train_form=train_form, model_output=model_output, validate_is_available=validate_is_available)

@app.route('/validate', methods=['POST', 'GET'])
def get_validate():
    file_form = ValidateForm()
    get_prediction = ''
    if request.method == 'POST' and file_form.validate_on_submit():
        stream = io.StringIO(file_form.file_path.data.stream.read().decode("UTF8"), newline=None)
        df.val = pd.read_csv(stream)
        df.val['date'] = pd.to_datetime(df.val['date']).dt.dayofyear
        if 'price' in df.val.columns:
            df.val.drop(['price'], axis='columns')
        df.val = df.val.to_numpy()
        y_pred = clf.clf.predict(df.X)
        print('Predicted.')
        submission = pd.DataFrame({'price': y_pred})
        submission.to_csv('submission.csv', index=False)
        get_prediction = 'Get prediction'
        return render_template('validate.html', file_form=file_form, get_prediction=get_prediction)
    return render_template('validate.html', file_form=file_form)

@app.route('/download')
def download():
    return send_file('submission.csv', as_attachment=True)