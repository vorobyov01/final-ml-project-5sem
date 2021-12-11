import io
import pandas as pd
import numpy as np

from sklearn.ensemble import RandomForestRegressor # <- needs to be replaced
from sklearn.metrics import mean_squared_error # <- needs to be replaced

from collections import namedtuple
from flask_wtf import FlaskForm
from flask_bootstrap import Bootstrap
from flask import Flask, request, url_for
from flask import render_template, redirect

from flask_wtf.file import FileAllowed
from wtforms.validators import DataRequired
from wtforms import StringField, SubmitField, FileField

app = Flask(__name__, template_folder='templates')
app.config['BOOTSTRAP_SERVE_LOCAL'] = True
app.config['SECRET_KEY'] = 'sergei'
Bootstrap(app)

class TextForm(FlaskForm):
    text = StringField('Text', validators=[DataRequired()])
    submit = SubmitField('Get Result')


class Response(FlaskForm):
    score = StringField('Score', validators=[DataRequired()])
    sentiment = StringField('Sentiment', validators=[DataRequired()])
    submit = SubmitField('Try Again')


class Score(FlaskForm):
    score = StringField('Score', validators=[DataRequired()])
    submit = SubmitField('Try Again')


class FileForm(FlaskForm):
    file_path = FileField('Choose dataset', validators=[
        DataRequired('Specify file'),
        FileAllowed(['csv'], 'CSV only!')
    ])
    submit = SubmitField('Send File')

class Dataset():
    df = None
df = Dataset()

@app.route('/', methods=['POST', 'GET'])
def get_index():
    file_form = FileForm()
    score_form = Score()

    if request.method == 'POST' and file_form.validate_on_submit():
        stream = io.StringIO(file_form.file_path.data.stream.read().decode("UTF8"), newline=None)
        df.df = pd.read_csv(stream)
        print(f'Uploaded {df.df.shape} lines')
        return redirect(url_for('get_index'))

    if request.method == 'POST' and score_form.validate_on_submit():
        try:
            print('fiting...')
            clf = RandomForestRegressor()
            X, y = df.df.drop(['price', 'date'], axis='columns'), df.df['price']
            clf.fit(X, y)
            y_pred = clf.predict(X)
            score_form.score.data = mean_squared_error(y_pred, y, squared=False)
            print('validate...')
            return render_template('index.html', file_form=file_form, score_form=score_form)
        except Exception as exc:
            app.logger.info('Exception: {0}'.format(exc))
            print('Exception')
    return render_template('index.html', file_form=file_form, score_form=score_form)