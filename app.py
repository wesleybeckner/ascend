from sklearn import linear_model
import pandas as pd
import numpy as np
from datetime import timedelta
import dash_core_components as dcc
import dash_daq as daq
import dash_html_components as html
from dash.dependencies import Input, Output
import plotly.graph_objects as go
import dash_bootstrap_components as dbc
import dash_table
import dash
import plotly.express as px
from flask import Flask

def predict_next_fail(n, T, beta):
    lamb = n / (T**beta)
    lamb_hat = lamb * beta * T ** (beta-1)
    TTF = 1 / lamb_hat
    return TTF

def wrap_calc_beta(site):
    beta = []
    T0 = site.loc[0, 'Date'] 
    cumTTF = []
    failN = []
    deltT0 = site.loc[0, 'Cum Days']
    ind = 0
    for n, T in enumerate(site['Date']):
        cumTTF.append((T-T0).days+deltT0)
        failN.append(n+1)
        if ind == 0:
            beta.append(np.nan)
        else:
            beta.append(calc_beta(n, (T-T0).days+deltT0, cumTTF))
        ind += 1
    result = site[['Date', 'Equipment/Process', 
                   'Failure Mode', 'Contributing Factor', 'Klbs']].join(
                    pd.DataFrame.from_dict(({'Beta': np.round(beta,3), 
                    'Cumulative Days': cumTTF, 'Failure Number': failN})))
    result.Klbs = result.Klbs.round(2)
    result = result.fillna(value='None')
    return result

def make_crow_data(site_val = 'CHB', 
                 fail_type='All'):
    site = sites.loc[sites['Plant'].str.contains(site_val)]
    site = site.reset_index(drop=True)
    if fail_type == 'All':
        pass
    else:
        site = site.loc[site['Equipment/Process'] == fail_type].reset_index()
    result = wrap_calc_beta(site)
    return result

def make_scatter(result, y, color='Equipment/Process', log=False):
    fig = px.scatter(result, x='Cumulative Days', y=y, color=color, size='Klbs',
                    log_x=log, log_y=log)
    fig.update_layout(
                    height= 400)
    return fig

def calc_beta(n, T, cumTTF):
    return n / (n * np.log(T) - np.sum(np.log(cumTTF[:n])))

def find_peaks(result):
    # find peaks, 4-padded
    betas = result['Beta'].values[1:]
    peaks = np.zeros(len(betas)+1)
    peaks[5:-4][((betas[4:-4] - betas[3:-5] > 0) & 
     (betas[4:-4] - betas[2:-6] > 0) &
     (betas[4:-4] - betas[1:-7] > 0) &
     (betas[4:-4] - betas[0:-8] > 0) &
     (betas[4:-4] - betas[5:-3] > 0) & 
     (betas[4:-4] - betas[6:-2] > 0) &
     (betas[4:-4] - betas[7:-1] > 0) &
     (betas[4:-4] - betas[8:] > 0)) |
    ((betas[4:-4] - betas[3:-5] < 0) & 
     (betas[4:-4] - betas[2:-6] < 0) &
     (betas[4:-4] - betas[1:-7] < 0) &
     (betas[4:-4] - betas[0:-8] < 0) &
     (betas[4:-4] - betas[5:-3] < 0) & 
     (betas[4:-4] - betas[6:-2] < 0) &
     (betas[4:-4] - betas[7:-1] < 0) &
     (betas[4:-4] - betas[8:] < 0)) ] = 1
    peaks = np.argwhere(peaks).flatten()
    return peaks

def make_scatter_with_regr(result, peaks):
    # plot len(peaks) + 1 linear models
    log = False
    fig = px.scatter(result, x='Cumulative Days', y='Failure Number', color='Peak', size='Klbs',
                        log_x=log, log_y=log)
    fig.update_layout(
                    height= 400)

    for i in range(len(peaks)+1):
        # define range, fit model

        model = linear_model.LinearRegression()
        if i == 0: # first seg
            segment = peaks[i]
            x = result.iloc[0:segment]['Cumulative Days'].values
            y = result.iloc[0:segment]['Failure Number'].values
        elif i == len(peaks): # last seg
            segment = peaks[i-1]
            x = result.iloc[segment:]['Cumulative Days'].values
            y = result.iloc[segment:]['Failure Number'].values
        else: # middle segs
            segment = peaks[i]
            x = result.iloc[old_segment:segment]['Cumulative Days'].values
            y = result.iloc[old_segment:segment]['Failure Number'].values
        model.fit(x.reshape(-1,1),y)

        # grab coefs, make line
        m = model.coef_[0]
        b = model.intercept_
        regrx = np.linspace(x[0], x[-1])
        regry = regrx*m+b

        old_segment = segment
        fig.add_trace(go.Scatter(x=regrx, y=regry,
                        mode='lines',
                        name=f'slope: {m:.3f}',
                            ))
    return fig

# Load Data, Init Variables
sites = pd.read_csv("crowamsaa_data.csv", parse_dates=['Date'])
external_stylesheets = ['https://codepen.io/chriddyp/pen/bWLwgP.css']
site_vals = ['CHB', 'DEC', 'PEN']
fail_types = ['Equipment', 'Process', 'Equipment (Design)', 'All']
result = make_crow_data('CHB', 'All')
TTF = predict_next_fail(*result.loc[result.shape[0]-1, 
                        ['Failure Number', 'Cumulative Days', 'Beta']].values)
next_fail = result.iloc[-1,0] + timedelta(days=TTF)
result.Date = pd.DatetimeIndex(result.Date).strftime("%Y-%m-%d")
peaks = find_peaks(result)
result['Peak'] = False
result.iloc[peaks, -1] = True

# Build App
server = Flask(__name__)
app = dash.Dash(server=server, external_stylesheets=external_stylesheets)

#app = dash.Dash(__name__, external_stylesheets=external_stylesheets)
#server = app.server

app.layout = html.Div([
    html.H1("Crow-AMSAA"),
    html.H2(f"Next Failure: {next_fail.date()}", id='fail_predict'),
    html.Div([
        html.Div([
            html.P('Site'),
            dcc.Dropdown(id='site_dropdown',
                         options=[{'label': i, 'value': i} for i in site_vals],
                         value=site_vals[0],
                         multi=False,
                         className="dcc_control"),      
            html.P('Fail Reason'),
            dcc.Dropdown(id='reason_dropdown',
                         options=[{'label': i, 'value': i} for i in fail_types],
                         value=fail_types[-1],
                         multi=False,
                         className="dcc_control"),   
            html.P('Auto Regress'),
            dcc.Dropdown(id='regress',
                         options=[{'label': i, 'value': i} for i in ['True', 'False']],
                         value='False',
                         multi=False,
                         className="dcc_control"),   
            dash_table.DataTable(
                        id='table',
                        columns=[{"name": i, "id": i} for i in result.columns],
                        data=result.to_dict('records'),
                        page_size=20,
                        style_table={'maxWidth': '100%',
                                     'overflowX': 'auto'}
                        )
                ],  
                className='mini_container',
                id='controls-block',
                style={'width': '32%', 'display': 'inline-block'}
                ),  
            
        html.Div([
                    dcc.Graph(id='plot_1',
                    figure=make_scatter(result, 'Beta', log=False)),
                    dcc.Graph(id='plot_2',
                    figure=make_scatter(result, 'Failure Number', log=False)),
                ], 
                className='mini_container',
                style={'width': '65%', 'float': 'right', 'display': 'inline-block'},
                id='plot-block'
                ),
            ], 
            className='row container-display',
            style={'height': '200px'}
            ),
        ], #className='pretty container',
           style={'width': '100%', 'float': 'center', 'display': 'inline-block'}
        )

@app.callback(
    [Output('plot_1', 'figure'),
     Output('plot_2', 'figure'),
     Output('table', 'data'),
     Output('fail_predict', 'children')],
    [Input('site_dropdown', 'value'),
     Input('reason_dropdown', 'value'),
    Input('regress', 'value')])
def update_plot(site, reason, regress):
    result = make_crow_data(site, reason)
    TTF = predict_next_fail(*result.loc[result.shape[0]-1, 
                        ['Failure Number', 'Cumulative Days', 'Beta']].values)
    next_fail = result.iloc[-1,0] + timedelta(days=TTF)
    result.Date = pd.DatetimeIndex(result.Date).strftime("%Y-%m-%d")
    if regress == 'True':
        peaks = find_peaks(result)
        result['Peak'] = False
        result.iloc[peaks, -1] = True
        return [
            make_scatter(result, 'Beta', 'Peak', False),
            make_scatter_with_regr(result, peaks),
            result.to_dict('records'),
           f"Next Failure: {next_fail.date()}"
        ]
    else:
        return [make_scatter(result, 'Beta', log=False),
                make_scatter(result, 'Failure Number', log=False),
               result.to_dict('records'),
               f"Next Failure: {next_fail.date()}"]

if __name__ == '__main__':
    app.run_server()
#    server.run(debug=False)
