# DASH components
import dash
import dash_core_components as dcc
import dash_html_components as html
import dash_table
from dash.dependencies import Input, Output, State
from dash.exceptions import PreventUpdate

# plotly graphs
import plotly.graph_objects as go

# DB components
import pyodbc

# Data components
import pandas as pd 

# Loading components
import base64
import io

# Config
import config

# Colors

colors = ['#800000', '#DC143C', '#FFA07A', '#FF8C00', '#FFD700', '#DAA520', '#808000', '#9ACD32',
		  '#556B2F', '#7CFC00', '#006400', '#2E8B57', '#20B2AA', '#00CED1', '#5F9EA0', '#6495ED',
		  '#1E90FF', '#000080', '#4169E1', '#8A2BE2', '#8B008B', '#A0522D', '#D2691E', '#778899']


# Data
report_data = None


class DBDataLoader():
	def __init__(self):
		pass

	def make_conn(self):
		conn_statement = "DRIVER={};SERVER={};PORT={};DATABASE={};UID={};PWD={}".format(
				config.driver,
				config.server_ip,
				config.port,
				config.db_name,
				config.user_name,
				config.user_password
			)

		conn = pyodbc.connect(conn_statement, timeout = config.connection_timeout)
		return conn

	def load_data(self, query):
		conn = self.make_conn()
		try:
			cursor = conn.cursor()
			cursor.execute(query)
			data = cursor.fetchall()
			columns = [c[0] for c in cursor.description]
			data = [list(r) for r in data]
			return pd.DataFrame(data, columns=columns)

		except Exception as e:
			raise e 
		finally:
			conn.close()

class LocDataLoader():
	def __init__(self):
		pass

	def parse_content(self, contents, filename):
		content_type, content_string = contents.split(',')
		decoded = base64.b64decode(content_string)
		print('TYPE DECODED: ', type(decoded))

		data = None
		df = None
		status_msg = ''
		print('IN PARSE CONTENT')
		try:
			if 'xlsx' in filename:
				print('XLSX')
				df = self.excel_report_loading(data_decoded=decoded, sheet_name='FULL_INFO')
		except Exception as e:
			status_msg = 'Error during data loading!'
			raise e
		"""
		if df is not None:
			data = dash_table.DataTable(
				data=df.to_dict('records'),
				columns=[{'name': col, 'id': col} for col in df.columns],
				fixed_rows={ 'headers': True, 'data': 0 },
				style_table={'overflowX': 'scroll', 'maxHeight': '300px'},
				style_cell={'width': '150px'}
			)
		"""
		return df

	def excel_report_loading(self, data_decoded, sheet_name='SHORT_INFO', convert_dates = False):
		print('IN REPORT LOADING')
		df = None
		if sheet_name == 'SHORT_INFO':
			print('SHORT')
			df = pd.read_excel(io.BytesIO(data_decoded), sheet_name='SHORT_INFO')
			renaming_dict = {
			    'Дата начала диапазона зафиксированного изменения конкурента':'TRAVEL_START',
			    'Дата конца диапазона зафиксированного изменения конкурента':'TRAVEL_END',
			    'Номер периода, для которого дается рекомендация':'PERIOD_NUM',
			    'Направление перевозки':'DIR',
			    'Конкурент':'AIRLINE',
			    'Нормированная доля рынка конкурента':'NORMED_SHARE',
			    'Код минимального тарифа на дату агрегации':'AGG_FARE_CODE',
			    'Баз. стоимость тарифа конкурента на дату агрегации в валюте публ.':'AGG_BASE_PRICE',
			    'Валюта публикации тарифа конкурента':'AGG_CURRENCY',
			    'Полная стоимость мин. тарифа конкурента на дату агрегации (USD)':'AGG_TOTAL_PRICE_USD',
			    'Дата начала действия мин. тарифа конкурента на дату агрегации':'AGG_EFFECTIVE_DATE',
			    'Дата окончания действия мин. тарифа конкурента на дату агрегации':'AGG_DISCONTINUE_DATE',
			    'Дата начала перевозки по мин. тарифу конкурента на дату агрегации':'AGG_TRAVEL_FROM',
			    'Дата окончания перевозки по мин. тарифу конкурента на дату агрегации':'AGG_TRAVEL_TO',
			    'Код минимального тарифа на дату прогноза':'PRED_FARE_CODE',
			    'Баз. стоимость тарифа конкурента на дату прогноза в валюте публ.':'PRED_BASE_PRICE',
			    'Валюта публикации тарифа конкурента.1':'PRED_CURRENCY',
			    'Полная стоимость минимального тарифа конкурента на дату прогноза (USD)':'PRED_TOTAL_PRICE_USD',
			    'Дата начала действия минимального тарифа конкурента на дату прогноза':'PRED_EFFECTIVE_DATE',
			    'Дата окончания действия минимального тарифа конкурента на дату прогноза':'PRED_DISCONTINUE_DATE',
			    'Дата начала перевозки по минимальному тарифу конкурента на дату прогноза':'PRED_TRAVEL_FROM',
			    'Дата окончания перевозки по минимальному тарифу конкурента на дату прогноза':'PRED_TRAVEL_TO',
			    'Изменение стоимости минимального тарифа конкурента на дату перевозки (USD)':'RIVAL_PRICE_CHANGE',
			    'Изменение полной стоимости тарифа АФЛ, USD':'SU_TOTAL_CHANGE_USD',
			    'Изменение стоимости минимального тарифа конкурента относительно АФЛ на дату перевозки (USD)':'RIVAL_SU_CHANGE',
			    'Нормированное изменение стоимости тарифа конкурента':'NORMED_CHANGE',
			    'Код минимального тарифа АФЛ, для которого рассчитывается рекомендация':'SU_FARE_CODE',
			    'Базовая стоимость минимального тарифа АФЛ, для которого рассчитывается рекомендация':'SU_BASE_PRICE',
			    'Полная стоимость минимального тарифа АФЛ (USD)':'SU_TOTAL_PRICE_USD',
			    'Доступность минимального тарифа АФЛ согласно Aviasales на дату перевозки':'SU_AVAILABILITY',
			    'Дата последнего зафиксированного изменения полной стоимости минимального тарифа АФЛ':'SU_LAST_CHANGE_DATE',
			    'Дата начала агрегации изменений конкурентов':'AGG_START_DATE',
			    'Дата последнего изменения минимального тарифа АФЛ':'LAST_SU_PRICE_CHANGE',
			    'Общее изменение стоимости минимальных тарифов конкурентов на дату перевозки (USD)':'TOTAL_CHANGE_USD',
			    'Общее изменение стоимости минимальных тарифов конкурентов на дату перевозки в валюте тарифов':'TOTAL_CHANGE',
			    'Рекомендованное изменение стоимости тарифа АФЛ на период с учетом условий агрегации изменений по периоду':'SU_REC',
			    'Рекомендованное изменение стоимости тарифа АФЛ на период с учетом доступности АФЛ':'SU_REC_AVAIL',
			    'Тренд по выручке согласно прогнозной модели':'TREND_VALUE',
			    'Рекомендация, совмещенная с трендом':'REC_TREND_VALUE',
			    'Рекомендованная базовая стоимость минимального тарифа АФЛ':'SU_REC_BASE_PRICE'
			}
			df.rename(columns=renaming_dict, inplace=True)
			if convert_dates:
				dates_columns = ['TRAVEL_START', 'TRAVEL_END', 'AGG_EFFECTIVE_DATE', 
                 'AGG_DISCONTINUE_DATE', 'AGG_TRAVEL_FROM', 'AGG_TRAVEL_TO',
                 'PRED_EFFECTIVE_DATE', 'PRED_DISCONTINUE_DATE', 'PRED_TRAVEL_FROM', 'PRED_TRAVEL_TO',
                 'SU_LAST_CHANGE_DATE', 'AGG_START_DATE', 'LAST_SU_PRICE_CHANGE']
				for d in dates_columns:
				    df[d] = pd.to_datetime(df[d])
		elif sheet_name == 'FULL_INFO':
			df = pd.read_excel(io.BytesIO(data_decoded),  sheet_name='FULL_INFO')
			df['FLIGHT_DATE'] = df['FLIGHT_DATE'].dt.strftime('%Y-%m-%d')
		
		return df




app = dash.Dash(__name__)
app.layout = html.Div(children=[
		dcc.Tabs(
			className='custom-tabs',
			value='rec-vis', 
			children=[
				dcc.Tab(className='custom-tab', 
					selected_className='custom-tab--selected', 
					label='Recommendation Visualizer', 
					value='rec-vis', 
					children=[
						html.Div(children=[
								dcc.Upload(id='upload-rec-data', 
									children=html.Div(className='dragndrop', children=[
										'Drag and Drop or ', 
										html.A('Select Report')
										]
									),
									multiple=False
								)
							]
						),
						html.Div(children=[
								html.Label(id='load-status', children=[''])
							],
							style={'width':'100%', 'textAlign':'center', 'fontSize':'250%'}
						),
						html.Div(id='output-data-upload'),
						dcc.Loading(
							html.Div(id='rec-container', children=[
								html.Div(children=[
									html.Label(className='custom-label', children=['Choose direction:']),
									dcc.Dropdown(
									id='dirs-dropdown',
									options=[],
									value=[],
									style={'width':'200px'}
									)
								]),
								html.Div(id='graph-container', children=[
									html.Div(id='rec-graph-container', children=[
										html.Div(children=[
											dcc.Graph(id='rec-graph',
												figure={'data':None, 
												'layout':dict(
												xaxis={'title': 'Travel Date'},
								                yaxis={'title': 'Total Price, USD'},
								                title='Minimal Fares'
											)}
										)], style={'width':'100%', 'display':'inline-block'})
									]),

									html.Div(id='change-graph-container', children=[
										html.Div([
											dcc.Graph(id='rec-changes-graph',
												figure={'data':None, 
												'layout':dict(
												xaxis={'title': 'Travel Date'},
								                yaxis={'title': 'Total Price Change, USD'},
								                title='Minimal Fares Changes'
											)}
										)], style={'width':'100%', 'display':'inline-block'})
									])
								])
							])
						),
						html.Div(id='intermediate-data', style={'display':'none'})
					]
				),
				dcc.Tab(className='custom-tab', selected_className='custom-tab--selected', label='History Visualizer', value='hist-vis', children=[

					]
				)
			]
		)
	]
)

@app.callback(
	[Output('dirs-dropdown', 'options'),
	 Output('rec-container', 'n_clicks')],
	[Input('upload-rec-data', 'contents')],
	[State('upload-rec-data', 'filename')]
)
def show_loaded_recs(contents, filename):
	if not all([contents, filename]):
		raise PreventUpdate

	ldl = LocDataLoader()
	data=ldl.parse_content(contents, filename)

	# Saving to global variable
	global report_data
	report_data = data

	return [{'label': d, 'value': d} for d in data.DIR.unique()], 0


@app.callback(
	[Output('rec-graph', 'figure'),
	 Output('rec-changes-graph', 'figure')],
	[Input('dirs-dropdown', 'value')],
	[State('intermediate-data', 'children')]
)
def update_rec_graph(chosen_dir, data_json):
	if not chosen_dir:
		raise PreventUpdate
	data = report_data[report_data.DIR == chosen_dir].copy()

	data['FLIGHT_DATE'] = pd.to_datetime(data['FLIGHT_DATE'])
	data.sort_values(by='FLIGHT_DATE', inplace=True)

	traces_rec = []
	traces_change = []

	# min max values
	price_columns = ['AF_PRICE_TOTAL_USD_LC', 'AF_PRICE_TOTAL_USD_P', 'SU_AF_PRICE_TOTAL_USD']

	def increase_range(value, border='low'):
		if (value > 0 and border == 'low') or (value < 0 and border == 'high'): return value * 0.9
		else: return value * 1.1


	rec_max = max(data[price_columns].max())
	rec_max = increase_range(rec_max, 'high')

	rec_min = min(data[price_columns].min())
	rec_min = increase_range(rec_min, 'low')

	change_columns = ['RIVAL_PRICE_CHANGE_USD', 'SU_PRICE_CHANGE_USD', 'TOTAL_NORMED_CHANGE_USD']
	change_max = max(data[change_columns].max())
	change_max = increase_range(change_max, 'high')

	change_min = min(data[change_columns].min())
	change_min = increase_range(change_min, 'low')

	# periods
	periods_list = data.PERIOD_NUM.unique()

	for p in periods_list:
		data_p = data[data.PERIOD_NUM == p]
		act = data_p.ACT.values[0]

		if act == -1:
			color = '#ffc6e5'
		elif act == 1:
			color = '#b2ffa8'
		else:
			color = '#f4f2f5'

		traces_rec.append(go.Scatter(
							x=data_p.FLIGHT_DATE, 
							y = [rec_max for d in data_p.FLIGHT_DATE], 
							fill='tozeroy', 
							line_color =color,
							hoverinfo = 'text',
							hovertext = 'PERIOD: {}'.format(p),
							name = None,
							fillcolor=color,
							showlegend=False))

		traces_change.append(go.Scatter(
							x=data_p.FLIGHT_DATE, 
							y = [change_max for d in data_p.FLIGHT_DATE], 
							fill='tozeroy', 
							line_color =color,
							hoverinfo = 'text',
							hovertext = 'PERIOD: {}'.format(p),
							name = None,
							fillcolor=color,
							showlegend=False))
		traces_change.append(go.Scatter(
							x=data_p.FLIGHT_DATE, 
							y = [change_min for d in data_p.FLIGHT_DATE], 
							fill='tozeroy', 
							line_color =color,
							hoverinfo = 'text',
							hovertext = 'PERIOD: {}'.format(p),
							name = None,
							fillcolor=color,
							showlegend=False))

	# Rivals plots
	for i, a in enumerate(data.AIRLINE.unique()):
		airline_data = data[data.AIRLINE == a]
		# normed share for one airline is constant
		normed_share = airline_data.NORMED_SHARE.values[0]
		color = colors[i]
		print('AIRLINE:{}, COLOR:{}'.format(a, color))
		for f_lc in airline_data.FARE_CLASS_CODE_LC.unique():
			fares_lc_data = airline_data[airline_data.FARE_CLASS_CODE_LC == f_lc]

			traces_rec.append(
				dict(
						x=fares_lc_data.FLIGHT_DATE,
						y=fares_lc_data.AF_PRICE_TOTAL_USD_LC,
						name=a + ' before',
						hovertext=f_lc,
						line=dict(dash='dot', width=2,  color=color),
						marker=dict(symbol='circle'),
						opacity=0.5,
						mode='lines+markers',
						legendgroup=a
					)
			)
		for f_pred in airline_data.FARE_CLASS_CODE_P.unique():
			fares_pred_data = airline_data[airline_data.FARE_CLASS_CODE_P == f_pred]

			traces_rec.append(
				dict(
						x=fares_pred_data.FLIGHT_DATE,
						y=fares_pred_data.AF_PRICE_TOTAL_USD_P,
						name=a + ' after',
						hovertext=f_pred + '/{:.2f}'.format(normed_share),
						line=dict(width=2, color=color),
						marker=dict(symbol='cross'),
						mode='lines+markers',
						legendgroup=a
					)
			)

		traces_change.append(
			dict(
				x=airline_data.FLIGHT_DATE,
				y=airline_data.RIVAL_PRICE_CHANGE_USD,
				name=a,
				line=dict(color=color),
				mode='lines+markers',
				marker=dict(symbol='square'),
				legendgroup=a
			)
		)

	# SU plots
	su_data = data[['FLIGHT_DATE', 
					'SU_FARE_CLASS_CODE', 'SU_AF_PRICE_TOTAL_USD', 
					'SU_PRICE_CHANGE_USD', 'SU_AF_FIRST_PRICE_TOTAL_USD']].drop_duplicates()
	for su_fare in su_data.SU_FARE_CLASS_CODE.unique():
		su_fare_data = su_data[su_data.SU_FARE_CLASS_CODE == su_fare]
		traces_rec.append(
			dict(
				x=su_fare_data.FLIGHT_DATE,
				y=su_fare_data.SU_AF_PRICE_TOTAL_USD,
				name='SU after',
				line=dict(width=2, color='black'),
				marker=dict(symbol='cross'),
				mode='lines+markers',
				legendgroup='SU'
			)
		)

		traces_rec.append(
			dict(
				x=su_fare_data.FLIGHT_DATE,
				y=su_fare_data.SU_AF_FIRST_PRICE_TOTAL_USD,
				name='SU before',
				line=dict(width=2, color='black'),
				marker=dict(symbol='circle'),
				mode='lines+markers',
				legendgroup='SU'
			)
		)

		traces_change.append(
			dict(
				x=su_fare_data.FLIGHT_DATE,
				y=su_fare_data.SU_PRICE_CHANGE_USD,
				name='SU',
				line=dict(color='black'),
				mode='lines+markers',
				marker=dict(symbol='square'),
				legendgroup='SU'
			)
		)

	rec_data = data[['FLIGHT_DATE', 'TOTAL_NORMED_CHANGE_USD']].drop_duplicates()
	traces_change.append(
		dict(
			x=rec_data.FLIGHT_DATE,
			y=rec_data.TOTAL_NORMED_CHANGE_USD,
			name='Recomm',
			line=dict(dash='dot', color='black'),
			marker=dict(symbol='x'),
			mode='lines+markers'
		)
	)



	fig_rec = {
	'data':traces_rec,
	'layout': dict(
			xaxis={'title': 'Travel Date'},
            yaxis={'title': 'Total Price, USD', 'range':[rec_min, rec_max]},
            title='Minimal Fares',
		)
	}

	fig_change = {
	'data':traces_change,
	'layout': dict(
			xaxis={'title': 'Travel Date'},
            yaxis={'title': 'Total Price Change, USD', 'range':[change_min, change_max]},
            title='Minimal Fares Changes'
		)
	}

	period_marks = {m:str(m) for m in periods_list}
	print('PERIOD MARKS', period_marks)


	return fig_rec, fig_change


if __name__ == '__main__':
	app.run_server(debug=True)
			
