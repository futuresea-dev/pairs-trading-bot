import MetaTrader5 as mt5
import pytz
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import statsmodels.api as sm
import warnings
import time
import colorama
from pylab import rcParams
from colorama import Fore, Back
from datetime import datetime, timedelta, date
from statsmodels.tsa.stattools import coint
from statsmodels.tsa.stattools import adfuller
from statsmodels.tsa.stattools import kpss
from sklearn import linear_model
from sklearn.metrics import mean_absolute_error, mean_squared_error

rcParams['figure.figsize'] = 15, 5
warnings.filterwarnings("ignore")
colorama.init(autoreset=False)

Path = "C:/Program Files/FBS MetaTrader 5/terminal64.exe"
Conta = 0000000000
Senha = "XXXXXXXXX"
Server = "FBS-Demo"

if not mt5.initialize(Path, login=Conta, server=Server, password=Senha, portable=False, timeout=10000):
    print("initialize() failed, error code =", mt5.last_error())
    mt5.shutdown()
    quit()

#account_info_dict = mt5.account_info()._asdict()
#print("Login efetuado com sucesso!!!")
#print("Saldo na conta " + str(account_info_dict["balance"]))

beta_fit = 0
alpha_fit = 0

# Função para avaliar estacionaridade de uma série temporal
def avalia_estacionaridade(X, cutoff=0.01):
    # Ho: teste com raiz unitária (nao - estacionária)
    pvalue = adfuller(X)[1]
    if pvalue < cutoff:
        print('p-value = ' + str(pvalue) + ' A série ' + X.name + ' é estacionária.')
        return True
    else:
        print('p-value = ' + str(pvalue) + ' A série ' + X.name + ' não é estacionária.')
        return False

# função para plotar pares
def plotPares(d2, par):
    (d2[par[0]] / np.mean(d2[par[0]])).plot()
    (d2[par[1]] / np.mean(d2[par[1]])).plot()
    plt.legend(par)
    plt.show()

# O ratio não é muito indicado para avaliar as possiveis entradas nos trades:
def zscore(series):
    return (series - series.mean()) / np.std(series)

def desenhaRatio(d2, par):
    data = d2
    S1 = data[par[0]]
    S2 = data[par[1]]

    score, pvalue, _ = coint(S1, S2)
    print('P-valor = ', pvalue)
    ratios = S1 / S2

    zscore(ratios).plot(figsize=(15, 7))

    plt.axhline(zscore(ratios).mean(), color='black')
    plt.axhline(2.0, color='red', linestyle='--')
    plt.axhline(-2.0, color='green', linestyle='--')
    plt.axhline(1.0, color='orange', linestyle='--')
    plt.axhline(-1.0, color='orange', linestyle='--')
    plt.legend(['Ratio z-score', 'Mean', '+2', '-2'])
    plt.show()

def regr(x,y):
    regr = linear_model.LinearRegression()
    x_constant = pd.concat([x,pd.Series([1]*len(x),index = x.index)], axis=1)
    regr.fit(x_constant, y)
    beta = regr.coef_[0]
    alpha = regr.intercept_
    spread = y - (x*beta - alpha)
    acuracia = regr.score(x_constant, y)
    return spread, beta, alpha, acuracia

def mean_norm(df_input):
    return df_input.apply(lambda x: (x - x.mean()) / x.std(), axis=0)

def normalize_and_accumulate_series(data):
    return data.pct_change().cumsum()

# Estabelecemos a conexão ao MetaTrader 5
if not mt5.initialize():
    print("initialize() failed, error code =", mt5.last_error())
    quit()

DIAS = [
    'Segunda-Feira',
    'Terca-Feira',
    'Quarta-Feira',
    'Quinta-Feira',
    'Sexta-Feira',
    'Sabado',
    'Domingo'
]

data_atual = date.today()
dia = int(data_atual.strftime('%d'))
mes = int(data_atual.strftime('%m'))
ano = int(data_atual.strftime('%Y'))

for periodo in range(4):
    if periodo == 0:
        d = date(ano, mes, dia)
        d = d - timedelta(days=90)
        dias = 90

    if periodo == 1:
        d = date(ano, mes, dia)
        d = d - timedelta(days=100)
        dias = 100

    if periodo == 2:
        d = date(ano, mes, dia)
        d = d - timedelta(days=110)
        dias = 110

    if periodo == 3:
        d = date(ano, mes, dia)
        d = d - timedelta(days=120)
        dias = 120

    data_atual = date.today()
    dia = int(data_atual.strftime('%d'))
    mes = int(data_atual.strftime('%m'))
    ano = int(data_atual.strftime('%Y'))

    novo_ano = d.year
    novo_mes = d.month
    novo_dia = d.day

    print(2 * '\n' + 'PARES CORRELACIONADOS E COINTEGRADOS COM -', '\033[1;37;41m' + str(dias), 'PERIODOS' + '\33[m')
    print('_________________________________________________________________________________________')

    data = date(year=ano, month=mes, day=dia)
    indice_da_semana = data.weekday()
    dia_da_semana = DIAS[indice_da_semana]

    relogio = time.localtime()
    hora = relogio.tm_hour
    minuto = relogio.tm_min

    # definimos o fuso horário como UTC
    timezone = pytz.timezone("ETC/UTC")
    utc_from = datetime(novo_ano, novo_mes, novo_dia, tzinfo=timezone)
    utc_to = datetime(ano, mes, dia, tzinfo=timezone)

    datahoje = datetime.now(pytz.timezone('ETC/GMT-3'))
    data_agora = datahoje.strftime("%Y-%m-%d-%H:%M:%S")
    datahoje_pregao = datahoje.strftime("%Y-%m-%d") + "-00:00:00"
    diff_inicio_pregao = pd.Timestamp(data_agora) - pd.Timestamp(datahoje_pregao)
    minutos = 5
    quant_barras = int(diff_inicio_pregao.total_seconds() / (minutos * 60))    

    #if (dia_da_semana == 'Sexta-Feira' and hora >= 18) or (dia_da_semana == 'Sabado') or (
            #dia_da_semana == 'Domingo' and hora < 18):
    if (dia_da_semana == 'Sexta-Feira' and hora >= 18) or (dia_da_semana == 'Domingo' and hora < 18):
        print(">>> MERCADO FECHADO - FORA DE HORARIO PARA OPERACOES !!!")
        time.sleep(300)
    else:
        Path = "C:/Program Files/FBS MetaTrader 5/terminal64.exe"
        Conta = 0000000000
        Senha = "XXXXXXXXX"
        Server = "FBS-Demo"

        if not mt5.initialize(Path, login=Conta, server=Server, password=Senha, portable=False, timeout=10000):
            mt5.shutdown()

        pares_ativo1 = ('EURUSD', 'USDJPY', 'GBPUSD', 'AUDUSD', 'NZDUSD', 'EURJPY',
                        'GBPJPY', 'EURGBP', 'EURCAD', 'EURCHF', 'AUDCAD', 'AUDCHF',
                        'AUDJPY', 'AUDNZD', 'CADCHF', 'CADJPY', 'CHFJPY', 'EURAUD',
                        'EURNZD', 'GBPAUD', 'GBPCAD', 'GBPCHF', 'GBPNZD', 'NZDCHF',
                        'NZDJPY', 'USDCAD', 'USDCHF', 'NZDCAD')                        

        pares_ativo2 = ('EURUSD', 'USDJPY', 'GBPUSD', 'AUDUSD', 'NZDUSD', 'EURJPY',
                        'GBPJPY', 'EURGBP', 'EURCAD', 'EURCHF', 'AUDCAD', 'AUDCHF',
                        'AUDJPY', 'AUDNZD', 'CADCHF', 'CADJPY', 'CHFJPY', 'EURAUD',
                        'EURNZD', 'GBPAUD', 'GBPCAD', 'GBPCHF', 'GBPNZD', 'NZDCHF',
                        'NZDJPY', 'USDCAD', 'USDCHF', 'NZDCAD')                      

        for info_ativo1 in pares_ativo1:
            ativo1 = info_ativo1
            for infor_ativo2 in pares_ativo2:
                ativo2 = infor_ativo2

                # Extraindo informações do PERIODO
                rates = mt5.copy_rates_range(ativo1, mt5.TIMEFRAME_M5, utc_from, utc_to)
                portfolio = pd.DataFrame(rates)

                if portfolio.empty:
                    print('PROBLEMA DE COMUNICACAO - DataFrame está vazio')
                    time.sleep(300)
                else:
                    try:
                        portfolio['time'] = pd.to_datetime(portfolio['time'], unit='s')
                        portfolio.drop(['open', 'high', 'low', 'tick_volume', 'spread', 'real_volume'], axis=1,
                                       inplace=True)
                        portfolio.rename(columns={'time': 'Date', 'close': ativo1}, inplace=True)

                        # Extraindo informações das BARRAS
                        barras = mt5.copy_rates_from_pos(ativo1, mt5.TIMEFRAME_M5, 0, quant_barras)
                        barras_frame = pd.DataFrame(barras)
                        barras_frame['time'] = pd.to_datetime(barras_frame['time'], unit='s')
                        barras_frame.drop(['open', 'high', 'low', 'tick_volume', 'spread', 'real_volume'], axis=1,
                                          inplace=True)
                        barras_frame.rename(columns={'time': 'Date', 'close': ativo1}, inplace=True)
                        portfolio = portfolio.append(barras_frame, ignore_index=True)

                        # Extraindo informações do PERIODO
                        dados = mt5.copy_rates_range(ativo2, mt5.TIMEFRAME_M5, utc_from, utc_to)
                        forex = pd.DataFrame(dados)
                        forex['time'] = pd.to_datetime(forex['time'], unit='s')
                        forex.drop(['time', 'open', 'high', 'low', 'tick_volume', 'spread', 'real_volume'], axis=1,
                                   inplace=True)
                        forex.rename(columns={'close': ativo2}, inplace=True)

                        # Extraindo informações das BARRAS
                        barras = mt5.copy_rates_from_pos(ativo2, mt5.TIMEFRAME_M5, 0, quant_barras)
                        barras_frame = pd.DataFrame(barras)
                        barras_frame['time'] = pd.to_datetime(barras_frame['time'], unit='s')
                        barras_frame.drop(['time', 'open', 'high', 'low', 'tick_volume', 'spread', 'real_volume'],
                                          axis=1,
                                          inplace=True)
                        barras_frame.rename(columns={'close': ativo2}, inplace=True)
                        forex = forex.append(barras_frame, ignore_index=True)
                        portfolio[ativo2] = forex

                        portfolio = portfolio.dropna()
                        portfolio.set_index('Date', inplace=True)
                    except:
                        print("VARIAVEL TIME - Problema ao carregar o DATAFRAME")
                        time.sleep(300)
                    else:
                        S1 = portfolio[ativo1]
                        S2 = portfolio[ativo2]

                        # Verificando estacionaridade dos pares
                        stats, p, lags, critical_values = kpss(S1, 'ct')
                        stats_1, p_1, lags_1, critical_values_1 = kpss(S2, 'ct')

                        # Se os pares não forem estacionarios
                        if ((p < 0.05) and (p_1 < 0.05)):
                            if (ativo1 != 'Date') and (ativo2 != 'Date'):
                                if (ativo1 != ativo2):
                                    correlacao = portfolio[ativo1].corr(portfolio[ativo2])

                                    # Se a correlaçao for maior que 70%
                                    if (correlacao > 0):
                                        cointegracao = coint(S1, S2)
                                        p_value = cointegracao[1]

                                        # if ((p_value < 0.05) and (p_value >= 0.01)):
                                        if (p_value < 0.05):
                                            price = pd.concat([S1, S2], axis=1)
                                            lp = np.log(price)
                                            x = lp[ativo1]
                                            y = lp[ativo2]
                                            spread, beta_fit, alpha_fit, acuracia = regr(x, y)
                                            reta = beta_fit*x+alpha_fit

                                            adf = sm.tsa.stattools.adfuller(spread)

                                            if (adf[1] < 0.05):
                                                scale = pd.DataFrame(spread)
                                                trade = mean_norm(scale)
                                                trade.reset_index(drop=True)
                                                # trade.to_csv('entradas_' + keys[j] + '-' + keys[i] + '.csv')
                                                # print(keys[i], keys[j], correlacao, cointegracao[1], mint)
                                                mint = trade[0].iloc[-1]

                                                if (mint >= 0.950) and (mint <= 1.10):
                                                    print('\033[1;37;42m' + str(ativo1) + '\33[m',
                                                          '\033[1;37;43m' + str(ativo2) + '\33[m',
                                                          ' :: ', round(mint, 10), beta_fit, alpha_fit, acuracia)
                                                    print('___________________________________________________________'
                                                          '______________________________')

                                                    #plt.scatter(x,y,label='y(x)')
                                                    #plt.plot(x,reta,label='Ajuste linear', color='red')
                                                    #plt.xlabel('x')
                                                    #plt.ylabel('y')
                                                    #plt.legend(['Reta da Regressão Linear'])
                                                    #plt.show()

                                                if (mint <= -0.950) and (mint >= -1.10):
                                                    print('\033[1;37;42m' + str(ativo1) + '\33[m',
                                                          '\033[1;37;43m' + str(ativo2) + '\33[m',
                                                          ' :: ', round(mint, 10), beta_fit, alpha_fit, acuracia)
                                                    print('___________________________________________________________'
                                                          '______________________________')
                                                    

                                                if (mint >= 2.05) and (mint <= 2.50):
                                                    print('\033[1;37;42m' + str(ativo1) + '\33[m',
                                                          '\033[1;37;43m' + str(ativo2) + '\33[m',
                                                          ' :: ', round(mint, 10), beta_fit, alpha_fit, acuracia)
                                                    print('___________________________________________________________'
                                                          '______________________________')
                                                    
                                                    
                                                if (mint <= -2.05) and (mint >= -2.50):
                                                    print('\033[1;37;42m' + str(ativo1) + '\33[m',
                                                          '\033[1;37;43m' + str(ativo2) + '\33[m',
                                                          ' :: ', round(mint, 10), beta_fit, alpha_fit, acuracia)
                                                    print('___________________________________________________________'
                                                          '______________________________')
