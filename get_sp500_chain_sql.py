#%%

import pandas as pd
import numpy as np
import yfinance as yf
import bs4 as bs
#import pickle
import requests
import datetime as dt
from datetime import datetime
import os
from tqdm.notebook import tqdm
import csv
import bisect
#from dateutil import parser
#from decimal import Decimal
#import math
import pymysql.cursors
#import pandas_datareader.data as pdr
#import bisect
import time
#from importlib import reload
#import sys
#import pandas_datareader.data as pdr

#%%
mydb = pymysql.connect(
   host="localhost",
   user="root",
   password="F2@Strap",
   database="optionsdatabase",
   #"client_flag": CLIENT.MULTI_STATEMENTS,
)

mycursor = mydb.cursor()


#today = datetime.now()

#Para forçar o dia a actualizar
#today=dt.datetime(2021,6,9,22,8)

#%%
#Não correr
def options_chain(symbol):

    tk = yf.Ticker(symbol)
    # Expiration dates
    
    try:
        exps = tk.options

    # Get options for each expiration
        options = pd.DataFrame()
        for e in exps:
            opt = tk.option_chain(e)
            opt = pd.DataFrame().append(opt.calls).append(opt.puts)
            opt['expirationDate'] = e
            options = options.append(opt, ignore_index=True)

        # Bizarre error in yfinance that gives the wrong expiration date
        # Add 1 day to get the correct expiration date
        #options['expirationDate'] = pd.to_datetime(options['expirationDate']) + datetime.timedelta(days = 1)
        #options['dte'] = (options['expirationDate'] - datetime.datetime.today()).dt.days / 365
        
        # Boolean column if the option is a CALL
        options['CALL'] = options['contractSymbol'].astype(str).str[4:].apply(
            lambda x: "C" in x)
        
        options[['bid', 'ask', 'strike']] = options[['bid', 'ask', 'strike']].apply(pd.to_numeric)
        options['mark'] = (options['bid'] + options['ask']) / 2 # Calculate the midpoint of the bid-ask
        
        # Drop unnecessary and meaningless columns
        options = options.drop(columns = ['contractSize', 'currency', 'change', 'percentChange', 'lastTradeDate', 'lastPrice'])
        
        return options

    except IndexError as e:
        print(f'{symbol}: {e}')  # print the ticker and the error
        options = pd.DataFrame()
        return options
#a=options_chain('AAPL')

#. to_pickle("a_file.pkl")
#output = pd. read_pickle("sp500tickers.pickle")

def save_sp500_tickers():
    html = requests.get('http://en.wikipedia.org/wiki/List_of_S%26P_500_companies')
    soup = bs.BeautifulSoup(html.text, 'lxml')
    table = soup.find('table', {'class': 'wikitable sortable'})
    tickers = []
    names=[]
    sectors=[]
    industries=[]
    for row in table.findAll('tr')[1:]:
            ticker = row.findAll('td')[0].text
            ticker = ticker[:-1]
            ticker=ticker.replace('.', '-')
            tickers.append(ticker)
            #tickers= sorted(tickers, key=str.lower)

            name = row.findAll('td')[1].text
            #name = name[:-1]
            names.append(name)
            #names= sorted(names, key=str.lower)

    YAHOO_VENDOR_ID=1
    #Existing tickers
    all_tickers = pd.read_sql("SELECT ticker, id FROM security", mydb)

    #códigos úteis
    #all_tickers['ticker'].isin(all_tickers['ticker'])
    #df2 = df2.append(df.loc[df['C'] == x])
    #csvs=os.path.join(folder_list[0],lista[0])


    for i,element in enumerate(tickers):
        if element not in all_tickers['ticker'].values:
            print(element)
            values_aux=[YAHOO_VENDOR_ID, element ,names[i]]
            mycursor.execute("INSERT INTO security (exchange_id, ticker, name) VALUES (%s, %s, %s)", tuple(values_aux))

    mydb.commit()




def get_data_from_yahoo_sp500(reload_sp500=False):
    if reload_sp500:
        save_sp500_tickers()
        
        all_tickers = pd.read_sql("SELECT ticker, id FROM security", mydb)
        ticker_index = dict(all_tickers.to_dict('split')['data'])
        tickers = list(ticker_index.keys())


        #if today.strftime('%Y%m%d') not in os.path.basename('options_dfs_'+ today.strftime('%Y%m%d-%H%M'))    
        if not os.path.exists('options_dfs_'+ today.strftime('%Y%m%d-%H%M')):
            os.makedirs('options_dfs_'+ today.strftime('%Y%m%d-%H%M'))

    else:
        
        all_tickers = pd.read_sql("SELECT ticker, id FROM security", mydb)
        ticker_index = dict(all_tickers.to_dict('split')['data'])
        tickers = list(ticker_index.keys())
        
        if not os.path.exists('options_dfs_'+ today.strftime('%Y%m%d-%H%M')):
            
            os.makedirs('options_dfs_'+ today.strftime('%Y%m%d-%H%M'))

    chunk_size=12
    n_chunks = -(-len(tickers) // chunk_size)        
    for i in range(n_chunks):
        start = chunk_size * i
        end = chunk_size * i + chunk_size
        print(start, end)

        for ticker in tickers[start:end]:
            
            if not os.path.exists('options_dfs_'+ today.strftime('%Y%m%d-%H%M')+'/{}.csv'.format(ticker)):
                
                ticker=ticker.replace('.', '-')
                try:
                    df =options_chain(ticker)
                except Exception as e:
                    print(str(e))
                    time.sleep(3)
                    continue
                print(ticker)
                #df.reset_index(inplace=True)
                #df.set_index("Date", inplace=True)
                if df.empty:
                    continue
                else:
                    df.to_csv('options_dfs_'+ today.strftime('%Y%m%d-%H%M')+'/{}.csv'.format(ticker))
            else:
                print('Already have {}'.format(ticker))
        #reload(yf)
        #reload(pd)
        #reload(dt)
            
        

        
            
# Para extrair a optionchain de um simbolo
#dataf=options_chain('^SPX')
#%%
import re
directory = os.listdir(os.path.dirname(os.path.abspath(__file__)))

aux=0
today = datetime.now()
searchstring=today.strftime("%Y%m%d")

for fname in directory:
    m=re.search(r"\d", fname)
    if searchstring in fname and aux==0:
        today=dt.datetime(int(fname[m.start():m.start()+4]),int(fname[m.start()+4:m.start()+6]),int(fname[m.start()+6:m.start()+8]),int(fname[m.start()+9:m.start()+11]),int(fname[m.start()+11:m.start()+13]))
        
        
        get_data_from_yahoo_sp500(reload_sp500=True)
        aux=1
    elif aux==1:
        break
    else:
        continue

    
if aux == 0:
    get_data_from_yahoo_sp500(reload_sp500=True)


# %%

#YAHOO_VENDOR_ID = 1

# Get present tickers
#all_tickers = pd.read_sql("SELECT ticker, id FROM security", mydb)
#ticker_index = dict(all_tickers.to_dict('split')['data'])


#def download_data_chunk(start_idx, end_idx, start_date, tickerlist):
#    """
#    Download stock data using pandas-datareader
#    :param start_idx: start index
#    :param end_idx: end index
#    :param tickerlist: which tickers are meant to be downloaded
#    :param start_date: the starting date for each ticker
#    :return: writes data to mysql database
#    """
#    ms_tickers = []
#    for ticker in tickerlist[start_idx:end_idx]:
#        
#        df = yf.download(ticker, start=start_date, interval="1d")
#        df=df[df.index>start_date.strftime('%Y-%m-%d')]
#        df1=df.where((pd.notnull(df)), None)
#        if df1.empty:
#            print(f"df is empty for {ticker}")
#            ms_tickers.append(ticker)
#            #time.sleep(5)
#            continue
#       
#        for row in df1.itertuples():
#            values = [YAHOO_VENDOR_ID, ticker_index[ticker]] + list(row)
#            values[2] = values[2].strftime('%Y-%m-%d %H:%M:%S')
#            try:
#                sql= "INSERT INTO daily_price (data_vendor_id,\
#                ticker_id, price_date, open_price,\
#                high_price, low_price, close_price,\
#                adj_close_price, volume) VALUES (%s, %s, %s, %s, %s, %s, %s, %s, %s)"
#                mycursor.execute(sql, tuple(values))
#            except Exception as e:
#                print(str(e))

#        mydb.commit()
#    return ms_tickers


""" def download_all_data(tickerlist, start_date, chunk_size=100):
    n_chunks = -(-len(tickerlist) // chunk_size)

    ms_tickers = []
    for i in range(n_chunks):
        start = 100 * i
        end = 100 * i + chunk_size
        print(start, end)
        # This will download data from the earliest possible date
        ms_from_chunk = download_data_chunk(start, end, start_date, tickerlist)
        ms_tickers.append(ms_from_chunk)
        

    return ms_tickers """


""" 
def update_prices():

    present_ticker_ids = pd.read_sql("SELECT DISTINCT ticker_id FROM daily_price",mydb)
    index_ticker = {v: k for k, v in ticker_index.items()}
    present_tickers = [index_ticker[i]
                       for i in list(present_ticker_ids['ticker_id'])]
    # Get last date
    sql = "SELECT price_date FROM daily_price WHERE ticker_id=1"
    dates = pd.read_sql(sql, mydb)
    #last_date = dates.iloc[-1, 0]+dt.timedelta(days=2) #queremos o dia seguinte. Como o Yf descarrega a data do dia anterior, na realidade queremos 2 dias à frente
    
    #if dates.iloc[-1, 0].weekday()==0:
        #last_date = dates.iloc[-1, 0]+ BDay(1) #queremos o dia seguinte. Como o Yf descarrega a data do dia anterior, na realidade queremos 2 dias à frente
    #else:
        #last_date = dates.iloc[-1, 0]+ BDay(2) #queremos o dia seguinte. Como o Yf descarrega a data do dia anterior, na realidade queremos 2 dias à frente
    last_date=dates.iloc[-1, 0]
    download_all_data(present_tickers, last_date, chunk_size=100)

update_prices() """


#UPDATE chains from csv
""" def update_chains_from_csv(lista_pastas, lista_tickers, data_inicio, datas):
    
    YAHOO_VENDOR_ID = 1
    
    #lower=bisect.bisect([d.date() for d in datas], data_inicio)
    lower=bisect.bisect([d.date() for d in datas], data_inicio)
    
    for f in tqdm(lista_pastas[(lower):]):
        if datetime.strptime(re.compile(r'\d+').findall(f)[0], '%Y%m%d').isoweekday() in range (1,6):
            
            for ticker in tqdm(lista_tickers):

                ticker=ticker+'.csv'
                # Download data
                if os.path.isfile(os.path.join(f, ticker)):

                    df = pd.read_csv(os.path.join(f,ticker),index_col=0)
                    df1 = df.where((pd.notnull(df)), None)
                    # Write to daily_price
                    for row in df1.itertuples(index=False):
                        try:
                            values = [YAHOO_VENDOR_ID, ticker_index[ticker[:-4]]] +re.compile(r'\d+').findall(f)[:-1] +  list(row)
                            values[2] = datetime.strptime(values[2], '%Y%m%d').strftime('%Y-%m-%d %H:%M:%S')
                            mycursor.execute("INSERT INTO options (data_vendor_id,\
                                ticker_id, chain_date, contractSymbol, strike, bid,\
                                    ask, volume, openInterest, impliedVolatility, inTheMoney, expirationDate, CALLOPTION, mark)\
                                        VALUES (%s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s)",tuple(values))
                        except Exception as e:
                            print(str(e))
                else:
                    print(ticker+' doesn t exist on folder '+f)
                    continue
            mydb.commit()

        else:
            continue
 """
#Update options csvs (testar primeiro a insercao na temp_options)
""" def update_chains():
    
    dates=[datetime.strptime(re.compile(r'\d+').findall(f)[0],'%Y%m%d') for f in os.listdir(os.path.dirname(os.path.abspath(__file__))) if f.startswith('options_dfs_')]
    folders=['options_dfs_'+re.compile(r'\d+').findall(f)[0]+'-'+re.compile(r'\d+').findall(f)[1] for f in os.listdir(os.path.dirname(os.path.abspath(__file__))) if f.startswith('options_dfs_')]
    folder_list=list(map(lambda suffix: os.path.join(os.path.dirname(__file__),suffix), folders))
    # Get present tickers
    present_ticker_ids = pd.read_sql("SELECT DISTINCT ticker_id FROM daily_price",mydb)
    index_ticker = {v: k for k, v in ticker_index.items()}  
    tickers = [index_ticker[i]
                       for i in list(present_ticker_ids['ticker_id'])]
    # Get last date
    sql = "SELECT distinct chain_date FROM options WHERE ticker_id=1"
    dates_q = pd.read_sql(sql, mydb)
    last_date = dates_q.iloc[-1, 0]
    update_chains_from_csv(folder_list, tickers, last_date, dates) """

""" update_chains() """
#####