@ECHO OFF 
TITLE Execute python script on anaconda environment
ECHO Please Wait...
:: Section 1: Activate the environment.
ECHO ============================
ECHO Conda Activate
ECHO ============================
@CALL "C:\Users\G\anaconda3\Scripts\activate.bat" base
:: Section 2: Execute python script.
ECHO ============================
ECHO Python get_sp500_chain_sql.py
ECHO ============================
python "D:\Mega\Assuntos Pessoais\Capital Management\Python\Options -VIX Calc\get_sp500_chain_sql.py"

timeout /t 300

python "D:\Mega\Assuntos Pessoais\Capital Management\Python\Options -VIX Calc\get_sp500_chain_sql.py"

ECHO ============================
ECHO End
ECHO ============================

PAUSE