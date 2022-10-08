import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.linear_model import LinearRegression
import statsmodels.formula.api as sm
from yellowbrick.regressor import ResidualsPlot

base = pd.read_csv('mt_cars.csv')
print(base.shape)

print(base.head())

# exclui a primeira coluna da base de dados
base = base.drop(['Unnamed: 0'], axis = 1)

# regressao linear simples
x = base.iloc[:,2].values
y = base.iloc[:,0].values
correlacao = np.corrcoef(x,y)
print(correlacao)

x = x.reshape(-1,1)

# modelo de regressor
modelo = LinearRegression()
modelo.fit(x,y)

# y = a*x +b
print(modelo.intercept_) # coeficiente 'b'

print(modelo.coef_) # coeficiente 'a'

print(modelo.score(x,y))

previsoes = modelo.predict(x)
print(previsoes)

modelo_ajustado = sm.ols(formula='mpg ~ disp', data=base)
modelo_treinado = modelo_ajustado.fit()
print(modelo_treinado.summary())

# visualização dos resultados
plt.scatter(x,y)
plt.plot(x, previsoes, color = 'red')
plt.ylabel('Autonomia')
plt.xlabel('Cilindradas')

modelo.predict([[240]])

grafi_resid = ResidualsPlot(modelo)
grafi_resid.fit(x,y)
grafi_resid.poof()

# regressao linear multipla

x1 = base.iloc[:,1:4].values
print(x1)

y1 = base.iloc[:,0].values
modelo_mult = LinearRegression()
modelo_mult.fit(x1,y1)

print('\n',f'A precisão é {round(modelo_mult.score(x1,y1)*100,2)}%','\n') # R^2

modelo_mult2 = sm.ols(formula='mpg ~ cyl + disp + hp', data= base)
modelo_mult2 = modelo_mult2.fit()
print(modelo_mult2.summary())

teste = np.array([6,240,140])
teste = teste.reshape(1,-1)
print(modelo_mult.predict(teste))

grafi_resid = ResidualsPlot(modelo_mult)
grafi_resid.fit(x1,y1)
grafi_resid.poof()
