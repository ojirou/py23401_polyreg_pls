import numpy as np
import seaborn as sns
from sklearn.linear_model import LinearRegression
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error, r2_score
from sklearn.preprocessing import PolynomialFeatures
import pandas as pd
import matplotlib as mpl
import matplotlib.pyplot as plt
import seaborn as sns
from matplotlib import rcParams
from matplotlib import ticker
import webbrowser
import subprocess
import pickle
base_folder=r'C:\\Users\\user\\git\\github\\py2401_polyreg_pls\\'
FileName=base_folder+'regression_pls.csv'
columns=['x1','x2','x3','x4','x5','x6','x7','x8','x9','x10','x11','x12','x13','x14','x15','x16','x17','x18','x19','Target']
df=pd.read_csv(FileName, encoding='utf-8', engine='python', usecols=columns)
features=[c for c in df.columns if c !='Target']
X=df[features].values
y=df['Target'].values
X_train, X_test, y_train, y_test=train_test_split(
        X, y, test_size=0.2, random_state=12)
poly_reg=PolynomialFeatures(degree=2)
X_train_poly=poly_reg.fit_transform(X_train)
X_test_poly=poly_reg.fit_transform(X_test)
model=LinearRegression()
model.fit(X_train_poly, y_train)
with open('polyreg_pls.pickle', mode='wb') as f:
    pickle.dump(model, f)
def adjusted_r2(X, y, model):
    r_squared=r2_score(y, model.predict(X))
    adjusted_r2=1-(1-r_squared)*(len(y)-1)/(len(y)-X.shape[1]-1)
    return adjusted_r2     
def get_model_evaluation(X_train, y_train, X_test, y_test, model):
    import subprocess
    from sklearn.metrics import explained_variance_score
    from sklearn.metrics import mean_absolute_error
    from sklearn.metrics import mean_squared_error
    from sklearn.metrics import mean_squared_log_error
    from sklearn.metrics import median_absolute_error
    yhat_test=model.predict(X_test)
    yhat_train=model.predict(X_train)
    with open('PrintOut.txt', 'w') as f:
        print('PolymonialFeatures(degree=2)', file=f)
        print('adjusted_r2(train) :', round(adjusted_r2(X_train, y_train, model), 3), file=f)
        print('adjusted_r2(test)  :', round(adjusted_r2(X_test, y_test, model), 3), file=f)
        print('MeanErrorRate(test) :', round(np.mean(abs(y_test / yhat_test-1)),3), file=f) 
        print('MAE(test)           :', round(mean_absolute_error(y_test, yhat_test),3), file=f)
        print('MedianAE(test)      :', round(median_absolute_error(y_test, yhat_test),3), file=f)
        print('MSE(train)          :', round(mean_squared_error(y_train, yhat_train),3), file=f)
        print('MSE(test)           :', round(mean_squared_error(y_test, yhat_test),3), file=f)
        print('RMSE(test)/MAE(test):', round(np.sqrt(mean_squared_error(y_test, yhat_test))/mean_absolute_error(y_test, yhat_test), 3), file=f)
    subprocess.Popen(['start', 'PrintOut.txt'], shell=True)
    return
get_model_evaluation(X_train_poly, y_train, X_test_poly, y_test, model)
def set_rcParams():
    rcParams['xtick.labelsize']=12
    rcParams['ytick.labelsize']=12
    rcParams['figure.figsize']=18,8
    sns.set_style('whitegrid')
set_rcParams
sns.set_color_codes()
plt.figure()
fig, ax=plt.subplots(figsize=(7,4))
ax=sns.regplot(y=y_test, x=model.predict(X_test_poly), fit_reg=False, color='#4F818D', scatter_kws={'s': 4})
plt.xlim([-12, 2])
plt.xlim([-12, 2])
ax.set_ylabel(u'Target')
ax.set_xlabel(u'Predicted Target')
ax.get_xaxis().set_major_formatter(ticker.FuncFormatter(lambda x, p: format(int(x), ',')))
ax.get_yaxis().set_major_formatter(ticker.FuncFormatter(lambda y, p: format(int(y), ',')))
ax.plot([-12,2],[-12,2], linewidth=1, color='#C0504D',ls='--')
with open('PrintOut.txt', 'r') as f:
    lines=f.readlines()
    for i, line in enumerate(lines):
        plt.text(-79, -24 -i*3.2, line.strip())
for spline in ax.splines.values():
    spline.set_edgecolor('black')
PdfFile='pdf\\polyreg_pls.pdf'
fig.savefig(PdfFile)
webbrowser.open_new(PdfFile)