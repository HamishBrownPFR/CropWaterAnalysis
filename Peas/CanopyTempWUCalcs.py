# ---
# jupyter:
#   jupytext:
#     formats: ipynb,py:light
#     text_representation:
#       extension: .py
#       format_name: light
#       format_version: '1.5'
#       jupytext_version: 1.4.2
#   kernelspec:
#     display_name: Python 3
#     language: python
#     name: python3
# ---

import dash
from dash import dash_table
import dash_html_components as html
import dash_core_components as dcc
import plotly.graph_objects as go
import plotly.express as px
import datetime
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.dates as mdates
from matplotlib.dates import DayLocator, HourLocator, DateFormatter, drange
import ETFunctions as et
import os
# %matplotlib inline
Lattitude = -43.38
LongitudeTZ = 185
LongitudeML = 360 - 172.28


def UpdateIRGraphData():
    RawData=pd.read_csv('I:/Science Projects/I211007-02/Exception Files/04 Research/LoggedData/RainShelterBeta_Energy.dat', #specify file path for data to read in
                         parse_dates=True, #tell the function to parse date columns to datetime formats
                         dayfirst=True, #tell the function that the day is before the year in the data i.e format='%d/%m/%Y %H:%M'
                         skiprows = [0,2,3], #leave out rows 1, 3 and 4 which have redundant information
                         index_col = 0, #Use the first column, which is Date, as an index
                         na_values = 'NAN')
    #Bring in index data
    DataIndex=pd.read_excel('I:/Science Projects/I211007-02/Exception Files/04 Research/LoggedData/RadiationAndTempIndex.xlsx',  engine='openpyxl',
                            sheet_name='Sensor positions',
                            usecols = range(9))
    DataIndex.loc[:,'Irrigation'] = pd.Categorical(DataIndex.loc[:,'Irrigation'],['Expt','2D','7D','14D','21D','MD','LD'])
    DataIndex.set_index('ColumnHeader',inplace=True)
    DataIndex.dropna(inplace=True)
    # #Apply indexes to data
    DataTransposed = RawData.transpose() 
    DataIndexed = pd.concat([DataIndex,DataTransposed], axis=1,sort=False)
    DataIndexed.index.name='ColumnHeader'
    DataIndexed.set_index(['Measurement','Irrigation','Species','Treatment','Units','Summary','Plot','Block'], 
                             append=False, inplace=True)
    Data=DataIndexed.transpose()
    Data.index = pd.to_datetime(Data.index)  ## for some reason the concat function changes the data type on the date indes so need to change it back
    Data = Data.drop(columns='Time',level=0) ## Drop blank time column
    Data.sort_index(inplace=True)
    Data.sort_index(axis=1,inplace=True)
    return Data


AllData = UpdateIRGraphData().loc['2021-12-10':,:]

AllData.index

# +
radn1 = AllData.loc[:,'GlobalRadn']
radn2 = AllData.loc[:,'AboveCanopyPAR1']
radn3 = AllData.loc['2021-12-09':,('ReflectedRadnKW','7D','Peas',2,'kW/m^2','Avg',109)]*1000
MetData = pd.DataFrame(index = AllData.index, data = radn1.join(radn2).join(radn3).mean(axis=1)*600/1000000,columns=['Rs'])
MetData.loc[:,'Ta'] = AllData.loc[:,'AirTemperature']
MetData.loc[:,'RH'] = AllData.loc[:,'RelativeHumidity']/100
MetData.loc[:,'u'] = AllData.loc[:,'Windspeed']
MetData.loc[:,'Es'] = [et.saturated_vapor_pressure(MetData.loc[d,'Ta'])/10 for d in MetData.index]
MetData.loc[:,'Eo'] = [MetData.loc[d,'Es'] * MetData.loc[d,'RH'] for d in MetData.index]
MetData.loc[:,'Ed'] = MetData.loc[:,'Es'] - MetData.loc[:,'Eo']
MetData.loc[:,'Rex'] = [et.ExtraterestialRadiation(Lattitude,
                                                       d.dayofyear,
                                                       0.16666,
                                                       float(d.time().hour) + float(d.time().minute)/60.0,
                                                       LongitudeTZ,
                                                       LongitudeML) for d in MetData.index]
MetData.loc[:,'Rso'] = [et.ClearSkyRadiation(Lattitude,
                                                 d.dayofyear,
                                                 0.16666,
                                                 float(d.time().hour) + float(d.time().minute)/60.0,
                                                 LongitudeTZ,
                                                 LongitudeML) for d in MetData.index]

MetData.loc[:,'Rn'] = [et.NetRadiation(MetData.loc[d,'Rs'],
                                                    MetData.loc[d,'Ta'],
                                                    MetData.loc[d,'Eo'],
                                                    Lattitude,
                                                    d.dayofyear,
                                                    0.16666,
                                                    float(d.time().hour) + float(d.time().minute)/60.0,
                                                    LongitudeTZ,
                                                    LongitudeML,
                                                    0.2) for d in MetData.index]
MetData.sort_index(inplace=True)
Daylight = (MetData.loc[:,'Rs'] > 0.02).values
# -

figure = plt.figure(figsize=(18,10))
start = '2021-12-25'
plt.plot(MetData.loc[start:,'Rex'],'-o',color='r',label='Extraterestial')
plt.plot(MetData.loc[start:,'Rso'],'-o',color='y',label='Clear Sky')
plt.plot(MetData.loc[start:,'Rs'],'-o',color='b',label='Surface')
plt.plot(MetData.loc[start:,'Rn'],'-o',color='orange',label='Net')
plt.legend(loc=2,fontsize=20)
plt.title('Solar Radiation (MJ/m2/10min)',fontsize=28)
plt.tick_params(labelsize=20)
plt.ylabel('Solar Radiation (MJ/m2/10min)',fontsize = 22)

AllData.loc[:,('IR_SurfaceTemp','7D','Peas',2,'Deg C','Avg',114,3)] = np.nan
for irc in AllData.loc[:,'IR_SurfaceTemp'].columns:
    key = ('IR_SurfaceTemp',) + irc
    AllData.loc[:,key] = AllData.loc[:,key].where(AllData.loc[:,key]>3,np.nan)
SurfaceTemp = AllData.loc[:,'IR_SurfaceTemp'].groupby('Irrigation',axis=1).mean()

tPAR = AllData.loc[Daylight,'BelowCanopyPAR'].resample('d').mean().groupby('Irrigation',axis=1).mean()
iPAR = AllData.loc[Daylight,'AboveCanopyPAR1'].resample('d').mean().groupby('Irrigation',axis=1).mean()
fPAR = (1-tPAR.divide(iPAR.values))
fPAR.plot()

# +
TempData = MetData.loc[:,'Ta'].resample('D').mean().cumsum()
#Layout = pd.read_excel('K:\CPDiary\Data\BaxtersMVI\TrialLayout.xlsx',index_col='Plot')
NDVIPlotMaping = pd.read_excel('I:/Science Projects/I211007-02/Exception Files/04 Research/GSDataFiles/PlotMapping.xlsx',index_col='NDVISample#', engine='openpyxl')
GSFiles = []
mydir = 'I:/Science Projects/I211007-02/Exception Files/04 Research/GSDataFiles'
for File in os.listdir(mydir):
    if File.endswith('.txt'):
        if ('Avg' not in File) & ('Diag' not in File) :
            GSFiles.append(os.path.join(mydir, File))

            #Set up blank dataframe to take data
ColumnLables = ['Time(ms)','SampleNumber','Count','NDVI','VI_2','REDrefc','NIRrefc','Plot','Zone','Rep','Irrig','Date']
TabIndex = pd.MultiIndex.from_arrays([[],[],[]],names=['Date','Irrigation','Rep'])
NDVIData = pd.DataFrame(columns=ColumnLables[2:],index = TabIndex)
ObsDates = []
#Read each file
for ObsTable in GSFiles:
    #Read in file and find which lines have legit data
    DateString = ObsTable.strip('.txt')[-8:]
    ObsDate = pd.to_datetime(DateString,format = '%y.%m.%d')
    ObsDates.append(ObsDate)
    DataTab = pd.read_csv(ObsTable,skiprows = [0],header=None)
    DataTab.columns = ColumnLables[0:7]
    DataTab.dropna(inplace=True)
    DataTab.drop('Time(ms)',axis=1,inplace=True)
    DataTab.loc[:,'Plot'] = [NDVIPlotMaping.loc[DataTab.loc[x,'SampleNumber'],'Plot'] for x in DataTab.index]
    DataTab.loc[:,'Rep'] = [NDVIPlotMaping.loc[DataTab.loc[x,'SampleNumber'],'Rep'] for x in DataTab.index]
    DataTab.loc[:,'Irrigation'] = [NDVIPlotMaping.loc[DataTab.loc[x,'SampleNumber'],'Treat'] for x in DataTab.index]
    DataTab.loc[:,'Date'] = ObsDate
    DataTab.set_index(['Date','Irrigation','Rep'],inplace=True,drop=False)
    NDVIData = NDVIData.append(DataTab)
NDVIData.sort_index(inplace=True)
ObsDates.sort()

NDVIMeans = NDVIData.groupby(level = ['Date','Irrigation','Rep']).mean()
CropNDVI  = NDVIMeans.loc[~NDVIMeans.index.isin(['soil','soil+pipes'],level=2),'NDVI']
BareNDVI = NDVIMeans.loc[NDVIMeans.index.isin(['soil+pipes'],level=2),'NDVI']
fPARndvi = pd.DataFrame(index = CropNDVI.index, columns=['est'])
for Date in ObsDates:
    BareVal = 0.15
    if BareNDVI[Date].values != np.nan:
        BareVal = BareNDVI[Date].values
    fPARndvi.loc[Date,'est'] = np.subtract(CropNDVI.loc[Date].values, BareVal)/np.subtract(0.82,BareVal)
fPARndvi.loc[:,'est'] = pd.to_numeric(fPARndvi.loc[:,'est']) 
fPARndvi.index  =fPARndvi.index.swaplevel('Date','Irrigation',)
fPARndvi.index  =fPARndvi.index.swaplevel('Date','Rep',)
fPARndvi.loc[:,'AccumTemp'] = [TempData.loc[x] for x in fPARndvi.index.get_level_values(2)]
fPARndvi.sort_index(inplace=True)
zeroDates = pd.Series(index  = ['2D', '7D', '14D', '21D', 'MD', 'LD'], data = [datetime.datetime.strptime('2022-1-25','%Y-%m-%d'),
                                                                               datetime.datetime.strptime('2022-1-22','%Y-%m-%d'),
                                                                               datetime.datetime.strptime('2022-1-20','%Y-%m-%d'),
                                                                               datetime.datetime.strptime('2022-1-19','%Y-%m-%d'),
                                                                               datetime.datetime.strptime('2022-1-19','%Y-%m-%d'),
                                                                               datetime.datetime.strptime('2022-1-19','%Y-%m-%d')])
for d in zeroDates.index:
    for r in [1,2,3,4]:
        fPARndvi.loc[(d,r,zeroDates[d]),'est'] = 0.0
        fPARndvi.loc[(d,r,zeroDates[d]),'AccumTemp'] = TempData.loc[zeroDates[d]]
        fd = datetime.datetime.strptime('2022-1-28','%Y-%m-%d')
        fPARndvi.loc[(d,r,fd),'est'] = 0.0
        fPARndvi.loc[(d,r,fd),'AccumTemp'] = TempData.loc[fd]
        
fPARndvi.sort_index(inplace=True)
fPARndvi.sort_index(axis=1,inplace=True)
fPARTreatMeans = fPARndvi.groupby(level = ['Irrigation','Date']).mean()

# +
Start = ObsDates[0]
Today = datetime.datetime.strptime('2022-1-28','%Y-%m-%d')
DailyDates = pd.date_range(Start,Today)
TreatIndex = fPARndvi.groupby(level=['Irrigation']).mean().index
DailyfPARMeans = pd.DataFrame(index = DailyDates, columns = TreatIndex)
for Treat in DailyfPARMeans.columns:
    DailyfPARMeans.loc[:,Treat] = np.interp(TempData.loc[Start:Today],fPARTreatMeans.loc[Treat,'AccumTemp'],fPARTreatMeans.loc[Treat,'est'])
    
fPAR = pd.concat([fPAR.loc[:'2021-12-28'],DailyfPARMeans.loc['2021-12-29':]])
fPAR.sort_index(inplace=True)
fPAR.sort_index(axis=1,inplace=True)
# -

Treats = ['2D','7D', '14D','21D', 'MD','LD']
NDVIGraph = plt.figure(figsize=(6,6))
Irrigs = DailyfPARMeans.columns.values
colors = ['tab:blue','tab:orange','tab:green','tab:red','tab:purple','tab:brown']
lines = ['-','-','-','--','--','--']
Reps = [1.0, 2.0, 3.0, 4.0]
pos = -1
for Irrig in Treats:
    pos+=1
    for Rep in Reps:
        Plot = fPARndvi.loc[(Irrig,Rep),:]
        plt.plot(Plot.index,Plot.est,'o',mec=colors[pos],mfc=colors[pos],ms=5,mew=2,label='_nolegend_')
    plt.plot(fPAR.index,fPAR.loc[:,(Irrig)],'-',color=colors[pos],label=Irrig,lw=2)
plt.ylabel('Estimated fPAR (from NDVI)',fontsize=12)
plt.xticks(rotation=45)
plt.tick_params(labelsize=10)
plt.ylim(0,1.1)
plt.legend(loc=3,fontsize=10)


# +
## Note.  We found an error in the calculations for estGDay coefficients when 
## revising the companion paper
## As the paper describing this work had already been referreed we have not 
## changed these and note it only makes a samll difference  
## The correct coefficients are commented below
def estGDay(SoilRadn, AirTemp):
    Const = -0.7091  ## -0.6139
    SoilRadEff = np.multiply(SoilRadn.values,0.2149) ##0.0801
    AirTempEff = np.multiply(AirTemp.values,0.0739) ##0.0659
    return Const + SoilRadEff  + AirTempEff 

#Calculate Ts for each treatment
Ts = SurfaceTemp.loc[Daylight,:].dropna().resample('d').mean()
#Calculate Rn for each treatment
Rn = MetData.loc[Daylight,'Rn'].resample('d').sum().loc[Ts.index]
#calculate Rs for the experiment
Rs = MetData.loc[Daylight,'Rs'].resample('d').sum().loc[Ts.index]
#Calculate Ta for the experiment
Ta = MetData.loc[Daylight,'Ta'].resample('d').mean().loc[Ts.index]
#Calculate Ed for the experiment
Ed = MetData.loc[Daylight,'Ed'].resample('d').mean().loc[Ts.index]
#Calculate u for the experiment
u = MetData.loc[Daylight,'u'].resample('d').mean().loc[Ts.index]
#Calculate G for each treatment
G = pd.DataFrame(index=Ts.index, columns=Ts.columns)
for p in G.columns:
    G.loc[:,p] = estGDay(Rs * (1-fPAR.loc[:,p]), Ta)
RnLessG = pd.DataFrame(index=Ts.index, columns=Ts.columns)
for p in RnLessG.columns:
    RnLessG.loc[:,p] = Rn -  G.loc[:,p]
#Calculate Eo for the experiment
Eo = MetData.loc[Daylight,'Eo'].resample('d').mean().loc[Ts.index]
# -

plt.plot(Rs,Rn,'o',color = 'magenta',lw=0.5,label = 'Net Rad (Rn)')
plt.plot(Rs,G,'o',color = 'brown',lw=0.5,label = 'Ground Heat (G)')
plt.plot(Rs,RnLessG,'o',color = 'cyan',lw=0.5,label = 'Rn - G')
plt.plot([0,30],[0,20],'-')
plt.text(0.1,20,'Slope = 0.66')
plt.plot([0,30],[0,0],'-')

# +
#Calculate aerodynamic temperature
To = pd.DataFrame(index = Ts.index,columns=Ts.columns)
for p in To.columns:
    To.loc[:,p] = Ts.loc[:,p] + Ed
    
#Calculate temperature difference
Td = pd.DataFrame(index = Ts.index,columns=Ts.columns)
for p in Td.columns:
    Td.loc[:,p] = To.loc[:,p] - Ta
#Calculate alpha
def AlphaCoeff(Td,fPAR):
    CoverFact = 0.1+1.6*fPAR
    if CoverFact > 1.0:
        CoverFact = 1.0
    return 1.3 * np.exp(Td*-0.13) * CoverFact
    #return 1/(0.68 + 0.18*Td ) * CoverFact
    #return np.exp(0.2663)  * np.exp(Td*-0.1270) * CoverFact
    #return np.exp(0.34)  * np.exp(Td*-0.14) * CoverFact
    
Alpha = pd.DataFrame(index = Ts.index,columns=Ts.columns)
for p in Alpha.columns:
    Alpha.loc[:,p] = [AlphaCoeff(Td.loc[x,p],
                                   fPAR.loc[x,p]) for x in Alpha.index]

#Estimate Water Use by surfaces
E = pd.DataFrame(index = Ts.index,columns=Ts.columns)
for p in E.columns:
    E.loc[:,p] = [et.Priestly_TaylorEO(RnLessG.loc[x,p],
                                         Ta[x],
                                         Alpha.loc[x,p],
                                         'net') for x in E.index]
#Calculate alpha
def Alpha_TsCoeff(Td,fPAR):
    CoverFact = 0.1+1.6*fPAR
    if CoverFact > 1.0:
        CoverFact = 1.0
    return 1/(0.83 + 0.17*Td ) * CoverFact

Alpha_Ts = pd.DataFrame(index = Ts.index,columns=Ts.columns)
for p in Alpha.columns:
    Alpha_Ts.loc[:,p] = [Alpha_TsCoeff(Td.loc[x,p],
                                   fPAR.loc[x,p]) for x in Alpha.index]

E_Ts = pd.DataFrame(index = Ts.index,columns=Ts.columns)
for p in E.columns:
    E_Ts.loc[:,p] = [et.Priestly_TaylorEO(RnLessG.loc[x,p],
                                         Ta[x],
                                         Alpha_Ts.loc[x,p],
                                         'net') for x in E.index]

E_Ts_R = pd.DataFrame(index = Ts.index,columns=Ts.columns)
for p in E_Ts_R.columns:
    E_Ts_R.loc[:,p] = [et.Priestly_TaylorEO(Rn[x],
                                         Ta[x],
                                         Alpha_Ts.loc[x,p],
                                         'net') for x in E.index]

PET = pd.DataFrame(index = Ts.index,columns=Ts.columns)
for plot in PET.columns:
    PET.loc[:,plot] = [et.PenmanEO(Rn[x],
                        Ta[x],
                        u[x],
                        Eo[x],
                        'net') for x in PET.index]
# -

DailyData = pd.DataFrame(Ts.unstack())
DailyData.columns = ['Ts']
DailyVariables = ['G','To','Td','Alpha','E','fPAR','E_Ts','E_Ts_R','PET']
for Var in DailyVariables:
    print(Var)
    VarFrame = globals()[Var]
    Varunstk = pd.DataFrame(VarFrame.unstack())
    Varunstk.columns = [Var]
    DailyData = DailyData.join(Varunstk)

Irrigs = DailyData.index.get_level_values(0).drop_duplicates()
DailyData.loc[:,'Rn'] = np.nan
DailyData.loc[:,'Rs'] = np.nan
DailyData.loc[:,'Ed'] = np.nan
DailyData.loc[:,'Ta'] = np.nan
DailyData.loc[:,'u'] = np.nan
for I in Irrigs:
        DailyData.loc[I,'Rs'] = Rs.values
        DailyData.loc[I,'Ed'] = Ed.values
        DailyData.loc[I,'Ta'] = Ta.values
        DailyData.loc[I,'u'] = u.values

Today = datetime.date.today()
Yesterday = Today - datetime.timedelta(days=1)
colors = ['orange','green','purple','orange','green','purple']
lines = ['-','-','-','--','--','--']
Start = '2021-12-10'
End = Yesterday
Graph = plt.figure(figsize=(18,40))
pos = 0
Fig = Graph.add_subplot(5,1,1)
for Irr in Irrigs:
    lab = Irr
    data = Ts.loc[Start:End,Irr].loc[Start:End]
    plt.plot(data,lines[pos],color = colors[pos],lw=5,label = lab)
    pos +=1
plt.plot(Ta.loc[Start:End],color='k',lw=4,label='AirT')
Fig.legend(loc = 2, fontsize=20)
plt.title('SurfaceTemperature',fontsize=28)
plt.xticks(drange(min(data.index), max(data.index), datetime.timedelta(days = 2)))
plt.tick_params(labelsize=20, rotation=45)
plt.ylabel('Surface temperature (oC)',fontsize = 22)
pos=0
Fig = Graph.add_subplot(5,1,2)
for Irr in Irrigs:
    lab =  Irr
    data = To.loc[Start:End,Irr].loc[Start:End]
    plt.plot(data,lines[pos],color = colors[pos],lw=5,label = lab)
    pos +=1
plt.plot(Ta.loc[Start:End],color='k',lw=4,label='AirT')
Fig.legend(loc = 2, fontsize=20)
plt.title('Aerodynamic Temperature',fontsize=28)
plt.xticks(drange(min(data.index), max(data.index), datetime.timedelta(days = 2)))
plt.tick_params(labelsize=20, rotation=45)
plt.ylabel('Aerodynamic Temperature (oC)',fontsize = 22)
pos=0
Fig = Graph.add_subplot(5,1,3)
for Irr in Irrigs:
    lab =  Irr
    data = Td.loc[Start:End,Irr].loc[Start:End]
    plt.plot(data,lines[pos],color = colors[pos],lw=5,label = lab)
    pos +=1
plt.plot([data.index.min(),data.index.max()],[0,0],color='k',lw=4,label='AirT')
Fig.legend(loc = 2, fontsize=20)
plt.title('Temperature difference',fontsize=28)
plt.xticks(drange(min(data.index), max(data.index), datetime.timedelta(days = 2)))
plt.tick_params(labelsize=20, rotation=45)
plt.ylabel('Temperature difference (oC)',fontsize = 22)
pos=0
Fig = Graph.add_subplot(5,1,4)
for Irr in Irrigs:
    lab =  Irr
    data = Alpha.loc[Start:End,Irr].loc[Start:End]
    plt.plot(data,lines[pos],color = colors[pos],lw=5,label = lab)
    pos +=1
Fig.legend(loc = 2, fontsize=20)
plt.title('Alpha',fontsize=28)
plt.xticks(drange(min(data.index), max(data.index), datetime.timedelta(days = 2)))
plt.tick_params(labelsize=20, rotation=45)
plt.ylabel('Alpha',fontsize = 22)
pos=0
Fig = Graph.add_subplot(5,1,5)
for Irr in Irrigs:
    lab = Irr
    data = E.loc[Start:End,Irr].loc[Start:End]
    plt.plot(data,lines[pos],color = colors[pos],lw=5,label = lab)
    pos +=1
Fig.legend(loc = 2, fontsize=20)
plt.title('Evaporation estimated from canopy temperature',fontsize=28)
plt.xticks(drange(min(data.index), max(data.index), datetime.timedelta(days = 2)))
plt.tick_params(labelsize=20, rotation=45)
plt.ylabel('Evaporation (mm)',fontsize = 22)
plt.tight_layout()

plt.plot(PET.loc[:,'2D'],E.loc[:,'2D'],'o')
plt.plot([0,10],[0,10],'-')
plt.plot([0,10],[0,8],'-')
plt.ylabel('WU from Canopy Temp (mm)')
plt.xlabel('Penman ET (mm)')



Today = datetime.date.today()
Yesterday = Today - datetime.timedelta(days=1)
colors = ['orange','green','purple','orange','green','purple']
lines = ['-','-','-','--','--','--']
Start = '2021-12-10'
End = Yesterday
Graph = plt.figure(figsize=(18,12))
pos = 0
Fig = Graph.add_subplot(2,1,1)
for Irr in Irrigs:
    lab = Irr
    data = Ts.loc[Start:End,Irr].loc[Start:End]
    plt.plot(data,lines[pos],color = colors[pos],lw=5,label = lab)
    pos +=1
plt.plot(Ta.loc[Start:End],color='k',lw=4,label='AirT')
Fig.legend(loc = 2, fontsize=20)
plt.title('SurfaceTemperature',fontsize=28)
plt.xticks(drange(min(data.index), max(data.index), datetime.timedelta(days = 2)))
plt.tick_params(labelsize=20, rotation=45)
plt.ylabel('Surface temperature (oC)',fontsize = 22)
pos=0
Fig = Graph.add_subplot(2,1,2)
for Irr in Irrigs:
    lab = Irr
    data = E.loc[Start:End,Irr].loc[Start:End]
    plt.plot(data,lines[pos],color = colors[pos],lw=5,label = lab)
    pos +=1
Fig.legend(loc = 2, fontsize=20)
plt.title('Evaporation estimated from canopy temperature',fontsize=28)
plt.xticks(drange(min(data.index), max(data.index), datetime.timedelta(days = 2)))
plt.tick_params(labelsize=20, rotation=45)
plt.ylabel('Evaporation (mm)',fontsize = 22)
plt.tight_layout()

E.cumsum().plot()

Ts.subtract(Ts.loc[:,'2D'],axis=0).plot()

E.to_pickle('E_IR_accum.pkl')
