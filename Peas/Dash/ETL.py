
# Aim: read data from iplant and do manipulation then uplaod to db for Dash to access
# RainShelterPea2022
# sqlalchemy and pandas solution
from sqlalchemy import create_engine
import pandas as pd
from sqlalchemy.pool import NullPool
import datetime
import numpy as np

def UpdateSWDGraphData():
    #Read in data
    print("Read .dat file.")
    AllData=pd.read_csv('I:/Science Projects/I211007-02/Exception Files/04 Research/LoggedData/RainShelterAlpha_CS650.dat', #specify file path for data to read in
                             parse_dates=True, #tell the function to parse date columns to datetime formats
                             dayfirst=True, #tell the function that the day is before the year in the data i.e format='%d/%m/%Y %H:%M'
                             skiprows = [0,2,3], #leave out rows 1, 3 and 4 which have redundant information
                             index_col = 0, #Use the first column, which is Date, as an index
                             na_values = 'NAN')

    #Bring in index data
    print("Read index file.")
    AllDataIndex=pd.read_csv('I:/Science Projects/I211007-02/Exception Files/04 Research/LoggedData/SoilWaterAndTempIndex.csv',
                             index_col = 0)    #Read in data
    AllDataIndex.loc[:,'Irrigation'] = pd.Categorical(AllDataIndex.loc[:,'Irrigation'],['date','2D','7D','14D','21D','MD','LD'])
    #Apply indexes to data
    print("Processing data.")
    AllDataTransposed = AllData.transpose()

    AllDataIndexed = pd.concat([AllDataIndex,AllDataTransposed], axis=1,sort=False)
    AllDataIndexed.index.name='ColumnHeader'
    AllDataIndexed.set_index(['Measurement','Depth','Irrigation','Species','Plot','Sensor', 'MUX', 'Port','Units','Summary','Block','Treatment'],
                            append=False, inplace=True)
    AllDataIndexed.sort_index(inplace=True)
    Data=AllDataIndexed.transpose()
    Data.index = pd.to_datetime(Data.index)  ## for some reason the concat function changes the data type on the date indes so need to change it back
    Data.columns = Data.columns.droplevel(level=[4,5,6,7,8,9,11]).reorder_levels(['Measurement', 'Depth', 'Irrigation', 'Species', 'Block'])

    ObsData =  Data.VolumetricWaterContent.copy()
    ObsData.sort_index(axis=0,inplace=True)
    ObsData.sort_index(axis=1,inplace=True)

    #Set up factorLevel descriptors
    Depths = ObsData.columns.get_level_values(0).drop_duplicates().values
    F1 = ObsData.columns.get_level_values(1).drop_duplicates().values
    F2 = ObsData.columns.get_level_values(2).drop_duplicates().values
    Blocks = ObsData.columns.get_level_values(3).drop_duplicates().values
    ObsData.loc['2021-12-13 05:00:00':'2021-12-13 17:00:00',:] =np.nan

    #Calculate treatment means omitting data prior to 2014-11-05 08:00:00 to avoid NaN values
    DataMeans =  ObsData.groupby(level=['Depth','Irrigation'],axis=1).mean()

    #Calculate the water content of the soil profile by multiplying the volumetric water content by each layers
    #depth and summing.  The 0-15 layers are divided by 2 to average the two readings
    ProfileWater = DataMeans.loc[:,'D1I'] * 150/2 + \
                   DataMeans.loc[:,'D1B'] * 150/2 + \
                   DataMeans.loc[:,'D2'] * 150 + \
                   DataMeans.loc[:,'D3'] * 300 + \
                   DataMeans.loc[:,'D4'] * 300 + \
                   DataMeans.loc[:,'D5'] * 300 + \
                   DataMeans.loc[:,'D6'] * 300 + \
                   DataMeans.loc[:,'D7'] * 300

    FieldCapacity = ProfileWater.resample('D').max()
    FieldCapacity = FieldCapacity.loc['2021-11-25']   # I would have though this would return a data frame with a single row but instead it returns a series with a multiindex in columns
    SoilWaterDeficit = -(FieldCapacity - ProfileWater.loc['2021-11-18':,:])
    SoilWaterDeficit.loc[:,'date'] = SoilWaterDeficit.index.get_level_values(0)
    engine = create_engine("postgresql://cflfcl_Rainshelter_SWC:o654UkI6iGNwhzHu@database.powerplant.pfr.co.nz/cflfcl_Rainshelter_SWC")
    ## write processed data to postgresql db
    print("Uploading processed data to postgresql database.")
    SoilWaterDeficit.to_sql(name="RainShelterPea2022",con=engine,if_exists='append', index_label="Irrigation")
UpdateSWDGraphData()
