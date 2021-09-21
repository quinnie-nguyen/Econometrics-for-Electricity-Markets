'''clean the raw data sets

1. drop missing data 

2. imputation'''


#%% import packages

import pandas as pd
import numpy as np


#%% import dataset

hu = pd.read_csv('HU.csv')
huPrice = pd.read_csv('HU_prices_and_fuels.csv')

#%% external regressors

col_names = hu.columns


'''missing values'''
mis = hu.isnull().sum()

'''drop the columns that do not have more than 100000 missing row'''

mis_index = []

for i in range(len(mis)):
    
    if mis[i] > 100000:
        
        mis_index.append(i)
        
col_mis = col_names[mis_index]

hu = hu.drop(columns = col_mis)
    

'''rename the hu'''

hu.columns =    ['DateTime',
                'ActualTotalLoad_BZN',
                'ActualTotalLoad_CTA',
                'ActualTotalLoad_CTY',
                'BiomassActualGenerationOutput_BZN',
                'BiomassActualGenerationOutput_CTA',
                'BiomassActualGenerationOutput_CTY',
                'FossilBrowncoalLigniteActualGenerationOutput_BZN',
                'FossilBrowncoalLigniteActualGenerationOutput_CTA',
                'FossilBrowncoalLigniteActualGenerationOutput_CTY',
                'FossilGasActualGenerationOutput_BZN',
                'FossilGasActualGenerationOutput_CTA',
                'FossilGasActualGenerationOutput_CTY',
                'HydroRun-of-riverandpoundageActualGenerationOutput_BZN',
                'HydroRun-of-riverandpoundageActualGenerationOutput_CTA',
                'HydroRun-of-riverandpoundageActualGenerationOutput_CTY',
                'HydroWaterReservoirActualGenerationOutput_BZN',
                'HydroWaterReservoirActualGenerationOutput_CTA',
                'HydroWaterReservoirActualGenerationOutput_CTY',
                'NuclearActualGenerationOutput_BZN',
                'NuclearActualGenerationOutput_CTA',
                'NuclearActualGenerationOutput_CTY',
                'OtherActualGenerationOutput_BZN',
                'OtherActualGenerationOutput_CTA',
                'OtherActualGenerationOutput_CTY',
                'OtherrenewableActualGenerationOutput_BZN',
                'OtherrenewableActualGenerationOutput_CTA',
                'OtherrenewableActualGenerationOutput_CTY',
                'WasteActualGenerationOutput_BZN',
                'WasteActualGenerationOutput_CTA',
                'WasteActualGenerationOutput_CTY',
                'WindOnshoreActualGenerationOutput_BZN',
                'WindOnshoreActualGenerationOutput_CTA',
                'WindOnshoreActualGenerationOutput_CTY',
                'DayAheadAggregatedGeneration_BZN',
                'DayAheadAggregatedGeneration_CTA',
                'DayAheadAggregatedGeneration_CTY',
                'DayAheadGenerationForecastWindSolar_BZN',
                'DayAheadGenerationForecastWindSolar_CTA',
                'DayAheadGenerationForecastWindSolar_CTY',
                'DayAheadTotalLoadForecast_BZN',
                'DayAheadTotalLoadForecast_CTA',
                'DayAheadTotalLoadForecast_CTY',
                'ConsumptionPrice_MBA',
                'NegativeImbalancePrice_MBA',
                'PositiveImbalancePrice_MBA',
                ]

# drop the first 504 rows because there is no data here
hu = hu.iloc[504:, :]
hu.reset_index(drop=True, inplace=True)

'''imputation'''

def impute_bysimilarcol(df = hu, col_name = 'ActualTotalLoad_'):
    
    for i in range(df.shape[0]):
        
        if np.isnan(df.loc[i, f'{col_name}BZN']) == 1 and np.isnan(df.loc[i, f'{col_name}CTA']) == 0:
            
            df.loc[i, f'{col_name}BZN'] = df.loc[i, f'{col_name}CTA']
            
        if np.isnan(df.loc[i, f'{col_name}BZN']) == 1 and np.isnan(df.loc[i, f'{col_name}CTY']) == 0:
            
            df.loc[i, f'{col_name}BZN'] = df.loc[i, f'{col_name}CTY']
            
    return df[f'{col_name}BZN']

hu_1 = impute_bysimilarcol()
hu_2 = impute_bysimilarcol(col_name = 'BiomassActualGenerationOutput_')       
hu_3 = impute_bysimilarcol(col_name = 'FossilBrowncoalLigniteActualGenerationOutput_')   
hu_4 = impute_bysimilarcol(col_name = 'FossilGasActualGenerationOutput_')    
hu_5 = impute_bysimilarcol(col_name = 'HydroRun-of-riverandpoundageActualGenerationOutput_')
hu_6 = impute_bysimilarcol(col_name = 'HydroWaterReservoirActualGenerationOutput_')
hu_7 = impute_bysimilarcol(col_name = 'NuclearActualGenerationOutput_')    
hu_8 = impute_bysimilarcol(col_name = 'OtherActualGenerationOutput_')
hu_9 = impute_bysimilarcol(col_name = 'OtherrenewableActualGenerationOutput_')
hu_10 = impute_bysimilarcol(col_name = 'WasteActualGenerationOutput_')
hu_11 = impute_bysimilarcol(col_name = 'WindOnshoreActualGenerationOutput_')
hu_12 = impute_bysimilarcol(col_name = 'DayAheadAggregatedGeneration_')
hu_13 = impute_bysimilarcol(col_name = 'DayAheadGenerationForecastWindSolar_')
hu_14 = impute_bysimilarcol(col_name = 'DayAheadTotalLoadForecast_')
hu_15 = hu.iloc[:, -3]

imputedF_hu = pd.DataFrame(data = [hu.DateTime, hu_1, hu_2, hu_3, hu_4, hu_5,
                                   hu_6, hu_7, hu_8, hu_9, hu_10, hu_11, hu_12,
                                   hu_13, hu_14, hu_15]).T
imputedF_hu.columns =  ['DateTime',
                        'Load_Actual',
                        'Biomass_Actual',
                        'FossilBrowncoalLignite_Actual',
                        'FossilGas_Actual',
                        'HydroRun-of-riverandpoundage_Actual',
                        'HydroWaterReservoir_Actual',
                        'Nuclear_Actual',
                        'Other_Actual',
                        'Otherrenewable_Actual',                        
                        'Waste_Actual',                       
                        'WindOn_Actual',                        
                        'AggregatedGeneration_DA',                        
                        'WindSolar_DA',                        
                        'Load_DA',                       
                        'Price_MBA'
                        ]

'''impute the missing values by median'''

imputedF_hu.DateTime = pd.to_datetime(imputedF_hu.DateTime)
imputedF_hu.insert(loc = 1, column = 'Year', value = imputedF_hu.DateTime.dt.year)
imputedF_hu.insert(loc = 2, column = 'Month', value = imputedF_hu.DateTime.dt.month)
imputedF_hu["Load_Actual_f"] = imputedF_hu.groupby(['Year', 'Month'])['Load_Actual'].transform(lambda x: x.fillna(x.median()))
imputedF_hu["Biomass_Actual_f"] = imputedF_hu.groupby(['Year', 'Month'])['Biomass_Actual'].transform(lambda x: x.fillna(x.median()))
imputedF_hu["FossilBrowncoalLignite_Actual_f"] = imputedF_hu.groupby(['Year', 'Month'])['FossilBrowncoalLignite_Actual'].transform(lambda x: x.fillna(x.median()))
imputedF_hu["FossilGas_Actual_f"] = imputedF_hu.groupby(['Year', 'Month'])['FossilGas_Actual'].transform(lambda x: x.fillna(x.median()))
imputedF_hu["HydroRun-of-riverandpoundage_Actual_f"] = imputedF_hu.groupby(['Year', 'Month'])['HydroRun-of-riverandpoundage_Actual'].transform(lambda x: x.fillna(x.median()))
imputedF_hu["HydroWaterReservoir_Actual_f"] = imputedF_hu.groupby(['Year', 'Month'])['HydroWaterReservoir_Actual'].transform(lambda x: x.fillna(x.median()))
imputedF_hu["Nuclear_Actual_f"] = imputedF_hu.groupby(['Year', 'Month'])['Nuclear_Actual'].transform(lambda x: x.fillna(x.median()))
imputedF_hu["Other_Actual_f"] = imputedF_hu.groupby(['Year', 'Month'])['Other_Actual'].transform(lambda x: x.fillna(x.median()))
imputedF_hu["Otherrenewable_Actual_f"] = imputedF_hu.groupby(['Year', 'Month'])['Otherrenewable_Actual'].transform(lambda x: x.fillna(x.median()))
imputedF_hu["Waste_Actual_f"] = imputedF_hu.groupby(['Year', 'Month'])['Waste_Actual'].transform(lambda x: x.fillna(x.median()))
imputedF_hu["WindOn_Actual_f"] = imputedF_hu.groupby(['Year', 'Month'])['WindOn_Actual'].transform(lambda x: x.fillna(x.median()))
imputedF_hu["AggregatedGeneration_DA_f"] = imputedF_hu.groupby(['Year', 'Month'])['AggregatedGeneration_DA'].transform(lambda x: x.fillna(x.median()))
imputedF_hu["WindSolar_DA_f"] = imputedF_hu.groupby(['Year', 'Month'])['WindSolar_DA'].transform(lambda x: x.fillna(x.median()))
imputedF_hu["Load_DA_f"] = imputedF_hu.groupby(['Year', 'Month'])['Load_DA'].transform(lambda x: x.fillna(x.median()))
imputedF_hu["Price_MBA_f"] = imputedF_hu.groupby(['Year', 'Month'])['Price_MBA'].transform(lambda x: x.fillna(x.median()))

imputedF_hu =  imputedF_hu[['DateTime',
                        'Load_Actual_f',
                        'Biomass_Actual_f',
                        'FossilBrowncoalLignite_Actual_f',
                        'FossilGas_Actual_f',
                        'HydroRun-of-riverandpoundage_Actual_f',
                        'HydroWaterReservoir_Actual_f',
                        'Nuclear_Actual_f',
                        
                        'Other_Actual_f',
                        
                        'Otherrenewable_Actual_f',
                        
                        'Waste_Actual_f',
                        
                        'WindOn_Actual_f',
                        
                        'AggregatedGeneration_DA_f',
                        
                        'WindSolar_DA_f',
                        
                        'Load_DA_f',
                        
                        'Price_MBA_f'
                        ]]
imputedF_hu.columns =  ['DateTime',
                        'Load_Actual',
                        'Biomass_Actual',
                        'FossilBrowncoalLignite_Actual',
                        'FossilGas_Actual',
                        'HydroRun-of-riverandpoundage_Actual',
                        'HydroWaterReservoir_Actual',
                        'Nuclear_Actual',
                        
                        'Other_Actual',
                        
                        'Otherrenewable_Actual',
                        
                        'Waste_Actual',
                        
                        'WindOn_Actual',
                        
                        'AggregatedGeneration_DA',
                        
                        'WindSolar_DA',
                        
                        'Load_DA',
                        
                        'Price_MBA'
                        ]


#%% electricity and other prices

huPrice = huPrice.iloc[288:, :]


huPrice.columns = ['DateTime', 'Price', 'EUA', 'BrentOil', 'TTFGas', 'API2Coal']

huPrice.DateTime = pd.to_datetime(huPrice.DateTime)
huPrice.reset_index(drop=True, inplace = True)

huPrice['Date'] = huPrice.DateTime.dt.date

'''EUA. brent oil, Gas, and Coal have only one price for a day'''

fill_missing = pd.DataFrame()

fill_missing['EUA'] = huPrice.groupby(['Date'])['EUA'].max()
fill_missing['BrentOil'] = huPrice.groupby(['Date'])['BrentOil'].max()
fill_missing['TTFGas'] = huPrice.groupby(['Date'])['TTFGas'].max()
fill_missing['API2Coal'] = huPrice.groupby(['Date'])['API2Coal'].max()
fill_missing.reset_index(inplace = True)
huPrice = pd.merge(huPrice, fill_missing, how = 'left',
                   on = 'Date')
#huPrice = huPrice.iloc[:54984, :]

huPrice = huPrice[['DateTime', 'Price', 'EUA_y', 'BrentOil_y' , 'TTFGas_y', 'API2Coal_y']]

huPrice.columns = ['DateTime', 'Price', 'EUA', 'BrentOil', 'TTFGas', 'API2Coal']

huPrice.insert(loc = 1, column = 'Year', value = huPrice.DateTime.dt.year)
huPrice.insert(loc = 1, column = 'Month', value = huPrice.DateTime.dt.month)

mprice = pd.DataFrame()

mprice['EUA'] = huPrice.groupby(['Year', 'Month'])['EUA'].median()
mprice['BrentOil'] = huPrice.groupby(['Year', 'Month'])['BrentOil'].median()
mprice['TTFGas'] = huPrice.groupby(['Year', 'Month'])['TTFGas'].median()
mprice['API2Coal'] = huPrice.groupby(['Year', 'Month'])['API2Coal'].median()

huPrice["EUA_f"] = huPrice.groupby(['Year', 'Month'])['EUA'].transform(lambda x: x.fillna(x.median()))
huPrice["BrentOil_f"] = huPrice.groupby(['Year', 'Month'])['BrentOil'].transform(lambda x: x.fillna(x.median()))
huPrice["TTFGas_f"] = huPrice.groupby(['Year', 'Month'])['TTFGas'].transform(lambda x: x.fillna(x.median()))
huPrice["API2Coal_f"] = huPrice.groupby(['Year', 'Month'])['API2Coal'].transform(lambda x: x.fillna(x.median()))

huPrice = huPrice[['DateTime', 'Price', 'EUA_f', 'BrentOil_f' , 'TTFGas_f', 'API2Coal_f']]

huPrice.columns = ['DateTime', 'Price', 'EUA', 'BrentOil', 'TTFGas', 'API2Coal']

#%%

#imputedF_hu.DateTime = pd.to_datetime(imputedF_hu.DateTime)

data = pd.merge(huPrice, imputedF_hu, how = 'left', on = ['DateTime'])
data = data[data['Price'].isnull() == False ]
data.reset_index(drop = True, inplace = True)
# data.to_csv('Final_Dataset.csv', index = False)

