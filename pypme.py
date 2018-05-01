#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Library with PME calculation functions, provides utility for Private Equity analysis.

@author: Michael Wielondek (michael@wielondek.com)
"""
import pandas as pd
import numpy as np
import scipy.optimize
from datetime import date

#Helper functions
def nearest(series, lookup, debug = False):
    if debug==True:
        print( "lookup: " + str(lookup) + "  | closest: " + str(series.iloc[(series-lookup).abs().argsort()[0]]))    
    return series.iloc[(series-lookup).abs().argsort()[0]]

def xnpv(rate, values, dates):
    '''Equivalent of Excel's XNPV function.

    >>> from datetime import date
    >>> dates = [date(2010, 12, 29), date(2012, 1, 25), date(2012, 3, 8)]
    >>> values = [-10000, 20, 10100]
    >>> xnpv(0.1, values, dates)
    -966.4345...
    '''
    if rate <= -1.0:
        return float('inf')
    datesx = pd.to_datetime(dates).apply(lambda x: date(x.year,x.month,x.day)) # Just in case conversion to native datime date
    d0 = datesx[0]    # or min(dates)
    return sum([ vi / (1.0 + rate)**((di - d0).days / 365.0) for vi, di in zip(values, datesx)])

def xirr(values, dates):
    '''Equivalent of Excel's XIRR function.

    >>> from datetime import date
    >>> dates = [date(2010, 12, 29), date(2012, 1, 25), date(2012, 3, 8)]
    >>> values = [-10000, 20, 10100]
    >>> xirr(values, dates)
    0.0100612...
    '''
    try:
        return scipy.optimize.newton(lambda r: xnpv(r, values, dates), 0.0)
    except RuntimeError:    # Failed to converge?
        return scipy.optimize.brentq(lambda r: xnpv(r, values, dates), -1.0, 1e10)
        
def TVM(value, value_date, money_date, discount_rate):
    ''' Calculates the discounted value of money to date money_date (i.e. either PV or FV depending on date)
    '''
    time_delta = ((pd.to_datetime(money_date) - pd.to_datetime(value_date)) / np.timedelta64(1, 'D')).astype(int)/365
    return value*(1+discount_rate)**time_delta
 
    
def xirr2(values, dates):
    datesx = pd.to_datetime(dates).apply(lambda x: date(x.year,x.month,x.day)) # Just in case conversion to native datime date
    transactions = list(zip(datesx,values))
    years = [(ta[0] - transactions[0][0]).days / 365.0 for ta in transactions]
    residual = 1
    step = 0.05
    guess = 0.05
    epsilon = 0.0001
    limit = 100000
    while abs(residual) > epsilon and limit > 0:
        limit -= 1
        residual = 0.0
        for i, ta in enumerate(transactions):
            residual += ta[1] / pow(guess, years[i])
        if abs(residual) > epsilon:
            if residual > 0:
                guess += step
            else:
                guess -= step
                step /= 2.0
    return guess-1

### PME Algorhitms       
#GENERATE DISCOUNTING TABLE     
def discount_table(dates_cashflows, cashflows, cashflows_type, dates_index, index, NAV_scaling = 1):
    ''' Automatically matches cashflow and index dates and subsequently generates discount table (which can be used to calculate Direct Alpha et al.). 
        Also useful for debugging and exporting.
    
    Args:
        dates_cashflows: An ndarray the dates corresponding to cashflows. 
        cashflows: An ndarray of the cashflow amounts (sign does not matter)
        cashflows_type: Accepts three types [Distribution \ Capital Call \ Value]
        dates_index: Ndarray of dates for the index, same logic as cashflows. 
        index: The index levels corresponding to the dates
        NAV_scaling: Coefficient which can be used to scale the NAV amount (so as to counteract systemic mispricing)
        auto_NAV: Toggle for automatic handling of the NAV. If False, NAV is not calculated and function returns a tuple of [sum_fv_distributions, sum_fv_calls] 
                    (allows for manual completion of the PME formula using appropriate NAV value)

    Returns:
        DataFrame(Date|Amount|Type|Status|Discounted|Index|FV_Factor)
    '''
    _dates_index = pd.to_datetime(dates_index)  
    df_cf = pd.concat([pd.to_datetime(dates_cashflows),cashflows,cashflows_type], axis=1)
    df_cf.columns = ["Date", "Amount","Type"]
    df_cf = df_cf.sort_values('Date')
    df_cf = df_cf.reset_index(drop=True)
    #Get NAV 
    if(df_cf[(df_cf['Type']=='Value') & (df_cf['Amount']==0)].empty): #Checks if liquidated by looking at 0 valuations
        NAV_record = df_cf[df_cf['Type']=='Value'].sort_values("Date", ascending = False).head(1).copy()
        NAV_date = NAV_record["Date"].iloc[0]
        NAV_index_value = index[_dates_index == nearest(_dates_index, NAV_date)].iloc[0]
        df_cf['Status'] = 'Active'
    else:  #Not liquidated
        NAV_record = df_cf[df_cf['Type']!='Value'].sort_values("Date", ascending = False).head(1).copy()
        NAV_record['Amount'].iloc[0] = 0 # force a 0 value
        NAV_date = NAV_record["Date"].iloc[0]
        NAV_index_value = index[_dates_index == nearest(_dates_index, NAV_date)].iloc[0]
        df_cf['Status'] = 'Liquidated'
    #Iterate and assign to table
    df_cf["Pre-Discounted"] = 0
    df_cf["Discounted"] = 0 
    df_cf["Index"] = 0
    df_cf["Index_date"] = 0
    df_cf["FV_Factor"] = 0
    for idx, cf in df_cf.iterrows():
        # Let us find the closest index value to the current date
        index_date = nearest(_dates_index, cf["Date"])
        index_value = index[_dates_index == index_date].iloc[0]
        df_cf.loc[idx,"Index"] = index_value  
        df_cf.loc[idx,"Index_date"] = index_date 
        df_cf.loc[idx,"FV_Factor"] = (NAV_index_value/index_value)
        if cf["Type"] == "Distribution":
            df_cf.loc[idx,'Discounted'] = abs(cf["Amount"])* (NAV_index_value/index_value)
            df_cf.loc[idx,'Pre-Discounted'] = abs(cf["Amount"])
        elif cf["Type"] == "Capital Call":
            df_cf.loc[idx,'Discounted'] = - abs(cf["Amount"])* (NAV_index_value/index_value)
            df_cf.loc[idx,'Pre-Discounted'] = - abs(cf["Amount"])
            
    # Attach relevant NAV value 
    df_cf.loc[(df_cf['Date']==NAV_date)&(df_cf['Type']=='Value'),'Discounted'] = NAV_record['Amount'].iloc[0] * NAV_scaling  
    df_cf.loc[(df_cf['Date']==NAV_date)&(df_cf['Type']=='Value'),'Pre-Discounted'] = NAV_record['Amount'].iloc[0] * NAV_scaling  
    
    #cut table at FV date 
    df_cf = df_cf[df_cf["Date"]<=NAV_date].copy()
    return df_cf.copy()

    
#KS-PME
def KS_PME(dates_cashflows, cashflows, cashflows_type, dates_index, index, NAV_scaling = 1, auto_NAV = True):
    """Calculates the Kalpan Schoar PME. Designed for plug & play with Preqin data.

    Args:
        dates_cashflows: An ndarray the dates corresponding to cashflows. 
        cashflows: An ndarray of the cashflow amounts (sign does not matter)
        cashflows_type: Accepts three types [Distribution \ Capital Call \ Value]
        dates_index: Ndarray of dates for the index, same logic as cashflows. 
        index: The index levels corresponding to the dates
        NAV_scaling: Coefficient which can be used to scale the NAV amount (so as to counteract systemic mispricing)
        auto_NAV: Toggle for automatic handling of the NAV. If False, NAV is not calculated and function returns a tuple of [sum_fv_distributions, sum_fv_calls] 
                    (allows for manual completion of the PME formula using appropriate NAV value)

    Returns:
        The KS-PME metric given the inputed index
    """
    _dates_index = pd.to_datetime(dates_index)  
    df_cf = pd.concat([pd.to_datetime(dates_cashflows),cashflows,cashflows_type], axis=1)
    df_cf.columns = ["Date", "Amount","Type"]
    #first let us run through the cashflow data and sum up all of the calls and distributions
    sum_fv_distributions = 0
    sum_fv_calls = 0
    for idx, cf in df_cf.iterrows():
        # Let us find the closest index value to the current date
        index_value = index[_dates_index == nearest(_dates_index, cf["Date"])].iloc[0]
        if cf["Type"] == "Distribution":
            sum_fv_distributions = sum_fv_distributions + abs(cf["Amount"])/index_value
        elif cf["Type"] == "Capital Call":
             sum_fv_calls = sum_fv_calls + abs(cf["Amount"])/index_value          
    #Now, let us also consider the nav
    if auto_NAV == True:
        #Let us find the nav
        NAV_record = df_cf[df_cf['Type']=='Value'].sort_values("Date", ascending = False).head(1)
        index_value = index[_dates_index == nearest(_dates_index, NAV_record["Date"].iloc[0])].iloc[0]
        discounted_NAV = (NAV_record['Amount'].iloc[0]/index_value) * NAV_scaling
        #return according to the KSPME formula 
        return (sum_fv_distributions+discounted_NAV)/sum_fv_calls
    else:
        return [sum_fv_distributions,sum_fv_calls]

#Direct Alpha
def Direct_Alpha_PME(dates_cashflows, cashflows, cashflows_type, dates_index, index, NAV_scaling = 1):
    """Calculates the Direct Alpha PME. Designed for plug & play with Preqin data.

    Args:
        dates_cashflows: An ndarray the dates corresponding to cashflows. 
        cashflows: An ndarray of the cashflow amounts (sign does not matter)
        cashflows_type: Accepts three types [Distribution \ Capital Call \ Value]
        dates_index: Ndarray of dates for the index, same logic as cashflows. 
        index: The index levels corresponding to the dates
        NAV_scaling: Coefficient which can be used to scale the NAV amount (so as to counteract systemic mispricing)
        auto_NAV: Toggle for automatic handling of the NAV. If False, NAV is not calculated and function returns a tuple of [sum_fv_distributions, sum_fv_calls] 
                    (allows for manual completion of the PME formula using appropriate NAV value)

    Returns:
        The Direct Alpha metric given the inputed index
    """
    #First let us grab the discount table - using wrapper function defined above       
    df_cf = discount_table(dates_cashflows, cashflows, cashflows_type, dates_index, index, NAV_scaling = NAV_scaling) 
    #Now let us calculate the IRR
    irr = xirr2(df_cf['Discounted'], pd.to_datetime(df_cf['Date']))
    direct_alpha = np.log(1+irr)
    return direct_alpha

#PME+
def PME_PLUS(dates_cashflows, cashflows, cashflows_type, dates_index, index, return_alpha = 1, NAV_scaling = 1):
    ''' Returns the alpha as generated by the PME+ method. I.e. (IRR - PME+_IRR)
    '''
    #First let us grab the discount table - using wrapper function defined above       
    df_cf = discount_table(dates_cashflows, cashflows, cashflows_type, dates_index, index, NAV_scaling = NAV_scaling) 
    
    sum_fv_distributions = 0
    sum_fv_calls = 0
    for idx, cf in df_cf.iterrows():
        if cf["Type"] == "Distribution":
            sum_fv_distributions = sum_fv_distributions + abs(cf["Amount"]) * cf["FV_Factor"]
        elif cf["Type"] == "Capital Call":
             sum_fv_calls = sum_fv_calls + abs(cf["Amount"]) * cf["FV_Factor"]
    
    # Check the NAV value
    df_cf['PME_PLUS IRR'] = 0
    if (df_cf.tail(1).iloc[0]['Type'] == "Value"):
        NAV_value = df_cf.tail(1).iloc[0]['Discounted']
        df_cf['PME_PLUS IRR'].iloc[-1] = NAV_value
    else:
        NAV_value = 0 
    scaling_factor = (sum_fv_calls - NAV_value) / sum_fv_distributions

    #Now lets add the PME IRR calc collumn
    for idx, cf in df_cf.iterrows():
        if cf["Type"] == "Distribution":
            df_cf.loc[idx,'PME_PLUS IRR'] = abs(cf["Amount"]) * scaling_factor
        elif cf["Type"] == "Capital Call":
            df_cf.loc[idx,'PME_PLUS IRR'] = - abs(cf["Amount"])
            
    pme_plus =xirr2(df_cf['PME_PLUS IRR'], pd.to_datetime(df_cf['Date']))
    irr = xirr2(df_cf['Pre-Discounted'],pd.to_datetime(df_cf['Date']))
    
    #check if return benchmark or the alpha
    if return_alpha == True:
        return float(irr-pme_plus)
    else:
        return float(pme_plus)
    
    
#MIRR 
def MIRR(dates_cashflows, cashflows, cashflows_type, reinvestment_rate, financing_rate, first_date = 0, NAV_scaling = 1):  
    ''' Calculates the Modified IRR (MIRR)
    '''
    df_cf = pd.concat([pd.to_datetime(dates_cashflows),cashflows,cashflows_type], axis=1)
    df_cf.columns = ["Date", "Amount", "Type"]
    df_cf = df_cf.sort_values('Date')
    df_cf = df_cf.reset_index(drop=True)
    #Grab the 
    if first_date == 0:
        start_date = df_cf['Date'].iloc[0]
    else:
        start_date = first_date
    end_date = df_cf['Date'].iloc[-1] 
    #Calculate the PV and FVs
    sum_pv_calls = 0
    sum_fv_dist = 0
    for idx, cf in df_cf.iterrows():
        if cf["Type"] == "Distribution":
            sum_fv_dist = sum_fv_dist + TVM(cf['Amount'],cf['Date'], end_date, reinvestment_rate)
        elif cf["Type"] == "Capital Call": 
            sum_pv_calls = sum_pv_calls + TVM(-cf['Amount'],cf['Date'], start_date, financing_rate)         
    #Grab the NAV if available
    if(df_cf[(df_cf['Type']=='Value') & (df_cf['Amount']==0)].empty): #Checks if liquidated by looking at 0 valuations
        NAV_record = df_cf[df_cf['Type']=='Value'].sort_values("Date", ascending = False).head(1).copy()['Amount'].iloc[0]
    else:  
        NAV_record = 0  
    #Get the Time delta (n)
    time_delta = ((pd.to_datetime(end_date) - pd.to_datetime(start_date)) / np.timedelta64(1, 'D')).astype(int)/365
    #Return the discounted CAGR
    return (((sum_fv_dist + NAV_record)/sum_pv_calls)**(1/time_delta))-1


### Index-adjustment functions below:
    
# Adjust Beta
def BETA_ADJ_INDEX(dates_index, index, rf_date, rf_rate, beta):  
    ''' Allows us to re-lever an index for a new beta value using a risk-free rate timeseries. 
        - Applies CAPM formula as follows: r_levered = r_f + beta(r_index - r_f)
        - Scales the r_f using the timedelta between dates (arithmetic, not continous) automatically
     Args:
       
        dates_index: Ndarray of dates for the index
        index: The index levels corresponding to the dates
        rf_date: Ndarray of the risk-free dates
        rf_rate: The rate corresponding to the dates
        beta: The beta level to be simulated

    Returns:
        A DataFrame (Date|Index_new|Index_original|Beta), with the adjusted index
    '''
    df_index = pd.concat([pd.to_datetime(dates_index),index], axis=1)
    df_index.columns = ["Date", "Index"]
    df_index = df_index.sort_values('Date')
    df_index = df_index.reset_index(drop=True)
     
    df_rf = pd.concat([pd.to_datetime(rf_date),rf_rate], axis=1)
    df_rf.columns = ["Date", "Value"]
       
    df_index['Delta_amount'] =  ( df_index['Index'] / df_index['Index'].shift(1) ) - 1
    df_index['Delta_time_days'] =  ((df_index['Date'] - df_index['Date'].shift(1)) / np.timedelta64(1, 'D'))
    df_index['Delta_time_years'] = df_index['Delta_time_days'] / 365
    df_index['Closest_rf'] = 0
    df_index['Adjusted_delta'] = 0
    df_index['Index_beta_adjusted'] = 100
    
    for idx, record in df_index.iterrows():
        #assign the closest rf
        closest_rf = float(df_rf.loc[df_rf["Date"] == nearest(pd.to_datetime(df_rf["Date"]), pd.to_datetime(record['Date']), 0), 'Value'])
        try:
            df_index.loc[idx,'Closest_rf'] = closest_rf
        except:
            pass
        #now lets compute the adjusted change
        closest_rf_adj = closest_rf * record['Delta_time_years']
        adjusted_delta_amount = closest_rf_adj + beta*(record['Delta_amount'] - closest_rf_adj)
        df_index.loc[idx,'Adjusted_delta'] = adjusted_delta_amount
        #finally, let us recompute a new index
        if(record['Index'] != 100 and int(idx) != 0):
            df_index.loc[idx,'Index_beta_adjusted'] = (1+df_index.loc[idx,'Adjusted_delta']) * df_index.loc[int(idx)-1,'Index_beta_adjusted']
    
    #clean up, produce new DF, and return it
    df_index_adjusted = pd.concat([df_index['Date'],df_index['Index_beta_adjusted'], df_index['Index']], axis=1)
    df_index_adjusted['Beta']=beta
    df_index_adjusted = df_index_adjusted.reset_index(drop=True)
    df_index_adjusted.columns = ['Date','Index_new','Index_original','Beta']
    return df_index_adjusted#df_index
