
import math as math #import library for math functions

def nett_radiation(total_radiation):
    """Net solar radiation (MJ/m2) which is total incomming radiation less that which is reflected.
    
    This function is taken from Jamieson P. 1982. Comparision of methods of estimating maximum 
    evapotranspiration from a barley crop. New Zealand Journal of Science, 25: 175-181.
    
    Args:
        total_radiation: Total incoming solar radiation (Units MJ/M2/day)
    
    Returns:
        Value of net solar radiation
    """
    _ret = None
    _ret = - 0.25 + 0.59 * total_radiation
    return _ret

def saturated_vapor_pressure(temperature):
    """This is the vapour pressure (in mbar) that the airs capacity to absorb water vapor is saturated.
    
    It increases exponentially with temperature.  The equation used here is from:
    Jenson ME, Burman RD, Allen RG. 1990. Evapotranspiration and irrigation requirements: a manual. 
    New York, U.S.A: American Society of Civil Engineers.
    
    Args:
        temperature is the temperature of the air (units degrees C)
        
    Returns:
        Value of saturated vapor pressure
    """
    _ret = None
    _ret = 0.611 * math.exp(( 17.27 * temperature )  /  ( temperature + 237.3 )) * 10
    return _ret

def vapor_pressure_deficit(temperature, vapor_pressure):
    """This is the difference (in mbar) between the current vapour presure and the saturated vapor pressure
    at the current air temperature
    
    Args:
        Temperature is Air temperature (units degrees C)
        vapor_pressure is the air vapor pressure (in mbar)
    
    """
    _ret = None
    saturated_vp = saturated_vapor_pressure(temperature)
    _ret = saturated_vp - vapor_pressure
    return _ret

#Slope of the saturated vapor pressure line at give temperature (kPa).  This is a different way of calculating Hslope
def saturated_vapor_pressure_slope(temperature):
    """#Slope of the saturated vapour presure curve against temperature for a given temp (kPa/oC)
    Args:
        Temperature is Air temperature (units degrees C)
    """
    _ret = None
    _ret = ( 4098 * 0.6108 * math.exp(( 17.27 * temperature )  /  ( temperature + 237.3 )) )  /  ( ( temperature + 237.3 )  ** 2 )
    return _ret

def lamda(temperature):
    """#latent heat of vapourisation (MJ/kg). 
    
    ET calculations solve an energy balance to work out how much energy is being removed from the system 
    by evaporation.  This is given by the latent heat of vapourisation.
    We need to divide latent heat flux (MJ) by LAMDA to convert to mm of water evaporated
    
    Args:
        temperature is air temperature (units degrees C)
    
    """
    _ret = None
    _ret = 2.5 - 0.002365 * temperature
    return _ret

def gama(temperature):
    """#The phycometric constant (kPa/oK)
    
    Args:
        temperature is air temperature (units degrees C)
    
    """
    _ret = None
    Cp = 0.001013
    p = 101
    e = 0.622
    l = lamda(temperature)
    _ret = ( Cp * p )  /  ( e * l )  
    return _ret

def PenmanEO(radiation, temperature, wind, vp, rad_type):
    """#Penman Evapotranspiration potential (mm/day)
    
    The amount of water that will be transpired by a short, actively growing area of crop that is
    fully covering the ground.  This is the formulation given by French BK, Legg BJ. 1979. 
    Rothamsted irrigation 1964-76. Journal of Agricultural Science, U.K, 92: 15-37.
    
    Args:
        radiation net radiaion (Units MJ/M2/day)
        temperature the mean air temperature for the day measured in a stevenson screen at 1.2m height (Units Degrees celcius)
        wind is mean wind speed (units m/s)
        vp is the vapor pressure of the air at 1.2m height (units kPa)
        rad_type is "net" or "total".  Function will convert radiation to net if given in total
    
    """
    _ret = None
    if rad_type == 'net':
        Rad = radiation
    else:
        Rad = nett_radiation(radiation)
    D = saturated_vapor_pressure_slope(temperature) * 10 # convert value D to mbar
    l = lamda(temperature)
    G = gama(temperature) * 10 # convert to mbar
    p = AirDensity(temperature)
    VPD = vapor_pressure_deficit(temperature, vp*10)
    Wind = wind * 60 *60 *24 / 1000
    Ea = 0.27 * VPD *  ( 1 + Wind / 160 )
    _ret = ( D * Rad / l + G * Ea )  /  ( D + G )
    return _ret

def Priestly_TaylorEO(radiation, temperature, alpha, rad_type):
    """Priestly Taylor evapotranspiration potential (mm/day)
    
    The amount of water that will be transpired by a short, actively growing area of crop that is
    fully covering the ground.  This is the formulation given by Priestly CHB, Taylor RJ. 1972. 
    On the assessment of surface heat flux and evaporation using large-scale parameters. Monthly Weather Review, 100: 81-92.
    
    Args:
        radiation is total incomming solar radiaion (Units MJ/M2/day)
        temperature the mean air temperature for the day measured in a stevenson screen at 1.2m height (Units Degrees celcius)
        alpha is an empirical coefficient representing advective effects.  A value of 1.3 is typical       
    
    """
    _ret = None
    if rad_type == 'net':
        Rad = radiation
    else:
        Rad = nett_radiation(radiation) # in MJ/m2
    D = saturated_vapor_pressure_slope(temperature) # in Kpa
    l = lamda(temperature)
    G = gama(temperature) # in kPa/oC
    PTe = alpha * D * Rad /  ( D + G )
    _ret = PTe / l
    return _ret

def AirDensity(temperature):
    """Density of air (kg/m3)
    
    Args:
        temperature is air temperature (units degrees C)

    """
    _ret = None
    p = 101
    GC = 287
    _ret = ( 1000 * p )  /  ( GC * 1.01 *  ( 273.16 + temperature ) )
    return _ret



def NetRadiation(radiation, Tmean, VapourPressure, Lattitude, DOY, HourDuration, Time, LongitudeTZ, LongitudeML, albedo):
    """Net solar radiation (MJ/m2) at the crop surface.
    
    This is total incomming radiation less that which is reflected.  
    
    Reference: ASCE-EWRI. 2005. The ASCE Standardized Reference Evapotranspiration Equation.  
    Report of the Task Committee on Standardization of Reference Evapotranspiration.
    
    Args:
        radiation is the total incomming solar radiation measured by a pyranometer for the period (Units MJ/m2)
        Tmean is the mean temperature for the period measured in a Stevenson screen at 1.2 m height (degrees C)
        VapourPressure is the mean vapor pressure for the period measured in a Stevenson screen at 1.2 m height (Units kPa)
        Lattitude (units degrees)
        DOY is day of year 1 Jan = 1
        HourDuration is the duration of the calculation period (Units hours) (range 0-24).  If HourDuration is set to less than 24 the following parameters also need to have appropriate values
        Time is the time (Units hours) (range 0-24) of the mid point of the measurement period
        LongitudeTZ is the longitude at the centre of the time zone where measurements are taken (Units degrees)
        LongitudeML is the longitude of the location of measurement (Units degrees)
        albedo is the proportion of radiation that the surface reflects back to the sky

    """
    
    _ret = None
    if radiation <= 0:
        RShortWave = 0
        RLongWave = 0
    else:
        RShortWave = NetSolarRadiation(radiation, albedo)
        RLongWave = NetLongwaveRadiation(radiation, Tmean, VapourPressure, Lattitude, DOY, HourDuration, Time, LongitudeTZ, LongitudeML)
    if RLongWave < 0:
        RLongWave = 0
    if RShortWave > 0 and RLongWave > 0:
        _ret = RShortWave - RLongWave
    else:
        _ret = 0
    return _ret

def NetSolarRadiation(radiation, albedo):
    """Solar (Short wave) radiation (MJ/m2) measured with a pyranometer and adjusted for reflection (albedo) and Net Solar Radiation (radiation):
    
    Args:
        radiation is Net solar radiation measured with a pyranometer (units Mj/m2/time period)
        albedo is the proportion of incoming radiation that is absorbed by the crop surface (units 0-1)
    """
    _ret = None
    _ret = ( 1 - albedo )  * radiation
    return _ret

def NetLongwaveRadiation(radiation, Tmean, VapourPressure, Lattitude, DOY, HourDuration, Time, LongitudeTZ, LongitudeML):
    """Solar radiation (MJ/m2) absorbed by the crop and lost again to the atmosphere and space by longwave radiation

    Args:
        radiation is the total incomming solar radiation measured by a pyranometer for the period (Units MJ/m2)
        Tmean is the mean temperature for the period measured in a Stevenson screen at 1.2 m height (degrees C)
        VapourPressure is the mean vapor pressure for the period measured in a Stevenson screen at 1.2 m height (Units kPa)
        Lattitude (units degrees)
        DOY is day of year 1 Jan = 1
        HourDuration is the duration of the calculation period (Units hours) (range 0-24).  If HourDuration is set to less than 24 the following parameters also need to have appropriate values
        Time is the time (Units hours) (range 0-24) of the mid point of the measurement period
        LongitudeTZ is the longitude at the centre of the time zone where measurements are taken (Units degrees)
        LongitudeML is the longitude of the location of measurement (Units degrees)

    """
    _ret = None
    RClearSky = ClearSkyRadiation(Lattitude, DOY, HourDuration, Time, LongitudeTZ, LongitudeML)
    if RClearSky > 0:
        sb = 0.000000004903 * HourDuration / 24
        a = ( Tmean + 273.16 )  ** 4
        b = 0.34 - 0.14 * VapourPressure ** 0.5
        SSoRatio = radiation / RClearSky
        if SSoRatio > 1:
            SSoRatio = 1
        c = 1.35 * SSoRatio - 0.35
        if c < 0.05:
            c = 0.05
        elif c > 1:
            c = 1
        _ret = sb * a * b * c
    else:
        _ret = 0
    return _ret

def ClearSkyRadiation(Lattitude, DOY, HourDuration, Time, LongitudeTZ, LongitudeML):
    """Radiation (MJ/m2) assuming no cloud cover
    
    Args:
        Lattitude (units degrees)
        DOY is day of year 1 Jan = 1
        HourDuration is the duration of the calculation period (Units hours) (range 0-24).  If HourDuration is set to less than 24 the following parameters also need to have appropriate values
        Time is the time (Units hours) (range 0-24) of the mid point of the measurement period
        LongitudeTZ is the longitude at the centre of the time zone where measurements are taken (Units degrees)
        LongitudeML is the longitude of the location of measurement (Units degrees)

    """
    _ret = None
    _ret = ( 0.75 + 0.00002 * 17 )  * ExtraterestialRadiation(Lattitude, DOY, HourDuration, Time, LongitudeTZ, LongitudeML)
    return _ret

def ExtraterestialRadiation(Latitude, DOY, HourDuration, Time, LongitudeTZ, LongitudeML):
    """Radiation (MJ/m2) at the top of the atmosphere

    Args:
        Lattitude (units degrees)
        DOY is day of year 1 Jan = 1
        HourDuration is the duration of the calculation period (Units hours) (range 0-24).  If HourDuration is set to less than 24 the following parameters also need to have appropriate values
        Time is the time (Units hours) (range 0-24) of the mid point of the measurement period
        LongitudeTZ is the longitude at the centre of the time zone where measurements are taken (Units degrees)
        LongitudeML is the longitude of the location of measurement (Units degrees)

    """
    
    _ret = None
    DR = InverseRelativeDistance(DOY)
    SD = SolarDecimation(DOY)
    Lat = Latitude*math.pi/180
    
    if HourDuration == 24 :
        SH = SunsetHourAngel(DOY, Latitude)
        _ret = (24/math.pi)*4.92*DR*(SH*math.sin(Lat)*math.sin(SD)+math.cos(Lat)*math.cos(SD)*math.sin(SH))
    else:
        SH = SunsetHourAngel(DOY, Latitude)
        SH1 = SHMidPoint(DOY, Time, LongitudeTZ, LongitudeML) - math.pi *  HourDuration / 24
        if SH1< -SH:
            SH1=-SH
        elif SH1>SH:
            SH1=SH
        
        SH2 = SHMidPoint(DOY, Time, LongitudeTZ, LongitudeML) + math.pi * HourDuration / 24
        if SH2< -SH:
            SH2= -SH
        elif SH2>SH:
            SH2=SH
        
        if SH1>SH2:
            SH1=SH2
        _ret = (12/math.pi)*4.92*DR*((SH2-SH1)*math.sin(Lat)*math.sin(SD)+math.cos(Lat)*math.cos(SD)*(math.sin(SH2)-math.sin(SH1)))
    return _ret
  
def SHMidPoint(DOY, Time, LongitudeTZ, LongitudeML):
    _ret = None
    b = ( 2 * math.pi *  ( DOY - 81 ) )  / 364
    SC = 0.1645 * math.sin(2 * b) - 0.1255 * math.cos(b) - 0.025 * math.sin(b)
    _ret = ( math.pi / 12 )  *  ( ( Time + 0.06667 *  ( LongitudeTZ - LongitudeML )  + SC )  - 12 )
    return _ret

def InverseRelativeDistance(DOY):
    _ret = None
    _ret = 1 + 0.033 * math.cos(( 2 * math.pi )  / 365 * DOY)
    return _ret

def SunsetHourAngel(DOY, Lattitude):
    _ret = None
    Lat = Lattitude * math.pi / 180
    _ret = math.acos(- math.tan(Lat) * math.tan(SolarDecimation(DOY)))
    return _ret

def SolarDecimation(DOY):
    _ret = None
    _ret = 0.409 * math.sin(( 2 * math.pi )  / 365 * DOY - 1.39)
    return _ret


def TsTaUL(Ra, Icul, Rn, Q, Cp):
    """Theoretical Upper limit to the difference between air and canopy temperature (oC)

    Reference Jackson etal 1988.  A reexaminatino of the crop water stress index.  Irrigation Sci 9:309-317
    
    Args:
        Ra is aerodynamic resistance (s/m)
        Icul is and interception factor to account of net radiation that goes into ground stored energy.
        Rn is nett radiation in W/m2
        Q is the density of air (kg/m3)
        Cp is the heat capacity of air (J/kg/oC)
    """
    _ret = None
    _ret = ( Ra * Icul * Rn )  /  ( Q * Cp )
    return _ret

def TsTaLL(Ra, Icll, Rn, Q, Cp, Gamma, Delta, SatVP, VP):
    """Theoretical Lower limit to the difference between air and canopy temperature (oC)
    
    Reference Jackson etal 1988.  A reexaminatino of the crop water stress index.  Irrigation Sci 9:309-317
    
    Args:
        Ra is aerodynamic resistance (s/m)
        Icll
        Rn is nett radiation in W/m2
        Q is the density of air (kg/m3)
        Cp is the heat capacity of air (J/kg/oC)
        Gamma is the phycometric constant
        Delta is the slope of the saturated vapor pressure curve at a given temperature
        SatVP is the saturated vapor pressure
        VP is the vapor pressure
    """
    _ret = None
    a = ( Ra * Icll * Rn )  /  ( Q * Cp )
    b = Gamma /  ( Gamma + Delta )
    c = ( SatVP - VP )  /  ( Delta + Gamma )
    _ret = a * b - c
    return _ret

def Ra(WindSpeed, Zu, h):
    """Aerodynamic resistance based on wind speed as described by Maes and Stepp 2012.  Includes an empurical adjustment
    to account for the effects of buoynacy.
    
    Original Reference: Thom and Oliver, 1977.  On Penmans equation for estimating regaional evaporation.  J Q R Meteorological Soc 98: 124
    
    Args:
        WindSpeed in m/s
        Zu in m is the hight that wind speed and temperature are measured 
        and is 1.2 m for a standard met station
        h in m is the height of the canopy 
    """
    d = (0.63*h)  # is the zero displacement height which is a complex function of canopy height and archicture
    Zom = (0.13*h)  # is the roughness length of momenum and is also influenced by canopy height and archicture
    LN = math.log((Zu-d)/Zom)
    if (WindSpeed == 0.0):
        _ret = 0.0
    else:
        _ret = math.pow(LN,2)/(0.16*WindSpeed)#4.72*math.pow(LN,2)/(1+0.54*WindSpeed)
    return _ret

def RaLAI(WindSpeed, Zu, h, LAI):
    """Aerodynamic resistance based on wind speed as described by Maes and Stepp 2012
    
    Original Reference: Thom and Oliver, 1977.  On Penmans equation for estimating regaional evaporation.  J Q R Meteorological Soc 98: 124
    
    Args:
        WindSpeed in m/s
        Zu in m is the hight that wind speed and temperature are measured 
        and is 1.2 m for a standard met station
        h in m is the height of the canopy 
    """
    d = h*(1-(2/LAI)*(1-math.exp(-LAI/2)))  # from Colaizzi, P.D., Evett, S.R., Howell, T.A. and Tolk, J.A., 2004. Comparison of aerodynamic 413 and radiometric surface temperature using precision weighing lysimeters. In: W. Gao 14 and D.R. Shaw (Editors), Remote Sensing and Modeling of Ecosystems for 415 Sustainability. Proceedings of the Society of Photo-Optical Instrumentation Engineers 416 (Spie). Spie-Int Soc Optical Engineering, Bellingham, pp. 215-229.
    Zom = h*math.exp(-LAI/2)*(1-exp(-LAI/2))  # is the roughness length of momenum and is also influenced by canopy height and archicture
    LN = math.log((Zu-d)/Zom)
    _ret = 4.72*math.pow(LN,2)/(1+0.54*WindSpeed)
    _ret
    return _ret

def PenmanMonteith(Rn, Ta, RH, Uz, ra, rs, duration):
    """Penman Monteith Evapotranspiration equation
    
    Original Reference: https://www.kimberly.uidaho.edu/water/asceewri/ASCE_Standardized_Ref_ET_Eqn_Phoenix2000.pdf
        
    Args:
        Rn is net radiation [MJ/m2]
        Ta is mean air temperature [oC]
        RH is relative humidity %
        Uz wind speed at height z [m s-1]
        Ra aerodynamic resistance [s m-1]
        Rs bulk surface resistance from a transpiring crop [s m-1] 70 for reference crop
        duration of the calculation period in seconds
        
    """
    Pa = AirDensity(Ta)
    D = saturated_vapor_pressure_slope(Ta)
    Cp = 0.001013 #Heat capacity of air MJoule/kg/oC
    G = gama(Ta)
    L = lamda(Ta)
    SatVP = saturated_vapor_pressure(Ta)/10  #divide by 10 to convert from mbar to kpa
    VP = SatVP * RH/100 
    VPD = SatVP - VP
    Numerator = D*Rn + duration * Pa * Cp * (VPD/ra)
    Denominator = D + G*(1+(rs/ra))
    _ret = Numerator/Denominator
    _ret = _ret * 1/L  #convert from energy to mm of water
    return _ret

def PotB(temperature,alpha):
    """returns a bowen energy ration for potential conditions
    Idea developed by Pete Jamieson in 2017
    args:
        temperature in oC
        alpha is a coefficient (0-infinity but typically less than 2)
    """
    gamma = gama(temperature)
    s = saturated_vapor_pressure_slope(temperature)
    _ret = (gamma + s * (1 - alpha)) / (alpha * s)
    return _ret
