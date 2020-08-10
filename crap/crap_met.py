WG = WeatherGenerator(lat, lon)

doy = 180.0
sw_rad_day = 14.0 # mj m-2 d-1
par_day = sw_rad_day * WG.SW_2_PAR_MJ


hours = np.arange(48) / 2.0

# Hacky.com just divide a MAP up equally (stochastically in time) across
# the year
N = 365 * 48
p = 0.8
map = 400.0
chance_of_rain = np.random.choice(a=[False, True], size=N, p=[p, 1-p])
nn=0
for i in range(len(chance_of_rain)):
    if chance_of_rain[i]:
        nn+=1

all_rain = np.zeros(365*48)
for i in range(len(chance_of_rain)):
    if chance_of_rain[i]:
        all_rain[i] = map / float(nn)

ndays = 365 * 48
all_par = np.zeros(0)
all_tair = np.zeros(0)
all_vpd = np.zeros(0)

check = []
for doy in np.arange(1, 366):

    if doy < 60 or doy > 300:
        tmin = 10.0
        tmax = 34.0
        vpd09 = 1.4
        vpd09_next = 1.8
        vpd15 = 4.3
        vpd15_prev = 3.4
    else:
        tmin = -5.0
        tmax = 19.0
        vpd09 = 1.4
        vpd09_next = 0.8
        vpd15 = 2.3
        vpd15_prev = 2.7

    par = WG.estimate_dirunal_par(par_day, doy)
    tday = WG.estimate_diurnal_temp(doy, tmin, tmax)
    vpd = WG.estimate_diurnal_vpd(vpd09, vpd15, vpd09_next, vpd15_prev)


    #check.append(np.max(tday))
    #check.append(np.max(par))
    check.append(np.max(vpd))

    all_par = np.append(all_par, par)
    all_tair = np.append(all_tair, tday)
    all_vpd = np.append(all_vpd, vpd)

#plt.plot(check)
#plt.show()

year = 2001 # not a leap
ndays = 365
nx = 48
day = np.repeat(np.arange(1, ndays+1), nx)

Ca = np.ones(len(day)) * 400 # umol mol-1
press = np.ones(len(day) ) * 101325.0 # kPa
wind = np.ones(len(day) ) * 5.0 # m s -1
lat = np.ones(len(day)) * lat
lon = np.ones(len(day)) * lon
oyear = np.ones(len(day)) * year

met = pd.DataFrame({'year':oyear, 'day':day, 'par':all_par, 'tair':all_tair,
                    'vpd':all_vpd, 'precip':all_rain, 'press':press,
                    'wind':wind, 'ca':Ca, 'lat':lat, 'lon':lon})
