import netCDF4 as nc
import math

fname = 'wavewindF.nc'
f = nc.Dataset(fname)
msldata = 'msl'
u10data = 'u10'
v10data = 'v10'
swhdata = 'swh'

msll = f[msldata]
u10 = f[u10data]
v10 = f[v10data]
swhh = f[swhdata]


msl = []
uv10 = []
swh = []

#print(len(msl))
for i in range(1464):
    for j in range(3):
        for k in range(2):
            msl.append(msll[i][j][k])
            uv10.append(math.sqrt(u10[i][j][k]**2 + v10[i][j][k]**2))
            swh.append(swhh[i][j][k])
