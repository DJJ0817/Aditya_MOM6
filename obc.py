import xarray as xr
import xesmf
import matplotlib.pyplot as plt
import bottleneck
import numpy as np
import subprocess as sp
import os
import glob
import cartopy.crs as ccrs
from netCDF4 import Dataset


def open_grid_uv(path,decode_times=False):
#    grid={}
    grid=xr.open_dataset(path,decode_times=False)
    grid=grid.drop_dims(['xt_ocean','yt_ocean'])
    grid=grid.rename_dims({'xu_ocean':'lon'})
    grid=grid.rename_dims({'yu_ocean':'lat'})
    grid=grid.rename_vars({'xu_ocean':'lon'})
    grid=grid.rename_vars({'yu_ocean':'lat'})
    return grid

def open_grid_tr(path,decode_times=False):
#    grid={}
    grid=xr.open_dataset(path,decode_times=False)
    grid=grid.drop_dims(['xu_ocean','yu_ocean'])
    grid=grid.rename_dims({'xt_ocean':'lon'})
    grid=grid.rename_dims({'yt_ocean':'lat'})
    grid=grid.rename_vars({'xt_ocean':'lon'})
    grid=grid.rename_vars({'yt_ocean':'lat'})
    return grid


path_regional_grid = '/lustre/jjd0817/copy/hgrid_Indian_Ocean_0125/ocean_hgrid.nc'
regional_grid      = xr.open_dataset(path_regional_grid)

path_model_data    = '/lustre/jjd0817/MOM6-examples/Indian/INPUT/soda3.4.2_mn_ocean_reg_2014.nc'
#model_data_uv      = xr.open_dataset(path_model_data,decode_times=False)

model_data_uv      = open_grid_uv(path_model_data)
model_data_tr      = open_grid_tr(path_model_data)


south = xr.Dataset()
south['lon'] = regional_grid['x'].isel(nyp=0)
south['lat'] = regional_grid['y'].isel(nyp=0)
east = xr.Dataset()
east['lon'] = regional_grid['x'].isel(nxp=-1)
east['lat'] = regional_grid['y'].isel(nxp=-1)
north = xr.Dataset()
north['lon'] = regional_grid['x'].isel(nyp=-1)
north['lat'] = regional_grid['y'].isel(nyp=-1)

us = regional_grid.x.shape
nyp=us[0]
nxp=us[1]

regrid_north_uv = xesmf.Regridder(model_data_uv, north, 'nearest_s2d', locstream_out=True, periodic=False, filename='regrid_north_uv.nc')
regrid_south_uv = xesmf.Regridder(model_data_uv, south, 'nearest_s2d', locstream_out=True, periodic=False, filename='regrid_south_uv.nc')
regrid_east_uv  = xesmf.Regridder(model_data_uv, east, 'nearest_s2d', locstream_out=True, periodic=False, filename='regrid_east_uv.nc')
regrid_north_tr = xesmf.Regridder(model_data_tr, north, 'nearest_s2d', locstream_out=True, periodic=False, filename='regrid_north_tr.nc')
regrid_south_tr = xesmf.Regridder(model_data_tr, south, 'nearest_s2d', locstream_out=True, periodic=False, filename='regrid_south_tr.nc')
regrid_east_tr  = xesmf.Regridder(model_data_tr, east, 'nearest_s2d', locstream_out=True, periodic=False, filename='regrid_east_tr.nc')



u_south = regrid_south_uv(model_data_uv['u'])
v_south = regrid_south_uv(model_data_uv['v'])
u_east  = regrid_east_uv(model_data_uv['u'])
v_east  = regrid_east_uv(model_data_uv['v'])
u_north = regrid_north_uv(model_data_uv['u'])
v_north = regrid_north_uv(model_data_uv['v'])
#ds_uv_south = xr.Dataset({'u':u_south,'v':v_south},coords={'lon':south.lon,'lat':south.lat,'z_l':model_data_uv.st_ocean})
ds_uv_south = xr.Dataset({'u':u_south,'v':v_south})
ds_uv_south.time.attrs['calendar']='gregorian'
fnam        ='uv_south.nc'
ds_uv_south.to_netcdf(fnam,unlimited_dims='time',format='NETCDF3_CLASSIC')
ds_uv_east  = xr.Dataset({'u':u_east,'v':v_east})
fnam        ='uv_east.nc'
ds_uv_east.time.attrs['calendar']='gregorian'
ds_uv_east.to_netcdf(fnam,unlimited_dims='time',format='NETCDF3_CLASSIC')
ds_uv_north  = xr.Dataset({'u':u_north,'v':v_north})
fnam        ='uv_north.nc'
ds_uv_north.time.attrs['calendar']='gregorian'
ds_uv_north.to_netcdf(fnam,unlimited_dims='time',format='NETCDF3_CLASSIC')

temp_south  = regrid_south_tr(model_data_tr['temp'])
salt_south  = regrid_south_tr(model_data_tr['salt'])
ds_tr_south = xr.Dataset({'temp':temp_south,'salt':salt_south})
ds_tr_south.time.attrs['calendar']='gregorian'
fnam        ='tracers_south.nc'
ds_tr_south.to_netcdf(fnam,unlimited_dims='time',format='NETCDF3_CLASSIC')
temp_east   = regrid_east_tr(model_data_tr['temp'])
salt_east   = regrid_east_tr(model_data_tr['salt'])
ds_tr_east  = xr.Dataset({'temp':temp_east,'salt':salt_east})
ds_tr_east.time.attrs['calendar']='gregorian'
fnam        ='tracers_east.nc'
ds_tr_east.to_netcdf(fnam,unlimited_dims='time',format='NETCDF3_CLASSIC')
temp_north   = regrid_north_tr(model_data_tr['temp'])
salt_north   = regrid_north_tr(model_data_tr['salt'])
ds_tr_north  = xr.Dataset({'temp':temp_north,'salt':salt_north})
ds_tr_north.time.attrs['calendar']='gregorian'
fnam        ='tracers_north.nc'
ds_tr_north.to_netcdf(fnam,unlimited_dims='time',format='NETCDF3_CLASSIC')

ds_ssh = xr.Dataset({'ssh':model_data_tr.ssh},coords={'lon':model_data_tr.lon,'lat':model_data_tr.lat})
ssh_south = regrid_south_tr(ds_ssh['ssh'])
ds_ssh_south = xr.Dataset({'ssh':ssh_south})
ds_ssh_south.time.attrs['calendar']='gregorian'
fnam='ssh_south.nc'
ds_ssh_south.to_netcdf(fnam,unlimited_dims='time',format='NETCDF3_CLASSIC')
ssh_east = regrid_east_tr(ds_ssh['ssh'])
ds_ssh_east = xr.Dataset({'ssh':ssh_east})
ds_ssh_east.time.attrs['calendar']='gregorian'
fnam='ssh_east.nc'
ds_ssh_east.to_netcdf(fnam,unlimited_dims='time',format='NETCDF3_CLASSIC')
ssh_north = regrid_north_tr(ds_ssh['ssh'])
ds_ssh_north = xr.Dataset({'ssh':ssh_north})
ds_ssh_north.time.attrs['calendar']='gregorian'
fnam='ssh_north.nc'
ds_ssh_north.to_netcdf(fnam,unlimited_dims='time',format='NETCDF3_CLASSIC')


############## step 8 ###################
params=[]
params.append({'suffix':'_segment_001','dim0':2,'index':1,'tr_in':'tracers_north.nc','tr_out':'obc_ts_north.nc','uv_in':'uv_north.nc','uv_out':'obc_uv_north.nc','ssh_in':'ssh_north.nc','ssh_out':'obc_ssh_north.nc'})
params.append({'suffix':'_segment_002','dim0':3,'index':2,'tr_in':'tracers_east.nc','tr_out':'obc_ts_east.nc','uv_in':'uv_east.nc','uv_out':'obc_uv_east.nc','ssh_in':'ssh_east.nc','ssh_out':'obc_ssh_east.nc'})
params.append({'suffix':'_segment_003','dim0':2,'index':1,'tr_in':'tracers_south.nc','tr_out':'obc_ts_south.nc','uv_in':'uv_south.nc','uv_out':'obc_uv_south.nc','ssh_in':'ssh_south.nc','ssh_out':'obc_ssh_south.nc'})

for pr in params:
    ds=xr.open_dataset(pr['tr_in'],decode_times=False)
    explon=xr.DataArray(ds.lon.ffill(dim='locations',limit=None).fillna(0.))
#    explon=explon.expand_dims('dim_0',pr['dim0']-2)
    explat=xr.DataArray(ds.lat.ffill(dim='locations',limit=None).fillna(0.))
#    explat=explat.expand_dims('dim_0',pr['dim0']-2)
    zl=ds.temp.st_ocean
    zi=0.5*(np.roll(zl,shift=-1)+zl)
    zi[-1]=6500.
    ds['z_i']=zi
    dz=zi-np.roll(zi,shift=1)
    dz[0]=zi[0]
    ds['dz']=dz
    nt=ds.time.shape[0]
    nx=ds.lon.shape[0]
    dz=np.tile(ds.dz.data[np.newaxis,:,np.newaxis],(nt,1,nx))
    intz = np.arange(0,50,dtype=np.int32)
    intz = intz.astype(np.int32)
    da_dz=xr.DataArray(dz,coords=[('time',ds.time),('st_ocean',intz),('locations',ds.locations)])
    da_dz=da_dz.expand_dims('dim_0',pr['dim0'])
    da_dz_temp=da_dz.rename(st_ocean='nz'+pr['suffix']+'_temp')
    da_dz_salt=da_dz.rename(st_ocean='nz'+pr['suffix']+'_salt')
    da_dz_u=da_dz.rename(st_ocean='nz'+pr['suffix']+'_u')
    da_dz_v=da_dz.rename(st_ocean='nz'+pr['suffix']+'_v')
    ds.time.attrs['modulo']=' '
    da_temp=xr.DataArray(ds.temp.ffill(dim='locations',limit=None).ffill(dim='st_ocean').fillna(0.))
    da_temp=da_temp.expand_dims('dim_0',pr['dim0'])
    da_temp['st_ocean'] = intz
    da_temp=da_temp.rename(st_ocean='nz'+pr['suffix']+'_temp')
    da_salt=xr.DataArray(ds.salt.ffill(dim='locations',limit=None).ffill(dim='st_ocean').fillna(0.))
    da_salt=da_salt.expand_dims('dim_0',pr['dim0'])
    da_salt['st_ocean'] = intz
    da_salt=da_salt.rename(st_ocean='nz'+pr['suffix']+'_salt')
    ds_=xr.Dataset({'temp'+pr['suffix']:da_temp,'salt'+pr['suffix']:da_salt,'lon'+pr['suffix']:explon,'lat'+pr['suffix']:explat,'dz_temp'+pr['suffix']:da_dz_temp,'dz_salt'+pr['suffix']:da_dz_salt})
#    ds_.drop_vars("lon")
#    ds_.drop_vars("lat")
#    tempnow = 'temp'+pr['suffix']
#    saltnow = 'salt'+pr['suffix']
#    ds_[tempnow].attrs['coords']= ' '
#    ds_[saltnow].attrs['coordinates']= 'ny nx'
#    ds_[tempnow].attrs['regrid_method'] = ' '
#    ds_[saltnow].attrs['regrid_method'] = ' '
    if pr['index'] == 1:
        ds_=ds_.transpose(...,'locations')
        ds_['dim_0'] = ds_['dim_0'].astype(np.int32)
        ds_['locations'] = ds_['locations'].astype(np.int32)
        ds_=ds_.rename(dim_0='ny'+pr['suffix'])
        ds_=ds_.rename(locations='nx'+pr['suffix'])
    else:
        ds_['dim_0'] = ds_['dim_0'].astype(np.int32)
        ds_['locations'] = ds_['locations'].astype(np.int32)
        ds_=ds_.rename(dim_0='nx'+pr['suffix'])
        ds_=ds_.rename(locations='ny'+pr['suffix'])
    for v in ds_:
        ds_[v].encoding['_FillValue']=1.e20   
    ds_.to_netcdf(pr['tr_out'],unlimited_dims=('time'))
    ds=xr.open_dataset(pr['uv_in'],decode_times=False)
    ds.time.attrs['modulo']=' '
    da_u=xr.DataArray(ds.u.ffill(dim='locations',limit=None).ffill(dim='st_ocean').fillna(0.))
    da_u=da_u.expand_dims('dim_0',pr['dim0'])
    da_u['st_ocean'] = intz
    da_u=da_u.rename(st_ocean='nz'+pr['suffix']+'_u')
    da_v=xr.DataArray(ds.v.ffill(dim='locations',limit=None).ffill(dim='st_ocean').fillna(0.))
    da_v=da_v.expand_dims('dim_0',pr['dim0'])
    da_v['st_ocean'] = intz
    da_v=da_v.rename(st_ocean='nz'+pr['suffix']+'_v')
    ds_=xr.Dataset({'u'+pr['suffix']:da_u,'v'+pr['suffix']:da_v,'lon'+pr['suffix']:explon,'lat'+pr['suffix']:explat,'dz_u'+pr['suffix']:da_dz_u,'dz_v'+pr['suffix']:da_dz_v})
    if pr['index'] == 1:
        ds_=ds_.transpose(...,'locations')
        ds_['dim_0'] = ds_['dim_0'].astype(np.int32)
        ds_['locations'] = ds_['locations'].astype(np.int32)
        ds_=ds_.rename(dim_0='ny'+pr['suffix'])
        ds_=ds_.rename(locations='nx'+pr['suffix'])
    else:
        ds_['dim_0'] = ds_['dim_0'].astype(np.int32)
        ds_['locations'] = ds_['locations'].astype(np.int32)
        ds_=ds_.rename(dim_0='nx'+pr['suffix'])
        ds_=ds_.rename(locations='ny'+pr['suffix'])
    for v in ds_:
        ds_[v].encoding['_FillValue']=1.e20    
    ds_.to_netcdf(pr['uv_out'],unlimited_dims=('time'))
    ds=xr.open_dataset(pr['ssh_in'],decode_times=False)
    ds.time.attrs['modulo']=' '
    da_ssh=xr.DataArray(ds.ssh.ffill(dim='locations',limit=None).fillna(0.))
    da_ssh=da_ssh.expand_dims('dim_0',pr['dim0']-1)
    ds_=xr.Dataset({'ssh'+pr['suffix']:da_ssh,'lon'+pr['suffix']:explon,'lat'+pr['suffix']:explat})
    if pr['index'] == 1:
        ds_=ds_.transpose(...,'locations')
        ds_['dim_0'] = ds_['dim_0'].astype(np.int32)
        ds_['locations'] = ds_['locations'].astype(np.int32)
        ds_=ds_.rename(dim_0='ny'+pr['suffix'])
        ds_=ds_.rename(locations='nx'+pr['suffix'])
    else:
        ds_['dim_0'] = ds_['dim_0'].astype(np.int32)
        ds_['locations'] = ds_['locations'].astype(np.int32)
        ds_=ds_.rename(dim_0='nx'+pr['suffix'])
        ds_=ds_.rename(locations='ny'+pr['suffix'])
    for v in ds_:
        ds_[v].encoding['_FillValue']=1.e20    
    ds_.to_netcdf(pr['ssh_out'],unlimited_dims=('time'))


# ncatted -a ,u_segment_001,d,, -a ,v_segment_001,d,, -a ,lon_segment_001,d,, -a ,lat_segment_001,d,, -a ,dz_u_segment_001,d,, -a ,dz_v_segment_001,d,, obc_uv_north.nc