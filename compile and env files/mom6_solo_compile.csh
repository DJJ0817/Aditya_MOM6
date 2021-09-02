#!/bin/csh
set root=/iitm5/junjie/MOM6

#template=${root}/src/mkmf/templates/ncrc-intel.mk
#set template=${root}/src/mkmf/templates/nccs-intel.mk
set platform=intel  #intel, gnu
set template=/iitm5/junjie/MOM6/src/mkmf/templates/iitm-intel.mk
set DO_NO_SIS2=0 # 1 if ocean_only; 0 if ice_sea_SIS2
set PART_TESTING=#-git

# Compiling the FMS shared mode

mkdir -p ${root}/build/${platform}/shared/repro/
(cd ${root}/build/${platform}/shared/repro/; rm -f path_names; \
${root}/src/mkmf/bin/list_paths ${root}/src/FMS; \
${root}/src/mkmf/bin/mkmf -t $template -p libfms.a -c " -heap-array -Duse_libMPI -lmpi -lmpich -Duse_netCDF -DSPMD" path_names)
(cd ${root}/build/${platform}/shared/repro/; source ../../env.dt2; make NETCDF=3 REPRO=1 libfms.a -j)

if (${DO_NO_SIS2} == 1) then
  # Compiling MOM6 in ocean-only mode (Without SIS)
  mkdir -p ${root}/build/${platform}/ocean_only/repro/
  cd ${root}/build/${platform}/ocean_only/repro/
  rm -f path_names

  # use executive file list_paths to list all the paths of the targeted file and generate "path_names"
  ${root}/src/mkmf/bin/list_paths ${root}/src/MOM6/{config_src/dynamic,config_src/solo_driver,config_src/external,src/{*,*/*},pkg/GSW-Fortran/{modules,toolbox}}

  # use exective file mkmf to compile all files listed in the "path_names" and generate the ex
  cd ${root}/build/${platform}/ocean_only/repro/
  ${root}/src/mkmf/bin/mkmf -t $template -o '-I../../shared/repro -I../../../../src/MOM6/pkg/CVMix-src/{include,bld/obj} -I../../../../src/MOM6/pkg/GSW-Fortran/modules -I../../../../src/MOM6/pkg/GSW-Fortran/toolbox' -p MOM6 -l '-L../../shared/repro -lfms -L../../../../src/MOM6/pkg/CVMix-src/lib -lcvmix -L../../../../src/MOM6/pkg/CVMix-src/bld/obj' -c '-Duse_libMPI -Duse_netCDF -DSPMD' path_names

  (cd ${root}/build/${platform}/ocean_only/repro/; source ../../env.dt2; make NETCDF=3 REPRO=1 MOM6 -j)
else
  # Compiling MOM6 in MOM6-SIS2 coupled mode
  mkdir -p ${root}/build/${platform}/ice_ocean_SIS2/repro/
  (cd ${root}/build/${platform}/ice_ocean_SIS2/repro/; rm -f path_names; \
  ${root}/src/mkmf/bin/list_paths ../../../../src/MOM6/config_src/{dynamic_symmetric,coupled_driver,external} ../../../../src/MOM6/src/{*,*/*}/ ../../../../src/MOM6/pkg/GSW-Fortran/{modules,toolbox} ../../../../src/{atmos_null,coupler,land_null,ice_param,icebergs,SIS2,FMS/coupler,FMS/include}/)
  cd ${root}/build/${platform}/ice_ocean_SIS2/repro/
#  ${root}/src/mkmf/bin/mkmf -t $template -o '-I../../shared/repro -I../../../../src/MOM6/pkg/CVMix-src/{include,bld/obj} -I../../../../src/MOM6/pkg/GSW-Fortran/modules -I../../../../src/MOM6/pkg/GSW-Fortran/toolbox' -p MOM6 -l '-L../../shared/repro -lfms -L../../../../src/MOM6/pkg/CVMix-src/lib -lcvmix -L../../../../src/MOM6/pkg/CVMix-src/bld/obj' -c '-Duse_libMPI -L/gpfs1/home/Libs/INTEL/OPENMPI/openmpi-2.1.6/lib/ -lmpi -lmpich -Duse_netCDF -DSPMD -Duse_AM3_physics -D_USE_LEGACY_LAND_' path_names

  ${root}/src/mkmf/bin/mkmf -t $template -o ' -I../../shared/repro -I../../../../src/MOM6/pkg/CVMix-src/{include,bld/obj} -I/gpfs1/home/Libs/INTEL/OPENMPI/openmpi-1.8.4/include -I../../../../src/MOM6/pkg/GSW-Fortran/modules -I../../../../src/MOM6/pkg/GSW-Fortran/toolbox' -p MOM6 -l ' -L../../shared/repro -lfms -L../../../../src/MOM6/pkg/CVMix-src/lib -lcvmix -L../../../../src/MOM6/pkg/CVMix-src/bld/obj' -c '-check bounds -traceback -Duse_libMPI -L/gpfs1/home/Libs/INTEL/OPENMPI/openmpi-1.8.4/lib  -heap-array  -Duse_netCDF -DSPMD -Duse_AM3_physics -D_USE_LEGACY_LAND_' path_names
  ## orignal line
  #${root}/src/mkmf/bin/mkmf -t $template -o '-I../../shared/repro -I../../../../src/MOM6/pkg/CVMix-src/{include,bld/obj} -I/gpfs1/home/Libs/INTEL/OPENMPI/openmpi-2.1.6/include/ -I../../../../src/MOM6/pkg/GSW-Fortran/modules -I../../../../src/MOM6/pkg/GSW-Fortran/toolbox' -p MOM6 -l '-L../../shared/repro -lfms -L../../../../src/MOM6/pkg/CVMix-src/lib -lcvmix -L../../../../src/MOM6/pkg/CVMix-src/bld/obj' -c '-Duse_libMPI -L/gpfs1/home/Libs/INTEL/OPENMPI/openmpi-2.1.6/lib/ -heap-array -lmpi_usempi -lmpich -Duse_netCDF -DSPMD -Duse_AM3_physics -D_USE_LEGACY_LAND_' path_names

  # compile the MOM6 sea-ice ocean coupled model with:
  (cd ${root}/build/${platform}/ice_ocean_SIS2/repro/; source ../../env.dt2; make NETCDF=3 REPRO=1 MOM6 -j)
endif

echo "NOTE: Natural end-of-script."
exit 0
