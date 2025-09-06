git checkout requirements.txt
rbuild build -d depend -B build -DGPU_TARGETS=$(/opt/rocm/bin/rocminfo | grep -o -m1 'gfx.*')
export MIGRAPHX_ENABLE_MLIR_GEG_FUSION=0
bash test-gemms.sh no_geg

export MIGRAPHX_ENABLE_MLIR_GEG_FUSION=1
bash test-gemms.sh geg 

export MIGRAPHX_MLIR_ENABLE_SPLITK=1
cp splitkreqs.txt requirements.txt
rbuild build -d depend -B build -DGPU_TARGETS=$(/opt/rocm/bin/rocminfo | grep -o -m1 'gfx.*')
bash test-gemms.sh geg_splitk 
