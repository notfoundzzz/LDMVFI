# UNPACK_LDMVFI_ENV

After uploading the packed local environment tarball to cloud, unpack it like this:

```bash
mkdir -p /data/Shenzhen/zhahongli/envs/ldmvfi
tar -xzf /data/Shenzhen/zhahongli/ldmvfi_env.tar.gz -C /data/Shenzhen/zhahongli/envs/ldmvfi
/data/Shenzhen/zhahongli/envs/ldmvfi/bin/conda-unpack
```

Basic validation:

```bash
/data/Shenzhen/zhahongli/envs/ldmvfi/bin/python -V
/data/Shenzhen/zhahongli/envs/ldmvfi/bin/python /data/Shenzhen/zhahongli/LDMVFI/check_env_ldmvfi.py
```

Recommended launch pattern:

```bash
unset LD_LIBRARY_PATH
unset CUDA_HOME
unset CUDA_PATH
/data/Shenzhen/zhahongli/envs/ldmvfi/bin/python -u evaluate.py ...
```
