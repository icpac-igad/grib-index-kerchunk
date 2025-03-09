
from a1_utils import get_gefs_gcs_gribtree
from a1_utils import get_gefs_gcs_mapping



ens_mem_list=['01','02','03','04','05','06','07','08','09','10','11','12','13','14','15','16','17','18','19','20','21','22','23','24','25','26','27','28','29','30']

for ens_mem in ens_mem_list:
    get_gefs_gcs_gribtree(ens_mem)
    get_gefs_gcs_mapping(ens_mem)



date_str='20240610'
ens_mem='09'
e1=make_store(date_str,ens_mem)
