﻿# dq-manual-check
For DL1
usage:

dq dl1 output directory is located in 
SAG_BASE_PATH/yyyymmdd/sb_{sb_id}_ob_{ob_id}/dq/dl1/

set two environment variables
```
export DQ_OUT=SAG_BASE_PATH/yyyymmdd/sb_{sb_id}_ob_{ob_id}/dq/dl1/
export PLOT_OUT=
```
run the commands:
```
source /dev/shm/miniconda3/etc/profile.d/conda.sh 
conda activate sag
python DQPlotMaker.py --outputdirectory $PLOT_OUT --inputdirectory $DQ_OUT

```

to run without conda sag environment:
```
pip install numpy astropy matplotlib
python DQPlotMaker.py --outputdirectory $PLOT_OUT --inputdirectory $DQ_OUT
```
