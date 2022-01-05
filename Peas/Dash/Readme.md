# what's going on here.

raw `.dat` file will be loaded and processed by `ETL.py` file because:
1. iplant has an unique authentication policy  
2. diffcult to load data from iplant to powerplant  
3. therefore, logic is to process data within the iplant network and upload
processed data into a postgresql database on powerplant
4. dash app can query data
