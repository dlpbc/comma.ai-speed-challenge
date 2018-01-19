cd ./data_prep_scripts/

# approximately 4 minutes
sh rgb_prepare_data.sh 

# approximately 45 minutes 
# 30 mins for train video optical flow conversion + 
# 15 mins for test video optical flow conversion + 
# 4 minutes for other processing similar to RGB)
sh flow_prepare_data.sh 

cd ..
