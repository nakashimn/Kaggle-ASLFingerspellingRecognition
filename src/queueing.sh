#! /bin/bash

# const
WAIT_TIME_HOUR=24
CHECK_CYCLE_HOUR=1

# variables
check_count=$((${WAIT_TIME_HOUR}/${CHECK_CYCLE_HOUR}))
wait_time_sec=$((60*60*${CHECK_CYCLE_HOUR}))

# check whether any python process is existing or not.
for i in $(seq 0 ${check_count}) ; do

  # if python process is NOT running on GPU
  if ! $(nvidia-smi | grep -q python); then
    # run command
    $@
    break
  fi

  # wait
  if [ ${i} -ne 0 ]; then
    echo [${i}/${check_count}] other process is running...
  fi
  sleep ${wait_time_sec}

done

echo TimeOut for waiting \"\>$@\"
