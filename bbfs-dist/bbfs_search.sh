#!/bin/bash

# Run the BBF search algorithm (both Earthward and tailward) for the
# selected year
while IFS=, read -r year month day; do
  echo $year
  if [ $1 -eq $year ]
  then
    python3.9 bbfs_search.py earthward ${year}-${month}-${day} &
    python3.9 bbfs_search.py tailward ${year}-${month}-${day}
  fi
done < ./data/mms_tail_seasons_days.txt