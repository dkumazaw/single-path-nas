#!/bin/bash

# d: depth_multiplier (*100)
# k: kernel size (for all MBConvs)
# e: expansio ratio (for all MBConvs)

mkdir tflite-models

d=100
for k in 3 5
do
  for e in 2 3 4 5 6
  do	    
      echo $d, $k, $e
      mkdir tflite-models/model-$d-$k-$e
      python main_tflite.py --tpu=$TPU_NAME --data_dir=$DATA_DIR --model_dir=${STORAGE_BUCKET}/model-runtime-model/model-tflite-$d-$k-$e --export_dir=$(pwd)/tflite-models/model-$d-$k-$e --depth_multiplier=$d --kernel=$k --expratio=$e --mode=train  --post_quantize=True
      python profiler_scripts.py $(pwd)/tflite-models/model-$d-$k-$e/
  done
done


