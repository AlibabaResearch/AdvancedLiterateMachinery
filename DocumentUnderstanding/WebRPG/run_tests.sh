#!/bin/bash

# Ensure the script stops on the first error
set -e

# Default values for the variables
EVAL_OUTPUT_DIR=""
CHECKPOINT_DIR=""

# Parse command line arguments
for i in "$@"
do
case $i in
    --eval_output_dir=*)
    EVAL_OUTPUT_DIR="${i#*=}"
    shift # past argument=value
    ;;
    --checkpoint_dir=*)
    CHECKPOINT_DIR="${i#*=}"
    shift # past argument=value
    ;;
    *)
    # unknown option
    echo "Usage: $0 --eval_output_dir=<eval_output_dir> --checkpoint_dir=<checkpoint_dir>"
    exit 1
    ;;
esac
done

# Check if the required arguments are provided
if [ -z "$EVAL_OUTPUT_DIR" ] || [ -z "$CHECKPOINT_DIR" ]; then
  echo "Both --eval_output_dir and --checkpoint_dir are required!"
  echo "Usage: $0 --eval_output_dir=<eval_output_dir> --checkpoint_dir=<checkpoint_dir>"
  exit 1
fi

# Assign the PT_DIR based on the EVAL_OUTPUT_DIR
PT_DIR="$EVAL_OUTPUT_DIR/result_dict"

# Run the model inference
python trainer_ar.py \
  --output_dir $CHECKPOINT_DIR \
  --cache_eval_path /Path/To/Cache/test.txt \
  --checkpoint_dir $CHECKPOINT_DIR \
  --eval_output_dir $EVAL_OUTPUT_DIR \
  --pretrained_markuplm_path /Path/To/MarkupLM/Large \
  --logging_step 10 \
  --do_eval \
  --per_device_eval_batch_size 64 \
  --dataloader_num_workers 32 

echo "The model inference is completed."

# Run the FID and IOU tests
python test_FID.py --pt_dir $PT_DIR --fid_type overall
python test_FID.py --pt_dir $PT_DIR --fid_type layout
python test_ele_iou.py --pt_dir $PT_DIR
python test_FID.py --pt_dir $PT_DIR --fid_type style
python test_sc.py --pt_dir $PT_DIR

echo "Tests completed successfully"