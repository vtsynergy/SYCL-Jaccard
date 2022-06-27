#/!bin/bash
DEV_NUM=$1
file=$2
INPUT_DIR=`pwd`/../../datasets-csrV2
MY_VARIANT=`pwd | sed 's/^.*\///'`
OUTPUT_DIR=`pwd`/../../testOutputs/"$MY_VARIANT"
if [[ ! -d $OUTPUT_DIR ]]; then
  mkdir -p $OUTPUT_DIR
fi
if [[ $COMPILER == "" ]]; then
  COMPILER=ICX
fi

echo "Testing $MY_VARIANT $file at `date`"

CFLAGS="-DDISABLE_WEIGHTED" SYCL_C_FLAGS="" SYCL_LD_FLAGS="" ./build_proteus.sh $COMPILER $DEV_NUM 

JACCARD_FORCE_EDGE_CENTRIC=1 JACCARD_FORCE_WEIGHTED=0 ./jaccardSYCL "$INPUT_DIR"/$file "$OUTPUT_DIR"/EC.$HOSTNAME.$COMPILER.$DEV_NUM-$file $DEV_NUM > "$OUTPUT_DIR"/EC.$HOSTNAME.$COMPILER.$DEV_NUM-$(echo $file | sed "s/csr/log/" | sed "s/mtx/log/") 2> "$OUTPUT_DIR"/ECprof.$HOSTNAME.$COMPILER.$DEV_NUM-$file 
JACCARD_FORCE_VERTEX_CENTRIC=1 JACCARD_FORCE_WEIGHTED=0 ./jaccardSYCL "$INPUT_DIR"/$file "$OUTPUT_DIR"/VC.$HOSTNAME.$COMPILER.$DEV_NUM-$file $DEV_NUM > "$OUTPUT_DIR"/VC.$HOSTNAME.$COMPILER.$DEV_NUM-$(echo $file | sed "s/csr/log/" | sed "s/mtx/log/") 2> "$OUTPUT_DIR"/VCprof.$HOSTNAME.$COMPILER.$DEV_NUM-$file 

