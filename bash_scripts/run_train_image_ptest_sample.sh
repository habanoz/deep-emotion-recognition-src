#!/usr/bin/env bash
for VARIABLE in {1..10}
do
	python ../train_image_ptest_sample.py "/mnt/sda2/dev_root/work2.1/merged/image-whole/1__1512579280.12-17403535-1-vggface-aligned-catvrossloss/sample_ttest_noseed/sample_$VARIABLE"
done
