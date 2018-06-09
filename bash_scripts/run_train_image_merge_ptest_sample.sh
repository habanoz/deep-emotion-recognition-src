#!/usr/bin/env bash
for VARIABLE in {1..10}
do
	python ../train_image_merge_ptest_sample.py "/mnt/sda2/dev_root/work2.1/merged/merged_base_models/sample_ttest_noseed_vggface_avrgpool_fer13only/sample_$VARIABLE"
done
