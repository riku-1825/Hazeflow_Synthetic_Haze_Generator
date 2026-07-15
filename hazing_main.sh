python inference_haze_main.py \
    --input_folder  \
    --output_folder  \
    --beta_folder  \
    --checkpoint  \
    --min_transmission 0.1 \
    --beta_min 1.0 \
    --beta_max 3.0 \
    --depth_power 1.0 \
    --depth_scale 1.2 \
    --device cuda:0 --num_shards 3 --shard_id 0 &
python inference_haze_main.py \
    --input_folder  \
    --output_folder  \
    --beta_folder  \
    --checkpoint  \
    --min_transmission 0.1 \
    --beta_min 1.0 \
    --beta_max 3.0 \
    --depth_power 1.0 \
    --depth_scale 1.2 \
    --device cuda:1 --num_shards 3 --shard_id 1 &
python inference_haze_main.py \
    --input_folder  \
    --output_folder  \
    --beta_folder  \
    --checkpoint  \
    --min_transmission 0.1 \
    --beta_min 1.0 \
    --beta_max 3.0 \
    --depth_power 1.0 \
    --depth_scale 1.2 \
    --device cuda:2 --num_shards 3 --shard_id 2 &
wait
