#!/bin/bash
cd ..
#while true; do
    python auto_client.py \
        --env carla8 \
        --learning-freq 100 \
        --num-train-steps 10 \
        --num-total-act 2 \
        --pred-step 10 \
        --buffer-size 20000 \
        --epsilon-frames 100000 \
        --batch-size 1 \
        --use-collision \
        --use-offroad \
        --use-speed \
        --sample-with-collision \
        --sample-with-offroad \
	--sample-with-offlane \
        --speed-threshold 15 \
        --use-guidance \
        --expert-bar 200 \
        --safe-length-collision 50 \
        --safe-length-offroad 30 \
        --data-parallel \
        --id 25 \
        --verbose \
	--vehicle-num 60 \
	--port 5000 \
	--recording-frame \
	--monitor \
	--use-offlane \
	--use-detection \
	--resume \
	# --use-orientation \
	# --use-colls-with \
	# --use-collision-other \
	# --use-collision-vehicles \
	# --monitor \
#done
