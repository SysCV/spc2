#!/bin/bash
cd ..
#while true; do
    python main.py \
        --env carla8 \
        --learning-freq 100 \
        --num-train-steps 20 \
        --num-total-act 2 \
        --pred-step 10 \
        --buffer-size 20000 \
        --epsilon-frames 100000 \
        --batch-size 2 \
	--save-freq 500 \
        --use-offroad \
        --use-speed \
	--use-collision \
        --sample-with-collision \
        --sample-with-offroad \
	--sample-with-offlane \
        --speed-threshold 15 \
        --use-guidance \
        --expert-bar 200 \
        --safe-length-collision 50 \
        --safe-length-offroad 30 \
        --data-parallel \
        --id 402 \
        --verbose \
	--wandb \
	--port 2666 \
	--vehicle-num 120 \
	--sample-type binary \
	--optim Adam \
	--lr 2e-4 \
	--weather-id 0 \
	--use-offlane \
	--use-3d-detection \
	--debug \
	--save-record \
	--frame-width 512 \
	--frame-height 256 \
	# --use-detection \
	# --resume \
	# --use-offlane \
	# --use-detection \
	# --CEM \
    	# --use-detection \
	#e--detach-seg
	# --use-orientation \
	# --use-colls-with \
	# --use-collision-other \
	# --use-collision-vehicles \
#done
