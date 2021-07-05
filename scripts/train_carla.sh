#!/bin/bash
cd ..
#while true; do
    python -W ignore main.py \
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
        --id 200 \
        --verbose \
	--wandb \
	--port 2066 \
	--vehicle-num 32 \
	--sample-type binary \
	--optim Adam \
	--lr 2e-4 \
	--weather-id 0 \
	--use-offlane \
	--frame-width 256 \
	--frame-height 256 \
	--use-depth \
	--resume \
	--pretrain-model pretrain/pretrained.pth\
	# --use-detection \
	# --steer-clip 0.1\
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
	# --monitor \
#done
