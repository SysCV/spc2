#!/bin/bash
cd ..
#while true; do
    python main.py \
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
        --id 1104 \
        --verbose \
	--vehicle-num 32 \
	--port 5136 \
	--resume \
	--use-offlane \
	--use-detection \
	--checkpoint exps/32vehicle/114/model/pred_model_000107100.pt \
	--eval \
    	--resume \
	--output_path demo114 \
	# --sample-type continous \
	# --user-detection\
	# --use-orientation \
	# --use-colls-with \
	# --use-collision-other \
	# --use-collision-vehicles \
	# --monitor \
#done
