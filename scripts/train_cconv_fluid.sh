python train.py 	\
	--set 			\
	loss_type chamfer 		\
	dataset CConvFluid	\
	run_dir	/workspace/VGPL-Visual-Prior/experiments/cconv_fluid_v5	\
	n_frames_eval	10	\
	group_loss_weight 0.0 \
	vis_frames_per_sample 15 \
	batch_size 50 \
	n_epochs 200



	