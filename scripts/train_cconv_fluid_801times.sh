python /workspace/VGPL-Visual-Prior/train.py 	\
	--set 			\
	loss_type l2 		\
	dataset CConvFluid801times	\
	run_dir	/workspace/VGPL-Visual-Prior/experiments/cconv_fluid_801times	\
	n_frames_eval	30	\
	group_loss_weight 0.0 \
	vis_frames_per_sample 64 \
	batch_size 50 \
	n_epochs 100



	