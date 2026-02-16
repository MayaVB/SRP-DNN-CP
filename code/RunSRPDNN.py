"""
	Function: Run training and test processes for source source localization on simulated dataset

    Reference: Bing Yang, Hong Liu, and Xiaofei Li, “SRP-DNN: Learning Direct-Path Phase Difference for Multiple Moving Sound Source Localization,” 
	IEEE International Conference on Acoustics, Speech and Signal Processing (ICASSP), 2022, pp. 721–725.
	Author:    Bing Yang
    Copyright Bing Yang
"""
import os
from OptSRPDNN import opt
import matplotlib.pyplot as plt
import numpy as np

opts = opt()
args = opts.parse()
dirs = opts.dir()
 
os.environ["OMP_NUM_THREADS"] = str(8) # limit the threads to reduce cpu overloads, will speed up when there are lots of CPU cores on the running machine
os.environ["CUDA_VISIBLE_DEVICES"] = str(args.gpu_id)

import torch
torch.backends.cuda.matmul.allow_tf32 = True  # The flag below controls whether to allow TF32 on matmul. This flag defaults to False in PyTorch 1.12 and later.
torch.backends.cudnn.allow_tf32 = True  # The flag below controls whether to allow TF32 on cuDNN. This flag defaults to True.

import numpy as np
import time
import scipy.io

# from tensorboardX import SummaryWriter
from torch.utils.tensorboard import SummaryWriter

import Dataset as at_dataset
import LearnerSRPDNN as at_learner
import ModelSRPDNN as at_model
import Module as at_module
from Dataset import Parameter
from utils import set_seed, set_random_seed, set_learning_rate
from crc_pipeline import find_top2_peaks, match_peaks_to_gt, extract_regions_test

if __name__ == "__main__":
	use_cuda = not args.no_cuda and torch.cuda.is_available()
	device = torch.device("cuda" if use_cuda else "cpu")

	# os.environ["CUDA_VISIBLE_DEVICES"] = str(args.gpu_id) if use_cuda else {}

	set_seed(args.seed)

	# %% Dataset
	speed = 343.0
	fs = 16000
	T = 20 # Trajectory length (s) 
	if args.source_state == 'static':
		traj_points = 1 # number of RIRs per trajectory
	elif args.source_state == 'mobile':
		traj_points = int(10*T) # number of RIRs per trajectory
	else:
		print('Source state model unrecognized~')

	# Array
	array = '12ch'
	if array == '2ch':
		array_setup = at_dataset.dualch_array_setup
		array_locata_name = 'dicit'
	elif array == '12ch':
		array_setup = at_dataset.benchmark2_array_setup
		array_locata_name = 'benchmark2'  # Name of the array in the LOCATA dataset

	if args.gen_on_the_fly:
		# Source signal
		sourceDataset_train = at_dataset.LibriSpeechDataset(
			path = dirs['sousig_train'], 
			T = T, 
			fs = fs, 
			num_source = max(args.sources), 
			return_vad=True, 
			clean_silence=True)
		sourceDataset_val = at_dataset.LibriSpeechDataset(
			path = dirs['sousig_val'], 
			T = T, 
			fs = fs, 
			num_source = max(args.sources), 
			return_vad=True, 
			clean_silence=True)
		sourceDataset_test = at_dataset.LibriSpeechDataset(
			path = dirs['sousig_test'], 
			T = T, 
			fs = fs, 
			num_source = max(args.sources),  
			return_vad=True, 
			clean_silence=True)
		
		# Noise signal
		noiseDataset_train = at_dataset.NoiseDataset(
			T = T, 
			fs = fs, 
			nmic = array_setup.mic_pos.shape[0], 
			noise_type = Parameter(['spatial_white'], discrete=True), 
			noise_path = None, 
			c = speed)
		noiseDataset_val = at_dataset.NoiseDataset(
			T = T, 
			fs = fs, 
			nmic = array_setup.mic_pos.shape[0], 
			noise_type = Parameter(['spatial_white'], discrete=True), 
			noise_path = None, 
			c = speed)
		# Configure test noise dataset with DirBurst support
		if args.enable_dirburst:
			test_noise_type = ['dirburst']
			test_dirburst_params = {
				'duration_range': (args.dirburst_duration_min, args.dirburst_duration_max),
				'num_bursts': args.dirburst_num_bursts,
				'snr_db': args.dirburst_snr,
				'max_tries': 50,
				'color': args.dirburst_color,
				'enabled': True
			}
			print(f"DirBurst enabled for testing: {args.dirburst_num_bursts} bursts, duration {args.dirburst_duration_min}-{args.dirburst_duration_max}s, SNR {args.dirburst_snr}dB, color {args.dirburst_color}")
		else:
			test_noise_type = ['spatial_white']
			test_dirburst_params = {'enabled': False}
			print("DirBurst disabled - using spatial white noise for testing")

		noiseDataset_test = at_dataset.NoiseDataset(
			T = T,
			fs = fs,
			nmic = array_setup.mic_pos.shape[0],
			noise_type = Parameter(test_noise_type, discrete=True),
			noise_path = None,
			c = speed,
			dirburst_params = test_dirburst_params)

	# Segmenting, STFT parameters
	# When win_shift_ratio = 0.5, then the number of time frames corresponding to one segment can be set to an integer
	win_len = 512
	nfft = 512
	win_shift_ratio = 0.5
	fre_used_ratio = 1
	if args.source_state == 'static':
		seg_len = T*fs
		seg_shift = 1
	elif args.source_state == 'mobile':
		seg_fra_ratio = 12 # one estimate per segment (namely seg_fra_ratio frames) 
		seg_len = int(win_len*win_shift_ratio*(seg_fra_ratio-1)+win_len)
		seg_shift = int(win_len*win_shift_ratio*seg_fra_ratio)
	else:
		print('Source state model unrecognized~')
	segmenting = at_dataset.Segmenting_SRPDNN(K=seg_len, step=seg_shift, window=None)

	# Room acoustics
	if args.gen_on_the_fly:
		dataset_train = at_dataset.RandomMicSigDataset(
			sourceDataset = sourceDataset_train,
			num_source = Parameter(args.sources, discrete=True), # Random number of sources
			source_state = args.source_state,
			room_sz = Parameter([3,3,2.5], [10,8,6]),  	# Random room sizes from 3x3x2.5 to 10x8x6 meters
			T60 = Parameter(0.2, 1.3),					# Random reverberation times from 0.2 to 1.3 seconds
			abs_weights = Parameter([0.5]*6, [1.0]*6),  # Random absorption weights ratios between walls
			array_setup = array_setup,
			array_pos = Parameter([0.1,0.1,0.1], [0.9,0.9,0.5]), # Ensure a minimum separation between the array and the walls
			noiseDataset = noiseDataset_train,
			SNR = Parameter(5, 30), 	
			nb_points = traj_points,	
			dataset_sz= 1000,
			c = speed, 
			transforms = [segmenting]
		)
		dataset_val = at_dataset.RandomMicSigDataset( 
			sourceDataset = sourceDataset_val,
			num_source = Parameter(args.sources, discrete=True),  
			source_state = args.source_state,
			room_sz = Parameter([3,3,2.5], [10,8,6]),
			T60 = Parameter(0.2, 1.3),
			abs_weights = Parameter([0.5]*6, [1.0]*6),
			array_setup = array_setup,
			array_pos = Parameter([0.1,0.1,0.1], [0.9,0.9,0.5]),
			noiseDataset = noiseDataset_val,
			SNR = Parameter(5, 30),
			nb_points = traj_points,
			dataset_sz= 1000,
			c = speed, 
			transforms = [segmenting]
		)

		dataset_cal = dataset_val
		dataset_cal.dataset_sz = 100
  
		dataset_test = at_dataset.RandomMicSigDataset( 
			sourceDataset = sourceDataset_test,
			num_source = Parameter(args.sources, discrete=True),  
			source_state = args.source_state,
			room_sz = Parameter([3,3,2.5], [10,8,6]),
			T60 = Parameter(0.2, 1.3),
			abs_weights = Parameter([0.5]*6, [1.0]*6),
			array_setup = array_setup,
			array_pos = Parameter([0.1,0.1,0.1], [0.9,0.9,0.5]),
			noiseDataset = noiseDataset_test,
			SNR = Parameter(5, 30),
			nb_points = traj_points,
			dataset_sz= 1000,
			c = speed, 
			transforms = [segmenting]
		)
	else:
		dataset_train = at_dataset.FixMicSigDataset( 
			data_dir = dirs['sensig_train'],
			dataset_sz = 51200,
			transforms = [segmenting]
			)			
		dataset_val = at_dataset.FixMicSigDataset( 
			data_dir = dirs['sensig_val'],
			dataset_sz = 1024,
			transforms = [segmenting]
			)
		dataset_cal = dataset_val
		dataset_test = at_dataset.FixMicSigDataset( 
			data_dir = dirs['sensig_test'],
			dataset_sz = 1024,
			transforms = [segmenting]
			)

	# %% Network declaration, learner declaration
	tar_useVAD = True
	ch_mode = 'MM' 
	res_the = 37 # Maps resolution (elevation) 
	res_phi = 73 # Maps resolution (azimuth) 

	net = at_model.CRNN(max_num_sources=int(args.localize_mode[2]))
	# from torchsummary import summary
	# summary(net,input_size=(4,256,100),batch_size=55,device="cpu")
	print('# Parameters:', sum(param.numel() for param in net.parameters())/1000000, 'M')

	learner = at_learner.SourceTrackingFromSTFTLearner(net, win_len=win_len, win_shift_ratio=win_shift_ratio, nfft=nfft, fre_used_ratio=fre_used_ratio,
				nele=res_the, nazi=res_phi, rn=array_setup.mic_pos, fs=fs, ch_mode = ch_mode, tar_useVAD = tar_useVAD, localize_mode = args.localize_mode) 

	if use_cuda:
		if len(args.gpu_id)>1:
			learner.mul_gpu()
		learner.cuda()
	else:
		learner.cpu()
	if args.use_amp:
		learner.amp()
	if args.gen_on_the_fly:
		args.workers = 0
	kwargs = {'num_workers': args.workers, 'pin_memory': True}  if use_cuda else {}


	if (args.cp_calibration): 
		print("CP Calibration Stage!")
		dataloader_cal = torch.utils.data.DataLoader(dataset_cal, batch_size=args.bs[1], shuffle=False, **kwargs)
		# dataloader_cal = torch.utils.data.DataLoader(dataset_cal, batch_size=10, shuffle=False, **kwargs)

		# Load trained model
		learner.resume_checkpoint(checkpoints_dir=dirs['log'], from_latest=False)

		# Run prediction
		pred, gt, _ = learner.predict(
			dataloader_cal,
			return_predgt=True,
			metric_setting=None,
			wDNN=True
		)

		# Concatenate batches (reuse your helper)
		def _concat_predgt(x):
			if isinstance(x, list):
				if len(x) == 0:
					return x
				if isinstance(x[0], dict):
					result = {}
					for k in x[0].keys():
						if torch.is_tensor(x[0][k]):
							# Concatenate tensors
							result[k] = torch.cat([xb[k] for xb in x], dim=0)
						else:
							# For non-tensor values (like lists), collect all samples
							result[k] = [item for xb in x for item in (xb[k] if isinstance(xb[k], list) else [xb[k]])]
					return result
				else:
					# list of tensors -> concatenated tensor
					return torch.cat(x, dim=0)
			return x

		# Debug: Check data types before concatenation
		print("Debug: Checking data types in pred and gt...")
		if isinstance(pred, list) and len(pred) > 0 and isinstance(pred[0], dict):
			print("pred[0] keys and types:")
			for k, v in pred[0].items():
				print(f"  {k}: {type(v)} {v.shape if torch.is_tensor(v) else len(v) if isinstance(v, list) else 'N/A'}")

		if isinstance(gt, list) and len(gt) > 0 and isinstance(gt[0], dict):
			print("gt[0] keys and types:")
			for k, v in gt[0].items():
				print(f"  {k}: {type(v)} {v.shape if torch.is_tensor(v) else len(v) if isinstance(v, list) else 'N/A'}")

		pred = _concat_predgt(pred)
		gt   = _concat_predgt(gt)

		ss_pred = pred['spatial_spectrum'].cpu().numpy()   # (nb, nt, nele, nazi)
		doa_gt  = gt['doa']                                # (nb, nt, 2, ns)

		nb, nt, nele, nazi = ss_pred.shape

		# Debug tensor shapes
		print(f"Tensor shapes:")
		print(f"  ss_pred: {ss_pred.shape}")
		print(f"  doa_gt: {doa_gt.shape}")

		deltas = []

		print(f"Processing {nb} batches with {nt} time frames each...")

		for b in range(nb):
			for t in range(nt):
				S = ss_pred[b, t]  # (nele, nazi)

				# 1) Detect top-2 peaks using proper peak detection
				peaks = find_top2_peaks(S, suppress_radius=(3, 3))

				if len(peaks) == 0:
					print(f"Warning: No peaks found in batch {b}, frame {t}")
					continue  # Skip frames with no detected peaks

				# 2) Convert GT DOA to grid indices for all speakers in this frame
				gt_speakers = []
				for s in range(doa_gt.shape[-1]):
					ele_rad = doa_gt[b, t, 0, s].item()
					azi_rad = doa_gt[b, t, 1, s].item()

					# Convert rad -> grid index with proper bounds checking
					ele_idx = int(round((ele_rad / np.pi) * (nele - 1)))
					azi_idx = int(round(((azi_rad + np.pi) / (2 * np.pi)) * (nazi - 1)))

					ele_idx = np.clip(ele_idx, 0, nele-1)
					azi_idx = np.clip(azi_idx, 0, nazi-1)

					gt_speakers.append((ele_idx, azi_idx))

				# 3) Match detected peaks to GT speakers
				matched_gt_indices = match_peaks_to_gt(peaks, gt_speakers, (nele, nazi))

				# 4) Compute nonconformity scores for matched pairs
				#
				# ⚠️  CRITICAL CP CORRECTNESS ISSUE - TODO: FIX THIS! ⚠️
				# Current implementation SILENTLY SKIPS unmatched GT speakers (when pred_idx is None).
				# This is INCORRECT for conformal prediction and breaks coverage guarantees!
				# See detailed explanation in crc_pipeline.py lines 395-415.
				#
				for gt_idx, pred_idx in enumerate(matched_gt_indices):
					if pred_idx is not None and gt_idx < len(gt_speakers):
						# Get peak value at predicted location
						peak_ele, peak_azi = peaks[pred_idx]
						S_peak = S[peak_ele, peak_azi]

						# Get spectrum value at ground truth location
						gt_ele, gt_azi = gt_speakers[gt_idx]
						S_gt = S[gt_ele, gt_azi]

						# Nonconformity score: ensure non-negative for CP validity
						delta = max(0.0, S_peak - S_gt)
						deltas.append(delta)
					# TODO: Add penalty for unmatched GT speakers:
					# else:
					#     deltas.append(np.inf)  # or some large penalty value

				# Debug info for first few frames
				if b < 2 and t < 2:
					print(f"  Batch {b}, Frame {t}: {len(peaks)} peaks, {len(gt_speakers)} GT speakers, "
						  f"{sum(1 for x in matched_gt_indices if x is not None)} matches")

		deltas = np.array(deltas)

		if len(deltas) == 0:
			print("ERROR: No valid nonconformity scores computed!")
			print("This likely indicates peak detection or matching failures.")
			exit(1)

		alpha = args.cp_alpha
		N = len(deltas)

		# Finite-sample CP quantile
		k = int(np.ceil((N + 1) * (1 - alpha))) - 1
		lambda_hat = np.sort(deltas)[k]

		print(f"Alpha: {alpha}")
		print(f"Lambda_hat: {lambda_hat}")
		print(f"Total calibration samples: {N}")

		# Empirical coverage calculation
		empirical_coverage = np.mean(deltas <= lambda_hat) * 100
		target_coverage = (1 - alpha) * 100
		print(f"Empirical coverage: {empirical_coverage:.1f}% (target: {target_coverage:.1f}%)")

		# Additional statistics
		print(f"Delta statistics: mean={np.mean(deltas):.3f}, std={np.std(deltas):.3f}, "
			  f"median={np.median(deltas):.3f}")
		print(f"Delta range: [{np.min(deltas):.3f}, {np.max(deltas):.3f}]")

		# Save lambda
		cal_dir = dirs['log'] + '/calibration'
		if not os.path.exists(cal_dir):
			os.makedirs(cal_dir)

		lambda_path = cal_dir + f'/lambda_alpha_{alpha}.npy'
		np.save(lambda_path, lambda_hat)
		print(f"Saved lambda to {lambda_path}")

		# Improved plot with empirical coverage
		plt.figure(figsize=(10, 6))

		# Histogram of nonconformity scores
		plt.hist(deltas, bins=min(50, max(10, N//20)), alpha=0.7, color='skyblue',
				 edgecolor='black', density=True, label='Nonconformity Scores')

		# Mark lambda_hat with a vertical line
		plt.axvline(lambda_hat, color='red', linestyle='--', linewidth=2,
				   label=f'λ_hat = {lambda_hat:.3f}')

		# Add text annotations
		plt.xlabel('Nonconformity Score (δ = S_peak - S_GT)')
		plt.ylabel('Density')
		plt.title(f'CP Calibration: Nonconformity Score Distribution\n'
				 f'α = {alpha} (Target Coverage: {target_coverage:.1f}%), '
				 f'Empirical Coverage: {empirical_coverage:.1f}%')
		plt.legend()
		plt.grid(True, alpha=0.3)

		# Add statistics text box
		stats_text = (f'N = {N} scores\n'
					 f'Mean: {np.mean(deltas):.3f}\n'
					 f'Std: {np.std(deltas):.3f}\n'
					 f'Min: {np.min(deltas):.3f}\n'
					 f'Max: {np.max(deltas):.3f}')
		plt.text(0.02, 0.98, stats_text, transform=plt.gca().transAxes,
				verticalalignment='top', bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.8))

		plt.tight_layout()
		plt.savefig(cal_dir + f'/delta_hist_alpha_{alpha}.png', dpi=150, bbox_inches='tight')
		print(f"Saved calibration plot to: {cal_dir}/delta_hist_alpha_{alpha}.png")
		plt.close()

		print("Calibration complete.")
		exit(0)

	elif (args.train):
		print('Training Stage!')

		if args.checkpoint_start:
			learner.resume_checkpoint(checkpoints_dir=dirs['log'], from_latest=True) # Train from latest checkpoints

		# %% TensorboardX
		train_writer = SummaryWriter(dirs['log'] + '/train/', 'train')
		val_writer = SummaryWriter(dirs['log'] + '/val/', 'val')
		test_writer = SummaryWriter(dirs['log'] + '/test/', 'test')

		# %% Network training
		nepoch = args.epochs

		dataloader_train = torch.utils.data.DataLoader(dataset=dataset_train, batch_size=args.bs[0], shuffle=True, **kwargs)
		dataloader_val = torch.utils.data.DataLoader(dataset=dataset_val, batch_size=args.bs[1], shuffle=False, **kwargs)
		dataloader_test = torch.utils.data.DataLoader(dataset=dataset_test, batch_size=args.bs[2], shuffle=False, **kwargs)

		for epoch in range(learner.start_epoch, nepoch+1, 1):
			print('\nEpoch {}/{}:'.format(epoch, nepoch))

			lr = set_learning_rate(epoch=epoch-1, lr_init=args.lr, step=args.epochs, gamma=0.05)

			set_random_seed(epoch)
			loss_train = learner.train_epoch(dataloader_train, lr=lr, epoch=epoch, return_metric=False)

			set_random_seed(args.seed)
			loss_val, metric_val = learner.test_epoch(dataloader_val, return_metric=True)
			print('Val loss: {:.4f}, Val MDR: {:.2f}%, Val FAR: {:.2f}%, Val MAE: {:.2f}deg'.\
					format(loss_val, metric_val['MDR']*100, metric_val['FAR']*100, metric_val['MAE']) )

			# loss_test, metric_test = learner.test_epoch(dataloader_test, return_metric=True)
			# print('Test loss: {:.4f}, Test MDR: {:.2f}%, Test FAR: {:.2f}%, Test MAE: {:.2f}deg'.\
			# 		format(loss_test, metric_test['MDR']*100, metric_test['FAR']*100, metric_test['MAE']) )

			# %% Save model
			is_best_epoch = learner.is_best_epoch(current_score=loss_val*(-1))
			learner.save_checkpoint(epoch=epoch, checkpoints_dir=dirs['log'], is_best_epoch=is_best_epoch)

			# %% Visualize parameters with tensorboardX
			train_writer.add_scalar('loss', loss_train, epoch)
			# train_writer.add_scalar('metric-MDR', metric_train['MDR'], epoch)
			# train_writer.add_scalar('metric-FAR', metric_train['FAR'], epoch)
			# train_writer.add_scalar('metric-MAE', metric_train['MAE'], epoch)
			val_writer.add_scalar('loss', loss_val, epoch)
			val_writer.add_scalar('metric-MDR', metric_val['MDR'], epoch)
			val_writer.add_scalar('metric-FAR', metric_val['FAR'], epoch)
			val_writer.add_scalar('metric-MAE', metric_val['MAE'], epoch)
			# test_writer.add_scalar('loss', loss_test, epoch)
			# test_writer.add_scalar('metric-MDR', metric_test['MDR'], epoch)
			# test_writer.add_scalar('metric-FAR', metric_test['FAR'], epoch)
			# test_writer.add_scalar('metric-MAE', metric_test['MAE'], epoch)
			test_writer.add_scalar('lr', lr, epoch)

		print('\nTraining finished\n')

	elif (args.test):
		print('Test Stage!')
		# Mode selection
		dataset_mode = args.eval_mode[0] # use no-cuda, use cuda, use amp
		method_mode = args.localize_mode[0]
		source_num_mode = args.localize_mode[1]
		
		# Metric Maya
		# metric_setting = {'ae_mode':['azi', 'ele'], 'ae_TH':30, 'useVAD':True, 'vad_TH':[2/3, 0.3],'metric_unfold':True}
		metric_setting = {'ae_mode':['azi', 'ele'], 'ae_TH':30, 'useVAD':False, 'vad_TH':[2/3, 0.0],'metric_unfold':True}
		nmetric = 3 + len(metric_setting['ae_mode']) * 2

		# Load model
		learner.resume_checkpoint(checkpoints_dir=dirs['log'], from_latest=False)

		if dataset_mode == 'simulate':
			print('- Simulated Dataset')
			ins_mode = args.eval_mode[1] 
			save_result_flag = True 

			if save_result_flag:
				result_dir = dirs['log'] + '/results_Simulate'
				if not os.path.exists(result_dir):
					os.makedirs(result_dir)

			# %% Analyze results
			if ins_mode == 'all':
				T60 = np.array((0.3, 1)) # Reverberation times to analyze
				SNR = np.array((30,))  # SNRs to analyze
				# SNR = np.array((5, 15, 30))  # SNRs to analyze
				dataset_test.dataset_sz = 10

				# CONTROL WHICH INSTANCES TO PLOT - modify this list!
				plot_instances = [0, 1, 2, 5]  # Plot instances 0, 1, 2, and 5
				
			elif ins_mode == 'some':
				T60 = np.array((0.4,))  # Reverberation times to analyze
				SNR = np.array((5,))  	# SNRs to analyze
				dataset_test.dataset_sz = 10  # Make sure dataset has enough instances

				# CONTROL WHICH INSTANCES TO PLOT - modify this list!
				plot_instances = [0, 1, 3, 7]  # Plot instances 0, 1, 3, and 7
				ins_idx = 1  # Keep for compatibility with existing code
    
			metrics = np.zeros((nmetric, len(T60), len(SNR)))
			metrics_woDNN = np.zeros((nmetric, len(T60), len(SNR)))
			for i in range(len(T60)):
				for j in range(len(SNR)):
					print('Analyzing scenes with T60=' + str(T60[i]) + 's and SNR=' + str(SNR[j]) + 'dB')
					dataset_test.T60 = Parameter(T60[i])
					dataset_test.SNR = Parameter(SNR[j])
					set_random_seed(args.seed)
					dataloader_test = torch.utils.data.DataLoader(dataset=dataset_test, batch_size=args.bs[2], shuffle=False, **kwargs)
					pred, gt, mic_sig = learner.predict(dataloader_test, return_predgt=True, metric_setting=None, wDNN=True)
					# pred_woDNN, _, = learner.predict(dataloader_test, return_predgt=True, metric_setting=None, wDNN=False)
					pred_woDNN, _, _ = learner.predict(dataloader_test, return_predgt=True, metric_setting=None, wDNN=False)

					# Maya fix: 
					# In simulate mode, learner.predict() returns lists of batches (pred, gt, mic_sig).
					# But below code expects concatenated dict/tensors (e.g., gt['doa']).
					# So i concatenate here to match the rest of the scenarios
					def _concat_predgt(x):
						"""Concatenate list-of-batches into a single tensor/dict-of-tensors."""
						if isinstance(x, list):
							if len(x) == 0:
								return x
							if isinstance(x[0], dict):
								# dict of tensors per batch -> dict of concatenated tensors
								result = {}
								for k in x[0].keys():
									if torch.is_tensor(x[0][k]):
										result[k] = torch.cat([xb[k] for xb in x], dim=0)
									else:
										# For non-tensor values like burst_data, collect all samples
										result[k] = [xb[k] for xb in x]
								return result
							else:
								# list of tensors -> concatenated tensor
								return torch.cat(x, dim=0)
						return x

					pred       = _concat_predgt(pred)
					gt         = _concat_predgt(gt)
					mic_sig    = _concat_predgt(mic_sig)
					pred_woDNN = _concat_predgt(pred_woDNN)

					# CP Region Extraction with hardcoded lambda_hat = 0.344 - BEFORE evaluation!
					lambda_hat = 0.344
					alpha = 0.1  # Corresponding to 90% coverage
					print(f"Extracting CP confidence regions with λ_hat = {lambda_hat:.3f} (α = {alpha})")

					# Extract CP regions for spatial spectrum analysis
					ss_pred_np = pred['spatial_spectrum'].cpu().numpy()   # (nb, nt, nele, nazi)
					doa_gt_rad = gt['doa']                               # (nb, nt, 2, ns) in radians

					nb, nt, nele, nazi = ss_pred_np.shape

					# Create DOA candidate grids for region conversion
					ele_candidate = np.linspace(0, np.pi, nele)           # 0 to 180 degrees
					azi_candidate = np.linspace(-np.pi, np.pi, nazi)      # -180 to 180 degrees
					doa_candidate = [ele_candidate, azi_candidate]

					# Storage for CP regions data for plotting
					cp_regions_all = []

					# Extract regions for first few samples and time frames
					cp_coverage_results = []

					for b in range(min(nb, 5)):  # Analyze first 5 batches
						cp_regions_batch = []
						for t in range(nt):  # All time frames per batch
							S = ss_pred_np[b, t]  # (nele, nazi)

							# Extract confidence regions using flood-fill (water-filling)
							regions = extract_regions_test(
								S=S,
								lambda_hat=lambda_hat,
								doa_candidate=doa_candidate,
								connectivity=8,
								prevent_overlap=True
							)

							# Check coverage against GT
							nt_gt = doa_gt_rad.shape[1]
							gt_t = min(t, nt_gt - 1)

							coverage = []
							for s in range(doa_gt_rad.shape[-1]):
								# Convert GT DOA from radians to grid indices
								ele_rad = doa_gt_rad[b, gt_t, 0, s].item()
								azi_rad = doa_gt_rad[b, gt_t, 1, s].item()

								ele_idx = int(round((ele_rad / np.pi) * (nele - 1)))
								azi_idx = int(round(((azi_rad + np.pi) / (2 * np.pi)) * (nazi - 1)))
								ele_idx = np.clip(ele_idx, 0, nele-1)
								azi_idx = np.clip(azi_idx, 0, nazi-1)

								# Check if GT falls within any region
								covered = False
								for region in regions:
									bounds = region['region_bounds']
									if (bounds['ele_min'] <= ele_idx <= bounds['ele_max'] and
										bounds['azi_min'] <= azi_idx <= bounds['azi_max']):
										covered = True
										break
								coverage.append(covered)

							cp_coverage_results.extend(coverage)

							# Store regions for plotting
							cp_regions_batch.append(regions)

							# Print first sample's results
							if b == 0 and t == 0:
								print(f"  Sample {b}-{t}: {len(regions)} CP regions, Coverage: {sum(coverage)}/{len(coverage)}")

						cp_regions_all.append(cp_regions_batch)

					# Store CP regions in learner for access by plotting functions
					learner.cp_regions = cp_regions_all
					print(f"DEBUG CP: Stored {len(cp_regions_all)} batches of CP regions in learner")
					for b_idx, batch_regions in enumerate(cp_regions_all):
						print(f"  Batch {b_idx}: {len(batch_regions)} time frames")
						for t_idx, frame_regions in enumerate(batch_regions[:3]):  # First 3 time frames
							if frame_regions is not None:
								print(f"    Time {t_idx}: {len(frame_regions)} regions")
							else:
								print(f"    Time {t_idx}: None")

					# CP Coverage statistics
					if len(cp_coverage_results) > 0:
						empirical_cp_coverage = np.mean(cp_coverage_results) * 100
						target_coverage = (1 - alpha) * 100
						print(f"  CP Empirical Coverage: {empirical_cp_coverage:.1f}% (target: {target_coverage:.1f}%)")
					else:
						print("  No CP coverage results computed")

					# Now run evaluation with CP regions available
					metrics[:, i, j], metric_keys = learner.evaluate(pred=pred, gt=gt, metric_setting=metric_setting)
					metrics_woDNN[:, i, j], _ = learner.evaluate(pred=pred_woDNN, gt=gt, metric_setting=metric_setting)

					doa_gt = (gt['doa'] * 180 / np.pi).cpu().numpy() 	# (nb, nt, 2, ns)
					vad_gt = (gt['vad_sources']).cpu().numpy()	# (nb, nt, ns)
					doa_pred = (pred['doa'] * 180 / np.pi).cpu().numpy()
					vad_pred = (pred['vad_sources']).cpu().numpy() 
					ss_pred = pred['spatial_spectrum'].cpu().numpy()
					doa_pred_woDNN = (pred_woDNN['doa'] * 180 / np.pi).cpu().numpy()
					vad_pred_woDNN = (pred_woDNN['vad_sources']).cpu().numpy()
					ss_pred_woDNN = pred_woDNN['spatial_spectrum'].cpu().numpy()
					sensig = mic_sig.cpu().numpy()

					time_stamp = np.arange(seg_len / 2, T * fs - seg_len / 2 + 1, seg_shift)
					time_stamp = np.array(time_stamp) / fs
					time_stamp = np.linspace(0.0, T, 21, endpoint=False)

					# %% Save analyzed results
					if save_result_flag:
						if ins_mode == 'some':
							sensig = sensig[ins_idx]
							doa = doa_gt[ins_idx, ...]
							vad = vad_gt[ins_idx, ...]
							doa_pred = doa_pred[ins_idx, ...]
							vad_pred = vad_pred[ins_idx, ...]
							ss_pred = ss_pred[ins_idx, ...]
							doa_pred_woDNN = doa_pred_woDNN[ins_idx, ...]
							vad_pred_woDNN = vad_pred_woDNN[ins_idx, ...]
							ss_pred_woDNN = ss_pred_woDNN[ins_idx, ...]

							scipy.io.savemat(result_dir + '/' + method_mode + '_' + source_num_mode + 'DOA_' + 'RT' + str(
								int(T60[i] * 1000)) + '_SNR' + str(SNR[j]) + '_NS' + str(args.sources[-1]) + '.mat', {
										'sensig': sensig, 'fs':fs, 'time': time_stamp, 
										'DOA': doa, 'VAD': vad,  
										'DOA_pred': doa_pred, 'VAD_pred': vad_pred, 'SS_pred': ss_pred})
							scipy.io.savemat(result_dir + '/' + method_mode + '_' + source_num_mode + 'DOA_woDNN_' + 'RT' + str(
								int(T60[i] * 1000)) + '_SNR' + str(SNR[j]) + '_NS' + str(args.sources[-1]) + '.mat', {
										'sensig':sensig, 'fs':fs, 'time': time_stamp, 
										'DOA': doa, 'VAD': vad, 
										'DOA_pred': doa_pred_woDNN, 'VAD_pred': vad_pred_woDNN, 'SS_pred': ss_pred_woDNN})
    
			if ins_mode == 'all':
				scipy.io.savemat(result_dir + '/metric_' + source_num_mode + '_NS' + str(args.sources[-1]) + '.mat', 
					{'metric': metrics, 'SNR': SNR, 'T60': T60, 'metric_key': metric_keys})
				scipy.io.savemat(result_dir + '/metric_woDNN_' + source_num_mode + '_NS' + str(args.sources[-1]) + '.mat', 
					{'metric': metrics_woDNN, 'SNR': SNR, 'T60': T60, 'metric_key': metric_keys})

			print(metric_keys, [round(i, 3) for i in np.mean(np.mean(metrics, axis=2), axis=1)])
			print('woDNN', metric_keys, ':', [round(i, 3) for i in np.mean(np.mean(metrics_woDNN, axis=2), axis=1)])

		elif dataset_mode == 'locata':
			print('- LOCATA Dataset')
			stage_mode = args.eval_mode[1:] 
			# %% Analyze LOCATA dataset
			if args.localize_mode[1] == 'unkNum':
				tasks = [(3,), (5,), (4,), (6,)] 
			elif args.localize_mode[1] == 'kNum':
				tasks = [(3,), (5,)] 
			path_locata = (dirs['sensig_locata'] + '/dev', dirs['sensig_locata'] + '/eval')
			
			signal_dir = dirs['log'] + '/signals_LOCATA'
			if not os.path.exists(signal_dir):
				os.makedirs(signal_dir)
			result_dir = dirs['log'] + '/results_LOCATA'
			if not os.path.exists(result_dir):
				os.makedirs(result_dir)
			visDOA = at_module.visDOA()

			if 'pred' in stage_mode:
				save_signal_flag = True
				save_result_flag = True
				args.bs[2] = 1
				args.workers = 0
				kwargs = {'num_workers': args.workers, 'pin_memory': True}  if use_cuda else {}
				
				 # %% Analyzing
				if array_locata_name != '' and array_locata_name is not None:
					## predict and save results
					ntask = len(tasks)
					for task in tasks:
						t_start = time.time()
						dataset_locata = at_dataset.LocataDataset(path_locata, array_locata_name, fs, dev=True, tasks=task, transforms=[segmenting])
						dataloader = torch.utils.data.DataLoader(dataset=dataset_locata, batch_size=args.bs[2], shuffle=False, **kwargs)
						pred, gt, mic_sig = learner.predict(dataloader, return_predgt=True, metric_setting=None, wDNN=True)
						t_end = time.time()

						nins = len(dataset_locata)
						print('Task '+str(task[0])+ ' processing time: ' + str(round(t_end - t_start, 1)))
						for ins_idx in range(nins):						
							## calculate time duration
							time_duration = (mic_sig[ins_idx].shape[1] - seg_len / 2 + 1) / fs
							print('Task '+str(task[0])+ '-'+ str(ins_idx+1) + 'st instance time duration: ' + str(round(time_duration, 1)))
							
							## save data
							if save_signal_flag:
								sensig = mic_sig[ins_idx].cpu().numpy()
								scipy.io.savemat(signal_dir + '/mic_signal_task' + str(task[0]) + '_' + str(ins_idx+1) + '.mat', {
									'mic_signal': sensig, 'fs': fs, # mic signals without start silence
								})
							if save_result_flag:
								sample_end = mic_sig[ins_idx].shape[1] - seg_len / 2 + 1
								sample_stamp = np.arange(seg_len / 2, sample_end, seg_shift)
								time_stamp = np.array(sample_stamp) / fs

								doa_gt = (gt[ins_idx]['doa'] * 180 / np.pi).cpu().numpy()	# (1, nt, 2, ns)
								vad_gt = (gt[ins_idx]['vad_sources']).cpu().numpy()		# (1, nt, ns)
								doa_pred = (pred[ins_idx]['doa'] * 180 / np.pi).cpu().numpy()
								vad_pred = (pred[ins_idx]['vad_sources']).cpu().numpy() 
								ss_pred = pred[ins_idx]['spatial_spectrum'].cpu().numpy()

								scipy.io.savemat(result_dir + '/' + method_mode + '_' + source_num_mode + '_DOA_task' + str(task[0]) + '_' + str(ins_idx+1) + '.mat', {
									'DOA': doa_gt, 'VAD': vad_gt, 
									'DOA_pred': doa_pred, 'VAD_pred': vad_pred, 
									'SS_pred': ss_pred, 'time_stamp': time_stamp})
								
			if 'eval' in stage_mode:	
				save_result_flag = True
				## adjust parameters and find a tradeoff between MDR and FAR
				if args.localize_mode[1] == 'unkNum':
					vad_TH_list = [i for i in np.arange(0.1, 0.6, 0.01)] # 0.35
				elif args.localize_mode[1] == 'kNum':
					vad_TH_list = [0.5]	
				nvad_TH = len(vad_TH_list) 
				dataset_locata = at_dataset.LocataDataset(path_locata, array_locata_name, fs, dev=True, tasks=tasks[0], transforms=[segmenting])
				nins = len(dataset_locata)
				ntask = len(tasks)
				metrics = np.zeros((nmetric, nins, ntask, nvad_TH))
				metrics_woDNN = np.zeros((nmetric, nins, ntask, nvad_TH))
				for vad_TH in vad_TH_list:
					vad_TH_idx = vad_TH_list.index(vad_TH)
					metric_setting['vad_TH'][1] = vad_TH
					for task in tasks:
						task_idx = tasks.index(task)
						for ins_idx in range(nins):
							## read saved data
							result = scipy.io.loadmat(result_dir + '/' + method_mode + '_' + source_num_mode + '_DOA_task'+str(task[0])+'_'+ str(ins_idx+1)+'.mat')

							gt = {'doa': torch.from_numpy(result['DOA'] / 180 * np.pi), 
								'vad_sources': torch.from_numpy(result['VAD'])}
							pred = {'doa': torch.from_numpy(result['DOA_pred'] / 180 * np.pi), 
								'vad_sources': torch.from_numpy(result['VAD_pred'])}
							
							## calculate metrics
							metrics[:, ins_idx, task_idx, vad_TH_idx], metric_keys = learner.evaluate(pred, gt, metric_setting=metric_setting)					
				
				metrics_ave_task = np.mean(metrics, axis=1)
				metrics_ave = np.mean(metrics_ave_task, axis=1)
				MDRs = metrics_ave[1, :]
				FARs = metrics_ave[2, :]
				scores = np.sqrt(MDRs * MDRs + FARs * FARs)
				best_idx = np.argmin(scores)
				best_vad_TH = vad_TH_list[best_idx]
				best_metric = metrics[..., best_idx] 	# (nmetric, nins, ntask)
				best_metric_ave_task = metrics_ave_task[..., best_idx] 	# (nmetric, ntask)
				best_metric_ave = metrics_ave[..., best_idx] 	# (nmetric)
				# ACC, MDR, FAR, MAE, RMSE

				print('Best VAD threshold for predictions is:', best_vad_TH)
				print(metric_keys, ':', [round(i, 3) for i in best_metric_ave])
				# print(metric_keys, ':', [round(i, 3) for i in np.mean(best_metric_ave_task[:,0:2],axis=1)])
				# print(metric_keys, ':', [round(i, 3) for i in np.mean(best_metric_ave_task[:,2:4],axis=1)])

				if save_result_flag:
					scipy.io.savemat(result_dir + '/' + source_num_mode + '_metric' + '.mat', 
							{'metric': best_metric, 'vad_TH': best_vad_TH, 'task': tasks, 'metric_key': metric_keys})
					
					for task in tasks:
						task_idx = tasks.index(task)
						for ins_idx in range(nins):
							result = scipy.io.loadmat(result_dir + '/' + method_mode + '_'  +source_num_mode + '_DOA_task' + str(task[0]) + '_' + str(ins_idx+1)+'.mat')

							# images
							doa_gt = result['DOA'][0, ...]
							vad_gt = result['VAD'][0, ...]
							doa_pred = result['DOA_pred'][0, ...]
							vad_pred = result['VAD_pred'][0, ...]
							time_stamp = result['time_stamp'][0, ...]
							vad_TH = [metric_setting['vad_TH'][0], best_vad_TH]
							vis = visDOA(doa_gt, vad_gt, doa_pred, vad_pred, vad_TH=vad_TH, time_stamp=time_stamp)
							
							savename = result_dir + '/' + method_mode + '_' + source_num_mode + '_DOAvis_task' + str(task[0]) + '_' + str(ins_idx+1)
							vis.savefig(savename)
       
       
       
       
       
       
					#  MAYA HEATMAP PLOTS (SOME + ALL)
					# from scipy.optimize import linear_sum_assignment

					# def wrap_az_err(a, b):
					# 	return abs(((a - b + 180.0) % 360.0) - 180.0)

					# def fmt_metric(vec):
					# 	vec = vec.tolist()
					# 	if len(vec) == 7:
					# 		ACC, MDR, FAR, MAE_azi, MAE_ele, RMSE_azi, RMSE_ele = vec
					# 		return (f"ACC={ACC:.2f} MDR={MDR:.2f} FAR={FAR:.2f}\n"
					# 				f"MAE(ele,azi)=({MAE_azi:.1f},{MAE_ele:.1f})\n"
					# 				f"RMSE(ele,azi)=({RMSE_azi:.1f},{RMSE_ele:.1f})")
					# 	return f"metric len={len(vec)}"

					# # how many instances to plot when ins_mode == 'all'
					# PLOT_K = 4  # change to 1/2/5 as you like

					# # Determine nb from doa_gt
					# nb = doa_gt.shape[0]

					# if ins_mode == 'some':
					# 	plot_indices = [ins_idx]  # exactly the one you chose
					# else:
					# 	plot_indices = list(range(min(PLOT_K, nb)))  # first K instances

					# for plot_ins_idx in plot_indices:

					# 	doa_gt_ins = doa_gt[plot_ins_idx]      # (nt_gt, 2, ns_gt)
					# 	doa_pr_ins = doa_pred[plot_ins_idx]    # (nt_pr, 2, ns_pr)
					# 	ss_ins     = ss_pred[plot_ins_idx]     # (nt_ss, nele, nazi)

					# 	# choose time index for markers (static: nt_gt=1 anyway)
					# 	t_gt = 0
					# 	t_pr = 0

					# 	gt_el = doa_gt_ins[t_gt, 0, :]
					# 	gt_az = doa_gt_ins[t_gt, 1, :]

					# 	est_el = doa_pr_ins[t_pr, 0, :]
					# 	est_az = doa_pr_ins[t_pr, 1, :]

					# 	# Hungarian reorder EST to match GT (plotting only)
					# 	ns_gt = gt_az.shape[0]
					# 	ns_pr = est_az.shape[0]
					# 	C = np.zeros((ns_gt, ns_pr), dtype=np.float32)
					# 	for g in range(ns_gt):
					# 		for p in range(ns_pr):
					# 			C[g, p] = wrap_az_err(est_az[p], gt_az[g]) + abs(est_el[p] - gt_el[g])

					# 	g_idx, p_idx = linear_sum_assignment(C)

					# 	est_el_re = np.full((ns_gt,), np.nan, dtype=np.float32)
					# 	est_az_re = np.full((ns_gt,), np.nan, dtype=np.float32)
					# 	for gi, pi in zip(g_idx, p_idx):
					# 		est_el_re[gi] = est_el[pi]
					# 		est_az_re[gi] = est_az[pi]

					# 	# spectrum mean over time
					# 	ss_map = ss_ins.mean(axis=0)  # (nele, nazi)
					# 	nele, nazi = ss_map.shape
					# 	el_grid = np.linspace(0.0, 180.0, nele)
					# 	az_grid = np.linspace(-180.0, 180.0, nazi, endpoint=False)

					# 	# metrics text for this (T60,SNR) cell
					# 	txt_w  = fmt_metric(metrics[:, i, j])
					# 	txt_wo = fmt_metric(metrics_woDNN[:, i, j])

					# 	# plot
					# 	plt.figure(figsize=(6.8, 3.4))
					# 	plt.imshow(
					# 		ss_map,
					# 		origin="lower",
					# 		aspect="auto",
					# 		extent=[az_grid[0], az_grid[-1], el_grid[0], el_grid[-1]],
					# 		cmap="jet",
					# 	)
					# 	plt.scatter(gt_az, gt_el, c="k", marker="x", s=90, linewidths=2.0, label="GT")
					# 	plt.scatter(est_az_re, est_el_re, facecolors="white", edgecolors="k",
					# 				marker="o", s=65, linewidths=1.2, label="EST")

					# 	plt.xlabel("Azimuth [°]")
					# 	plt.ylabel("Elevation [°]")
					# 	plt.title(f"{method_mode}-{source_num_mode} | RT60={T60[i]}s SNR={SNR[j]}dB | ins={plot_ins_idx}")
					# 	plt.colorbar()
					# 	plt.legend(loc="upper right")

					# 	# metric box
					# 	plt.gcf().text(
					# 		0.02, 0.02,
					# 		"wDNN:\n"+txt_w+"\n\nwoDNN:\n"+txt_wo,
					# 		fontsize=9, va="bottom", ha="left",
					# 		bbox=dict(boxstyle="round", facecolor="white", alpha=0.85, edgecolor="0.5")
					# 	)

					# 	plt.tight_layout()

					# 	# filename: include mode, RT, SNR, instance
					# 	tag = "SOME" if ins_mode == "some" else "ALL"
					# 	outname = (f"{result_dir}/plots/HEAT_{tag}_{method_mode}_{source_num_mode}"
					# 			f"_RT{int(T60[i]*1000)}_SNR{int(SNR[j])}_ins{plot_ins_idx}.jpg")
					# 	plt.savefig(outname, dpi=200)
					# 	plt.close()
					# 	print("Saved:", outname)
     
					# choose instance index to plot
     
					# MAYA - HEATMAP PLOT
					# choose instance index to plot