import torch
import numpy as np
import torch
import random
import pickle
import soundfile
from copy import deepcopy

def load_burst_data_from_npz(npz_file_path):
	""" Load burst noise data from .npz file
	Args:
		npz_file_path: path to the .npz file containing burst_positions
	Returns:
		burst_positions: list of burst dictionaries, or None if file not found/error
	"""
	try:
		data = np.load(npz_file_path, allow_pickle=True)
		if 'burst_positions' in data:
			burst_positions = data['burst_positions']
			# Handle both cases: already a list or needs conversion
			if isinstance(burst_positions, list):
				print(f"Loaded burst_positions list, total bursts: {len(burst_positions)}")
				print(f"data is: {burst_positions}")
				return burst_positions
			else:
				return burst_positions.tolist()
		else:
			print(f"Warning: 'burst_positions' not found in {npz_file_path}")
			return None
	except Exception as e:
		print(f"Warning: Could not load burst data from {npz_file_path}: {e}")
		return None

## for training process 

def set_seed(seed):
	""" Function: fix random seed.
	"""
	torch.manual_seed(seed)
	torch.cuda.manual_seed_all(seed)
	torch.cuda.manual_seed(seed)
	torch.backends.cudnn.deterministic = True
	torch.backends.cudnn.benchmark = False
	
	np.random.seed(seed)
	random.seed(seed)

def set_random_seed(seed):

    np.random.seed(seed)
    random.seed(seed)

def get_learning_rate(optimizer):
    """ Function: get learning rates from optimizer
    """ 
    lr = []
    for param_group in optimizer.param_groups:
        lr += [param_group['lr']]
    return lr

def set_learning_rate(epoch, lr_init, step, gamma):
    """ Function: adjust learning rates 
    """ 
    lr = lr_init*pow(gamma, epoch/step)
    return lr

## for data number

def detect_infnan(data, mode='torch'):
    """ Function: check whether there is inf/nan in the element of data or not
    """ 
    if mode == 'troch':
        inf_flag = torch.isinf(data)
        nan_flag = torch.isnan(data)
    elif mode == 'np':
        inf_flag = np.isinf(data)
        nan_flag = np.isnan(data)
    else:
        raise Exception('Detect infnan mode unrecognized')
    if (True in inf_flag):
        raise Exception('INF exists in data')
    if (True in nan_flag):
        raise Exception('NAN exists in data')


## for room acoustic data saving and reading 

def save_file(mic_signal, acoustic_scene, sig_path, acous_path):
    
    if sig_path is not None:
        soundfile.write(sig_path, mic_signal, acoustic_scene.fs)

    if acous_path is not None:
        file = open(acous_path,'wb')
        file.write(pickle.dumps(acoustic_scene.__dict__))
        file.close()

def load_file(acoustic_scene, sig_path, acous_path):

    burst_data = None

    if sig_path is not None:
        mic_signal, fs = soundfile.read(sig_path)

    if acous_path is not None:
        file = open(acous_path,'rb')
        dataPickle = file.read()
        file.close()
        acoustic_scene.__dict__ = pickle.loads(dataPickle)

        # Load burst data from the same npz file
        burst_data = load_burst_data_from_npz(acous_path)

    if (sig_path is not None) & (acous_path is not None):
        return mic_signal, acoustic_scene, burst_data
    elif (sig_path is not None) & (acous_path is None):
        return mic_signal, None, None
    elif (sig_path is None) & (acous_path is not None):
        return None, acoustic_scene, burst_data
    else:
        return None, None, None

def forgetting_norm(input, num_frame_set=None):
    """
        Function: Using the mean value of the near frames to normalization
        Args:
            input: feature [B, C, F, T]
            num_frame_set: length of the training time frames, used for calculating smooth factor
        Returns:
            normed feature
        Ref: Online Monaural Speech Enhancement using Delayed Subband LSTM, INTERSPEECH, 2020
    """
    assert input.ndim == 4
    batch_size, num_channels, num_freqs, num_frames = input.size()
    input = input.reshape(batch_size, num_channels * num_freqs, num_frames)

    if num_frame_set == None:
        num_frame_set = deepcopy(num_frames)

    mu = 0
    mu_list = []
    for frame_idx in range(num_frames):
        if frame_idx<=num_frame_set:
            alpha = (frame_idx - 1) / (frame_idx + 1)
        else:
            alpha = (num_frame_set - 1) / (num_frame_set + 1)
        current_frame_mu = torch.mean(input[:, :, frame_idx], dim=1).reshape(batch_size, 1) # [B, 1]
        mu = alpha * mu + (1 - alpha) * current_frame_mu
        mu_list.append(mu)
    mu = torch.stack(mu_list, dim=-1) # [B, 1, T]
    output = mu.reshape(batch_size, 1, 1, num_frames)

    return output

def save_file(mic_signal, acoustic_scene, sig_path, acous_path):
    
    if sig_path is not None:
        soundfile.write(sig_path, mic_signal, acoustic_scene.fs)

    if acous_path is not None:
        file = open(acous_path,'wb')
        file.write(pickle.dumps(acoustic_scene.__dict__))
        file.close()

    # data_path = save_dir+'/'+name+'.mat'
	# scipy.io.savemat(data_path, {'mic_signals': mic_signals, 'acoustic_scene': acoustic_scene})


# def load_file(acoustic_scene, sig_path, acous_path):

#     burst_data = None

#     if sig_path is not None:
#         mic_signal, fs = soundfile.read(sig_path)

#     if acous_path is not None:
#         file = open(acous_path,'rb')
#         dataPickle = file.read()
#         file.close()
#         acoustic_scene.__dict__ = pickle.loads(dataPickle)

#         # Load burst data from the same npz file
#         burst_data = load_burst_data_from_npz(acous_path)

#     if (sig_path is not None) & (acous_path is not None):
#         return mic_signal, acoustic_scene, burst_data
#     elif (sig_path is not None) & (acous_path is None):
#         return mic_signal, None
#     elif (sig_path is None) & (acous_path is not None):
#         return acoustic_scene, burst_data

    ## When reading mat file, the array_setup cannot present normally
    # data = scipy.io.loadmat(load_dir+'/'+name+'.mat')
	# mic_signals = data['mic_signals']
	# acoustic_scene0 =data['acoustic_scene'][0,0]
	# keys = acoustic_scene0.dtype.names
    # for idx in range(len(keys)):
	# 	key = keys[idx]
	# 	value = acoustic_scene0[key]
	# 	sh = value.shape
	# 	if len(sh)==2:
	# 		if (sh[0]==1) & (sh[1]==1):
	# 			value = value[0,0]
	# 		elif (sh[0]==1) & (sh[1]>1):
	# 			value = value[0,:]
	# 	print(key ,value)
	# 	acoustic_scene.__dict__[key] = value