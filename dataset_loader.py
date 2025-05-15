import torch
import torch.utils.data as data
import os
import numpy as np
import utils
import torch.nn as nn
import random


class UCF_crime(data.DataLoader):
    def __init__(self, root_dir, modal, mode, num_segments, len_feature, seed=-1, is_normal=None):
        if seed >= 0:
            utils.set_seed(seed)
        self.mode = mode
        self.modal = modal
        self.num_segments = num_segments
        self.len_feature = len_feature
        split_path = os.path.join('list','UCF_{}.list'.format(self.mode))
        split_file = open(split_path, 'r')
        self.vid_list = []
        for line in split_file:
            self.vid_list.append(line.split())
        split_file.close()
        if self.mode == "Train":
            if is_normal is True:
                self.vid_list = self.vid_list[8100:]
            elif is_normal is False:
                self.vid_list = self.vid_list[:8100]
            else:
                assert (is_normal == None)
                print("Please sure is_normal=[True/False]")
                self.vid_list=[]
    def __len__(self):
        return len(self.vid_list)

    def __getitem__(self, index):
        
        if self.mode == "Test":
            data,label,name = self.get_data(index)
            return data,label,name
        else:
            data,label = self.get_data(index)
            return data,label

    def get_data(self, index):
        vid_info = self.vid_list[index][0]  
        name = vid_info.split("/")[-1].split("_x264")[0]
        video_feature = np.load(vid_info).astype(np.float32)   

        if "Normal" in vid_info.split("/")[-1]:
            label = 0
        else:
            label = 1
        if self.mode == "Train":
            new_feat = np.zeros((self.num_segments, video_feature.shape[1])).astype(np.float32)
            r = np.linspace(0, len(video_feature), self.num_segments + 1, dtype = np.int)
            for i in range(self.num_segments):
                if r[i] != r[i+1]:
                    new_feat[i,:] = np.mean(video_feature[r[i]:r[i+1],:], 0)
                else:
                    new_feat[i:i+1,:] = video_feature[r[i]:r[i]+1,:]
            video_feature = new_feat
        if self.mode == "Test":
            return video_feature, label, name      
        else:
            return video_feature, label      
class XDVideo(data.DataLoader):
    def __init__(self, root_dir, mode, modal, num_segments, len_feature, seed=-1, is_normal=None):
        if seed >= 0:
            utils.set_seed(seed)
        self.data_path=root_dir
        self.mode=mode
        self.modal=modal
        self.num_segments = num_segments
        self.len_feature = len_feature
        if self.modal == 'all':
            self.feature_path = []
            if self.mode == "Train":
                for _modal in ['RGB', 'Flow']:
                    self.feature_path.append(os.path.join(self.data_path, "i3d-features",_modal))
            else:
                for _modal in ['RGBTest', 'FlowTest']:
                    self.feature_path.append(os.path.join(self.data_path, "i3d-features",_modal))
        ############### New Code for multimodal ###############
        elif self.modal == 'multimodal':
            split_path_audio = os.path.join("list", 'XD_audio_{}.list'.format(self.mode))
            split_file_audio = open(split_path_audio, 'r', encoding="utf-8")
            self.aud_list = []
            for line in split_file_audio:
                # Split the line
                # print(f"Audio file line: {line}")
                line_split = line.split()
                # print(f"audio split line: {line_split}")
                # Add the line five times to the list
                for _ in range(5):
                    self.aud_list.append(line_split)
            split_file_audio.close()
        #######################################################
        else:
            self.feature_path = os.path.join(self.data_path, modal)
        split_path = os.path.join("list",'XD_{}.list'.format(self.mode))
        split_file = open(split_path, 'r',encoding="utf-8")
        self.vid_list = []
        for line in split_file:
            self.vid_list.append(line.split())
        split_file.close()
        if self.mode == "Train":
            if is_normal is True:
                self.vid_list = self.vid_list[9525:]
            elif is_normal is False:
                self.vid_list = self.vid_list[:9525]
            else:
                assert (is_normal == None)
                print("Please sure is_normal = [True/False]")
                self.vid_list=[]
        
    def __len__(self):
        return len(self.vid_list)

    def __getitem__(self, index):
        data,label = self.get_data(index)
        return data, label

    def get_data(self, index):
        vid_name = self.vid_list[index][0]
        vid_file = self.vid_list[index]
        # print(f"vid_name: {vid_name}")
        label=0
        if "_label_A" not in vid_name:
            label=1  
        # video_feature = np.load(os.path.join(self.feature_path[0], vid_name )).astype(np.float32)
        video_feature = np.load(vid_name).astype(np.float32)
        # print(f"video_feature.shape: {video_feature.shape}")
        ############### New Code for multimodal ###############
        if self.modal == 'multimodal':
            aud_name = self.aud_list[index]
            aud_file_name = self.aud_list[index][0]
            # print(f"Aud_name: {aud_file_name}")
            # audio_feature = np.load(os.path.join(self.data_path, "audio-features", aud_name)).astype(np.float32)
            audio_feature = np.load(aud_file_name).astype(np.float32)
            # print(f"audio_feature.shape: {audio_feature.shape}")
            min_len = min(video_feature.shape[0], audio_feature.shape[0])
            video_feature = video_feature[:min_len]
            audio_feature = audio_feature[:min_len]
            video_feature = np.concatenate((video_feature, audio_feature), axis=1)
            # print(f"concatinated feature: {video_feature.shape}")
        #######################################################
        if self.mode == "Train":
            new_feature = np.zeros((self.num_segments,self.len_feature)).astype(np.float32)
            sample_index = utils.random_perturb(video_feature.shape[0],self.num_segments)
            for i in range(len(sample_index)-1):
                if sample_index[i] == sample_index[i+1]:
                    new_feature[i,:] = video_feature[sample_index[i],:]
                else:
                    new_feature[i,:] = video_feature[sample_index[i]:sample_index[i+1],:].mean(0)
                    
            video_feature = new_feature
            # exit()
        return video_feature, label
class XDVideoSB(data.DataLoader): # Class for the single_branch
    def __init__(self, root_dir, mode, modal, num_segments, len_feature, seed=-1, is_normal=None):
        if seed >= 0:
            utils.set_seed(seed)
        self.data_path = root_dir
        self.mode = mode
        self.modal = modal
        self.num_segments = num_segments
        self.len_feature = len_feature
        self.audio_proj = nn.Linear(in_features=128, out_features=1024)  # <-- Adjust 128 based on actual audio dim

        if self.modal == 'single_branch':
            if self.mode == 'Test':
                split_path_audio = os.path.join("list", f'XD_audio_{self.mode}.list')
                with open(split_path_audio, 'r', encoding="utf-8") as split_file_audio:
                    self.aud_list = []
                    for line in split_file_audio:
                        line_split = line.split()
                        for _ in range(5):
                            self.aud_list.append(line_split)

                split_path = os.path.join("list", f'XD_{self.mode}.list')
                with open(split_path, 'r', encoding="utf-8") as split_file:
                    self.vid_list = [line.split() for line in split_file]

            elif self.mode == 'Train':
                split_path_vid = os.path.join("list", f'XD_SB_{self.mode}.list')
                with open(split_path_vid, 'r', encoding="utf-8") as split_file:
                    self.vid_list = [line.split() for line in split_file]
            else:
                raise ValueError(f"Incorrect mode: {self.mode}")
        else:
            raise ValueError(f"Incorrect Modality Settings: {self.modal}")

        if self.mode == "Train":
            if is_normal is True:
                self.vid_list = self.vid_list[9525 * 2:]
            elif is_normal is False:
                self.vid_list = self.vid_list[:9525 * 2]
            elif is_normal is None:
                print("Please ensure is_normal = [True/False]")
                self.vid_list = []
            else:
                raise ValueError("Invalid value for is_normal. Expected True, False, or None.")

    def __len__(self):
        return len(self.vid_list)

    def __getitem__(self, index):
        if self.mode == 'Train':
            data, label = self.get_data(index)
            return data, label
        elif self.mode == 'Test':
            video_feature, audio_feature, label = self.get_data(index)
            return video_feature, audio_feature, label
        else:
            raise NotImplementedError("Only Train and Test modes are supported.")

    def get_data(self, index):
        if self.mode == 'Train':
            vid_name = self.vid_list[index][0]
            # print(f"Training Filename: {vid_name}")
            label = 0 if "_label_A" in vid_name else 1
            video_feature = np.load(vid_name).astype(np.float32)
            if video_feature.shape[1] != 1024:
                video_feature = torch.from_numpy(video_feature)  # <-- Convert to torch.Tensor
                # video_feature = self.audio_proj(video_feature)
                video_feature = self.audio_proj(video_feature).detach().numpy() # <-- back to numpy

            new_feature = np.zeros((self.num_segments, self.len_feature), dtype=np.float32)
            sample_index = utils.random_perturb(video_feature.shape[0], self.num_segments)

            for i in range(len(sample_index) - 1):
                if sample_index[i] == sample_index[i + 1]:
                    new_feature[i, :] = video_feature[sample_index[i], :]
                else:
                    new_feature[i, :] = video_feature[sample_index[i]:sample_index[i + 1], :].mean(0)

            video_feature = new_feature
            return video_feature, label

        elif self.mode == 'Test':
            vid_name = self.vid_list[index][0]
            # print(f"Testing Filename: {vid_name}")
            label = 0 if "_label_A" in vid_name else 1
            video_feature = np.load(vid_name).astype(np.float32)

            aud_name = self.aud_list[index][0]
            # print(f"Testing Filename: {aud_name}")
            audio_feature = np.load(aud_name).astype(np.float32)
            if audio_feature.shape[1] != 128:
                raise ValueError(f"Feature dimension mismatch: expected {128}, got {audio_feature.shape[1]}")
            elif video_feature.shape[1] != 1024:
                raise ValueError(f"Feature dimension mismatch: expected {1024}, got {video_feature.shape[1]}")
            min_len = min(video_feature.shape[0], audio_feature.shape[0])
            video_feature = video_feature[:min_len]
            audio_feature = audio_feature[:min_len]

            # Project audio features to match video dimensions
            audio_feature = torch.from_numpy(audio_feature)  # <-- Convert to torch.Tensor
            audio_feature_proj = self.audio_proj(audio_feature)
            # print(f"shape of audio feature:{np.shape(audio_feature_proj)} shape of rgb feature:{np.shape(video_feature)} shape of label:{np.shape(label)} ")
            return video_feature, audio_feature_proj, label
        else:
            raise ValueError("Incorrect mode selected.")
class XDVideoMisM(data.DataLoader): # Class for Evaluting Missing Modalities
    def __init__(self, root_dir, mode, modal, num_segments, len_feature, seed=-1,
                 is_normal=None, mismodal=None, mispercentage=0):
        if seed >= 0:
            utils.set_seed(seed)
        self.data_path = root_dir
        self.mode = mode
        self.modal = modal
        self.num_segments = num_segments
        self.len_feature = len_feature
        self.mismodal = mismodal  # 'rgb' or 'audio'
        self.mispercentage = mispercentage  # 0-100

        if self.modal == 'all':
            self.feature_path = []
            if self.mode == "Train":
                for _modal in ['RGB', 'Flow']:
                    self.feature_path.append(os.path.join(self.data_path, "i3d-features", _modal))
            else:
                for _modal in ['RGBTest', 'FlowTest']:
                    self.feature_path.append(os.path.join(self.data_path, "i3d-features", _modal))

        elif self.modal == 'multimodal':
            split_path_audio = os.path.join("list", f'XD_audio_{self.mode}.list')
            with open(split_path_audio, 'r', encoding="utf-8") as split_file_audio:
                self.aud_list = []
                for line in split_file_audio:
                    line_split = line.split()
                    for _ in range(5):
                        self.aud_list.append(line_split)

        else:
            self.feature_path = os.path.join(self.data_path, modal)

        split_path = os.path.join("list", f'XD_{self.mode}.list')
        with open(split_path, 'r', encoding="utf-8") as split_file:
            self.vid_list = [line.split() for line in split_file]

        if self.mode == "Train":
            if is_normal is True:
                self.vid_list = self.vid_list[9525:]
            elif is_normal is False:
                self.vid_list = self.vid_list[:9525]
            else:
                assert (is_normal is None)
                print("Please ensure is_normal = [True/False]")
                self.vid_list = []

        self._init_modality_masking(seed)

    def _init_modality_masking(self, seed):
        self.masked_indices = set()
        if self.modal == 'multimodal' and self.mismodal in ['audio', 'rgb'] and self.mispercentage > 0:
            num_to_mask = int(len(self.vid_list) * self.mispercentage / 100)
            rng = np.random.default_rng(seed if seed >= 0 else None)
            self.masked_indices = set(rng.choice(len(self.vid_list), size=num_to_mask, replace=False))

    def __len__(self):
        return len(self.vid_list)

    def __getitem__(self, index):
        data, label = self.get_data(index)
        return data, label

    def get_data(self, index):
        vid_name = self.vid_list[index][0]
        label = 0 if "_label_A" in vid_name else 1
        video_feature = np.load(vid_name).astype(np.float32)

        if self.modal == 'multimodal':
            aud_file_name = self.aud_list[index][0]
            audio_feature = np.load(aud_file_name).astype(np.float32)

            # Align lengths
            min_len = min(video_feature.shape[0], audio_feature.shape[0])
            video_feature = video_feature[:min_len]
            audio_feature = audio_feature[:min_len]

            # Apply sample-level modality masking
            if index in self.masked_indices:
                if self.mismodal == 'rgb':
                    video_feature = np.zeros_like(video_feature)
                elif self.mismodal == 'audio':
                    audio_feature = np.zeros_like(audio_feature)

            # Concatenate features
            video_feature = np.concatenate((video_feature, audio_feature), axis=1)

        if self.mode == "Train":
            new_feature = np.zeros((self.num_segments, self.len_feature)).astype(np.float32)
            sample_index = utils.random_perturb(video_feature.shape[0], self.num_segments)
            for i in range(len(sample_index) - 1):
                if sample_index[i] == sample_index[i + 1]:
                    new_feature[i, :] = video_feature[sample_index[i], :]
                else:
                    new_feature[i, :] = video_feature[sample_index[i]:sample_index[i + 1], :].mean(0)
            video_feature = new_feature

        return video_feature, label

# class XDVideoCorrM(data.DataLoader):  # Class for Evaluating Corrupted Modalities
#     def __init__(self, root_dir, mode, modal, num_segments, len_feature, seed=-1,
#                  is_normal=None, mismodal=None, mispercentage=0):
#
#         if seed >= 0:
#             utils.set_seed(seed)
#
#         self.data_path = root_dir
#         self.mode = mode
#         self.modal = modal
#         self.num_segments = num_segments
#         self.len_feature = len_feature
#         self.mismodal = mismodal  # 'rgb' or 'audio'
#         self.mispercentage = mispercentage  # 0-100
#
#         # Define file paths
#         clean_file_rgb = os.path.join("corrputed_list", 'test_rgb_clean.list')
#         corrupted_file_rgb = os.path.join("corrputed_list", 'test_rgb_corrp_motion.list')
#         clean_file_audio = os.path.join("corrputed_list", 'test_audio_clean.list')
#         corrupted_file_audio = os.path.join("corrputed_list", 'test_audio_corrp_babble_3.list')
#
#         if self.modal == 'multimodal':
#             self.rgb_list = []
#             self.audio_list = []
#
#             with open(clean_file_rgb, 'r') as f_clean_rgb, open(corrupted_file_rgb, 'r') as f_corr_rgb, \
#                  open(clean_file_audio, 'r') as f_clean_audio, open(corrupted_file_audio, 'r') as f_corr_audio:
#
#                 clean_rgb_lines = f_clean_rgb.readlines()
#                 corr_rgb_lines = f_corr_rgb.readlines()
#                 clean_audio_lines = f_clean_audio.readlines()
#                 corr_audio_lines = f_corr_audio.readlines()
#
#                 clean_audio_lines = [line for line in clean_audio_lines for _ in range(5)]
#                 corr_audio_lines = [line for line in corr_audio_lines for _ in range(5)]
#
#                 total_samples = len(clean_rgb_lines)
#                 num_corrupted = int((self.mispercentage / 100.0) * total_samples)
#                 rng = random.Random(seed if seed >= 0 else None)
#                 corrupted_indices = set(rng.sample(range(total_samples), num_corrupted))
#
#                 for i in range(total_samples):
#                     if self.mismodal == 'rgb':
#                         rgb_line = corr_rgb_lines[i] if i in corrupted_indices else clean_rgb_lines[i]
#                         audio_line = clean_audio_lines[i]
#                     elif self.mismodal == 'audio':
#                         rgb_line = clean_rgb_lines[i]
#                         audio_line = corr_audio_lines[i] if i in corrupted_indices else clean_audio_lines[i]
#                     else:
#                         raise ValueError(f"Incorrect mode: {self.mismodal}")
#
#                     self.rgb_list.append(rgb_line.strip().split())
#                     self.audio_list.append(audio_line.strip().split())
#
#
#             # 🔍 Print debugging and verification info
#             print("🔍=== XDVideoMisM Dataset Info ===")
#             print(f"Total Samples      : {total_samples}")
#             print(f"Corrupt RGB Samples: {len(corrupted_indices) if self.mismodal == 'rgb' else 0}")
#             print(f"Corrupt Audio Samp.: {len(corrupted_indices) if self.mismodal == 'audio' else 0}")
#             print(f"RGB List Length    : {len(self.rgb_list)}")
#             print(f"Audio List Length  : {len(self.audio_list)}")
#             print("================================")
#
#             self.vid_list = self.rgb_list  # or a combined version for length reference
#         else:
#             # Single modality: just load clean data
#             self.feature_path = os.path.join(self.data_path, modal)
#             with open(os.path.join("list/", f'XD_{modal}.txt'), 'r') as f:
#                 self.vid_list = [line.strip().split() for line in f.readlines()]
#
#         if self.mode == "Train":
#             if is_normal is True:
#                 self.vid_list = self.vid_list[9525:]
#             elif is_normal is False:
#                 self.vid_list = self.vid_list[:9525]
#             else:
#                 assert (is_normal is None)
#                 print("Please ensure is_normal = [True/False]")
#                 self.vid_list = []
#
#         self._init_modality_masking(seed)
#
#     def _init_modality_masking(self, seed):
#         self.masked_indices = set()
#         # Optional: Add logic to mark masked samples (e.g. to skip or handle specially during training) ignore for now
#
#     def __len__(self):
#         return len(self.vid_list)
#
#     def __getitem__(self, index):
#         data, label = self.get_data(index)
#         return data, label
#
#     def get_data(self, index):
#         if self.modal == 'multimodal':
#             rgb_path = self.rgb_list[index][0]
#             audio_path = self.audio_list[index][0]
#             label = 0 if "_label_A" in rgb_path else 1
#
#             rgb_feat = np.load(rgb_path).astype(np.float32)
#             audio_feat = np.load(audio_path).astype(np.float32)
#
#             # Align lengths
#             min_len = min(rgb_feat.shape[0], audio_feat.shape[0])
#             rgb_feat = rgb_feat[:min_len]
#             audio_feat = audio_feat[:min_len]
#             video_feature = np.concatenate((rgb_feat, audio_feat), axis=1)
#         else:
#             vid_path = self.vid_list[index][0]
#             label = 0 if "_label_A" in vid_path else 1
#             video_feature = np.load(vid_path).astype(np.float32)
#
#         return video_feature, label

class XDVideoCorrM(data.DataLoader):  # Class for Evaluating Corrupted Modalities
    def __init__(self, root_dir, mode, modal, num_segments, len_feature, seed=-1,
                 is_normal=None, mismodal=None, mispercentage=0):

        if seed >= 0:
            import utils  # Assuming utils.set_seed exists
            utils.set_seed(seed)

        self.data_path = root_dir
        self.mode = mode
        self.modal = modal
        self.num_segments = num_segments
        self.len_feature = len_feature
        self.mismodal = mismodal  # 'rgb' or 'audio'
        self.mispercentage = mispercentage  # 0-100

        # Define file paths
        clean_file_rgb = os.path.join("corrputed_list", 'test_rgb_clean.list')
        corrupted_file_rgb = os.path.join("corrputed_list", 'test_rgb_corrp_fog.list')
        clean_file_audio = os.path.join("corrputed_list", 'test_audio_clean.list')
        corrupted_file_audio = os.path.join("corrputed_list", 'test_audio_corrp_random_dropout_3.list')

        if self.modal == 'multimodal':
            self.rgb_list = []
            self.audio_list = []

            with open(clean_file_rgb, 'r') as f_clean_rgb, open(corrupted_file_rgb, 'r') as f_corr_rgb, \
                 open(clean_file_audio, 'r') as f_clean_audio, open(corrupted_file_audio, 'r') as f_corr_audio:

                clean_rgb_lines = f_clean_rgb.readlines()
                corr_rgb_lines = f_corr_rgb.readlines()
                clean_audio_lines = f_clean_audio.readlines()
                corr_audio_lines = f_corr_audio.readlines()

                total_audio_samples = len(clean_audio_lines)
                total_rgb_samples = len(clean_rgb_lines)

                num_corrupted = int((self.mispercentage / 100.0) * total_audio_samples)
                rng = random.Random(seed if seed >= 0 else None)
                corrupted_indices = set(rng.sample(range(total_audio_samples), num_corrupted))

                for i in range(total_audio_samples):
                    # Select audio line based on corruption and mismodal
                    if self.mismodal == 'audio':
                        audio_line = corr_audio_lines[i] if i in corrupted_indices else clean_audio_lines[i]
                        # RGB lines block (5 lines per audio line)
                        rgb_start = i * 5
                        rgb_end = rgb_start + 5
                        # For mismodal == 'rgb' corruption, but since mismodal == 'audio', rgb always clean here
                        rgb_lines_block = clean_rgb_lines[rgb_start:rgb_end]
                    elif self.mismodal == 'rgb':
                        audio_line = clean_audio_lines[i]
                        rgb_start = i * 5
                        rgb_end = rgb_start + 5
                        # If the rgb block is corrupted for this index
                        if i in corrupted_indices:
                            rgb_lines_block = corr_rgb_lines[rgb_start:rgb_end]
                        else:
                            rgb_lines_block = clean_rgb_lines[rgb_start:rgb_end]
                    else:
                        raise ValueError(f"Incorrect mode: {self.mismodal}")

                    # Append 5 RGB lines
                    for rgb_line in rgb_lines_block:
                        self.rgb_list.append(rgb_line.strip().split())
                    # Append the audio line 5 times
                    self.audio_list.extend([audio_line.strip().split()] * 5)

            # 🔍 Print debugging and verification info
            print("🔍=== XDVideoMisM Dataset Info ===")
            print(f"Total Audio Samples : {total_audio_samples}")
            print(f"Total RGB Samples   : {total_rgb_samples}")
            print(f"Corrupt RGB Samples : {len(corrupted_indices) if self.mismodal == 'rgb' else 0}")
            print(f"Corrupt Audio Samp. : {len(corrupted_indices) if self.mismodal == 'audio' else 0}")
            print(f"RGB List Length     : {len(self.rgb_list)}")
            print(f"Audio List Length   : {len(self.audio_list)}")
            print("================================")

            self.vid_list = self.rgb_list

        else:
            # Single modality: just load clean data
            self.feature_path = os.path.join(self.data_path, modal)
            with open(os.path.join("list/", f'XD_{modal}.txt'), 'r') as f:
                self.vid_list = [line.strip().split() for line in f.readlines()]

        if self.mode == "Train":
            if is_normal is True:
                self.vid_list = self.vid_list[9525:]
            elif is_normal is False:
                self.vid_list = self.vid_list[:9525]
            else:
                assert (is_normal is None)
                print("Please ensure is_normal = [True/False]")
                self.vid_list = []


    def __len__(self):
        return len(self.vid_list)

    def __getitem__(self, index):
        data, label = self.get_data(index)
        return data, label

    def get_data(self, index):
        if self.modal == 'multimodal':
            rgb_path = self.rgb_list[index][0]
            audio_path = self.audio_list[index][0]
            label = 0 if "_label_A" in rgb_path else 1

            rgb_feat = np.load(rgb_path).astype(np.float32)
            audio_feat = np.load(audio_path).astype(np.float32)

            # Align lengths
            min_len = min(rgb_feat.shape[0], audio_feat.shape[0])
            rgb_feat = rgb_feat[:min_len]
            audio_feat = audio_feat[:min_len]
            video_feature = np.concatenate((rgb_feat, audio_feat), axis=1)
        else:
            vid_path = self.vid_list[index][0]
            label = 0 if "_label_A" in vid_path else 1
            video_feature = np.load(vid_path).astype(np.float32)

        return video_feature, label
