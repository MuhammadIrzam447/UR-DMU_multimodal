import torch
import torch.utils.data as data
import os
import numpy as np
import utils 


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


class XDVideoMisM(data.DataLoader):
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