import librosa
import librosa.display
import matplotlib.pyplot as plt
import numpy as np
import torch
import time
import os
from tqdm import tqdm
from trainer.base_trainer_SIT import BaseTrainer
from model.DCCRN.ConvSTFT import ConvSTFT, ConviSTFT 
#from util.utils import compute_STOI, compute_PESQ
plt.switch_backend('agg')


class Trainer(BaseTrainer):
    def __init__(
            self,
            config,
            resume: bool,
            model_t,
            model_bs,
            model,
            loss_function,
            optimizer,
            train_dataloader,
            validation_dataloader,
    ):
        super(Trainer, self).__init__(config, resume, model_t, model_bs, model, loss_function, optimizer)
        self.train_data_loader = train_dataloader
        self.validation_data_loader = validation_dataloader


    def _train_epoch(self, epoch):
        loss_total = 0.0
        num_batchs = len(self.train_data_loader)
        start_time = time.time()
        with tqdm(total = num_batchs) as pbar:
            for i, (mixture, clean, dual_l, dual_r, name) in enumerate(self.train_data_loader):
                mixture = mixture.to(self.device).squeeze(1).type(torch.cuda.FloatTensor)
                clean = clean.to(self.device).squeeze(1).type(torch.cuda.FloatTensor)
                dual_l = dual_l.to(self.device).squeeze(1).type(torch.cuda.FloatTensor)
                dual_r = dual_r.to(self.device).squeeze(1).type(torch.cuda.FloatTensor)

                self.optimizer.zero_grad()
                _, _, t1_l, t1_r, t2_l, t2_r, t3_l, t3_r, t4_l, t4_r = self.model_t(dual_l, dual_r)
                _, _, b1_l, b1_r, b2_l, b2_r, b3_l, b3_r, b4_l, b4_r = self.model_bs(mixture)
                enhanced_cpl, enhanced, e1_l, e1_r, e2_l, e2_r, e3_l, e3_r, e4_l, e4_r = self.model(mixture)
                loss0 = self.model.loss(enhanced, clean, mode='SiSNR')
                #loss, _, _ = self.model.loss(enhanced_cpl, clean, mode='Mix')
                e1_l_t = TaKD(e1_l, t1_l)
                e1_l_b = TaKD(e1_l, b1_l)
                e1_r_t = TaKD(e1_r, t1_r)
                e1_r_b = TaKD(e1_r, b1_r)
				
                e2_l_t = TaKD(e2_l, t2_l)
                e2_l_b = TaKD(e2_l, b2_l)
                e2_r_t = TaKD(e2_r, t2_r)
                e2_r_b = TaKD(e2_r, b2_r)
				
                e3_l_t = TaKD(e3_l, t3_l)
                e3_l_b = TaKD(e3_l, b3_l)
                e3_r_t = TaKD(e3_r, t3_r)
                e3_r_b = TaKD(e3_r, b3_r)
				
                e4_l_t = TaKD(e4_l, t4_l)
                e4_l_b = TaKD(e4_l, b4_l)
                e4_r_t = TaKD(e4_r, t4_r)
                e4_r_b = TaKD(e4_r, b4_r)
                

                loss1 = 0.5 * (KDL(t1_l, b1_l, e1_l_t, e1_l_b) + KDL(t1_r, b1_r, e1_r_t, e1_l_b))
                loss2 = 0.5 * (KDL(t2_l, b2_l, e2_l_t, e2_l_b) + KDL(t2_r, b2_r, e2_r_t, e2_r_b))
                loss3 = 0.5 * (KDL(t3_l, b3_l, e3_l_t, e3_l_b) + KDL(t3_r, b3_r, e3_r_t, e3_r_b))
                loss4 = 0.5 * (KDL(t4_l, b4_l, e4_l_t, e4_l_b) + KDL(t4_r, b4_r, e4_r_t, e4_r_b))
                loss = loss0 + 0.25*(loss1 + loss2 + loss3 + loss4)

                #print(loss)
                #print(amp_loss)
                #print(phase_loss)
                loss.backward()
                self.optimizer.step()

                loss_total += loss.item()
                pbar.update(1)
            end_time = time.time()
            dl_len = len(self.train_data_loader)
            print("loss:", loss_total / dl_len)         
            self.writer.add_scalar(f"Train/Loss", loss_total / dl_len, epoch)
 
    @torch.no_grad()
    def _validation_epoch(self, epoch):
        loss_total = 0.0
        visualize_audio_limit = self.validation_custom_config["visualize_audio_limit"]
        visualize_waveform_limit = self.validation_custom_config["visualize_waveform_limit"]
        visualize_spectrogram_limit = self.validation_custom_config["visualize_spectrogram_limit"]
        num_batchs = len(self.train_data_loader)
        sample_length = self.validation_custom_config["sample_length"]
        self.stft = ConvSTFT(512, 256, 512, 'hamming', 'complex').to(self.device)
        self.istft = ConviSTFT(400, 100, 512, 'hamming', 'complex').to(self.device)
        stoi_c_n = []  # clean and noisy
        stoi_c_e = []  # clean and enhanced
        pesq_c_n = []
        pesq_c_e = []
        with tqdm(total = num_batchs) as pbar:  
            for i, (mixture, clean, dual_l, dual_r, name) in enumerate(self.validation_data_loader):
                assert len(name) == 1, "Only support batch size is 1 in enhancement stage."
                name = name[0]
                padded_length = 0

                mixture = mixture.to(self.device).squeeze(1).type(torch.cuda.FloatTensor)
                clean = clean.to(self.device).squeeze(1).type(torch.cuda.FloatTensor)
                dual_l = dual_l.to(self.device).squeeze(1).type(torch.cuda.FloatTensor)
                dual_r = dual_r.to(self.device).squeeze(1).type(torch.cuda.FloatTensor)               
                _, _, t1_l, t1_r, t2_l, t2_r, t3_l, t3_r, t4_l, t4_r = self.model_t(dual_l, dual_r)
                _, _, b1_l, b1_r, b2_l, b2_r, b3_l, b3_r, b4_l, b4_r = self.model_bs(mixture)
                enhanced_cpl, enhanced, e1_l, e1_r, e2_l, e2_r, e3_l, e3_r, e4_l, e4_r = self.model(mixture)
                loss0 = self.model.loss(enhanced, clean, mode='SiSNR')
                e1_l_t = TaKD(e1_l, t1_l)
                e1_l_b = TaKD(e1_l, b1_l)
                e1_r_t = TaKD(e1_r, t1_r)
                e1_r_b = TaKD(e1_r, b1_r)
				
                e2_l_t = TaKD(e2_l, t2_l)
                e2_l_b = TaKD(e2_l, b2_l)
                e2_r_t = TaKD(e2_r, t2_r)
                e2_r_b = TaKD(e2_r, b2_r)
				
                e3_l_t = TaKD(e3_l, t3_l)
                e3_l_b = TaKD(e3_l, b3_l)
                e3_r_t = TaKD(e3_r, t3_r)
                e3_r_b = TaKD(e3_r, b3_r)
				
                e4_l_t = TaKD(e4_l, t4_l)
                e4_l_b = TaKD(e4_l, b4_l)
                e4_r_t = TaKD(e4_r, t4_r)
                e4_r_b = TaKD(e4_r, b4_r)
                

                loss1 = 0.5 * (KDL(t1_l, b1_l, e1_l_t, e1_l_b) + KDL(t1_r, b1_r, e1_r_t, e1_l_b))
                loss2 = 0.5 * (KDL(t2_l, b2_l, e2_l_t, e2_l_b) + KDL(t2_r, b2_r, e2_r_t, e2_r_b))
                loss3 = 0.5 * (KDL(t3_l, b3_l, e3_l_t, e3_l_b) + KDL(t3_r, b3_r, e3_r_t, e3_r_b))
                loss4 = 0.5 * (KDL(t4_l, b4_l, e4_l_t, e4_l_b) + KDL(t4_r, b4_r, e4_r_t, e4_r_b))
                loss = loss0 + 0.25*(loss1 + loss2 + loss3 + loss4)

                #print(enhanced_cpl)
                #loss = self.model.loss(enhanced, clean, mode='SiSNR')
                #loss, _, _ = self.model.loss(enhanced_cpl, clean, mode='Mix')
                loss_total += loss.item()

                #enhanced = self.istft(enhanced_cpl, None)


                enhanced = enhanced.reshape(-1).cpu().numpy()
                clean = clean.cpu().numpy().reshape(-1)
                mixture = mixture.cpu().numpy().reshape(-1)
                assert len(mixture) == len(enhanced) == len(clean)


            # Visualize audio
                if i <= visualize_audio_limit:
                    self.writer.add_audio(f"Speech/{name}_Noisy", mixture, epoch, sample_rate=16000)
                    self.writer.add_audio(f"Speech/{name}_Enhanced", enhanced, epoch, sample_rate=16000)
                    self.writer.add_audio(f"Speech/{name}_Clean", clean, epoch, sample_rate=16000)

                # Visualize waveform
                if i <= visualize_waveform_limit:
                    fig, ax = plt.subplots(3, 1)
                    for j, y in enumerate([mixture, enhanced, clean]):
                        ax[j].set_title("mean: {:.3f}, std: {:.3f}, max: {:.3f}, min: {:.3f}".format(
                            np.mean(y),
                            np.std(y),
                            np.max(y),
                            np.min(y)
                        ))
                        librosa.display.waveplot(y, sr=16000, ax=ax[j])
                    plt.tight_layout()
                    self.writer.add_figure(f"Waveform/{name}", fig, epoch)

            # Visualize spectrogram
                noisy_mag, _ = librosa.magphase(librosa.stft(mixture, n_fft=320, hop_length=160, win_length=320))
                enhanced_mag, _ = librosa.magphase(librosa.stft(enhanced, n_fft=320, hop_length=160, win_length=320))
                clean_mag, _ = librosa.magphase(librosa.stft(clean, n_fft=320, hop_length=160, win_length=320))

                if i <= visualize_spectrogram_limit:
                    fig, axes = plt.subplots(3, 1, figsize=(6, 6))
                    for k, mag in enumerate([
                        noisy_mag,
                        enhanced_mag,
                        clean_mag,
                    ]):
                        axes[k].set_title(f"mean: {np.mean(mag):.3f}, "
                                          f"std: {np.std(mag):.3f}, "
                                          f"max: {np.max(mag):.3f}, "
                                          f"min: {np.min(mag):.3f}")
                        librosa.display.specshow(librosa.amplitude_to_db(mag), cmap="magma", y_axis="linear", ax=axes[k], sr=16000)
                    plt.tight_layout()
                    self.writer.add_figure(f"Spectrogram/{name}", fig, epoch)
                output_path = os.path.join('H:/data3/r4', f"{name}.wav")
                librosa.output.write_wav(output_path, enhanced, sr = 16000)
                pbar.update(1)
            # Metric
            # stoi_c_n.append(compute_STOI(clean, mixture, sr=16000))
            # stoi_c_e.append(compute_STOI(clean, enhanced, sr=16000))
            # pesq_c_n.append(compute_PESQ(clean, mixture, sr=16000))
            # pesq_c_e.append(compute_PESQ(clean, enhanced, sr=16000))

        # get_metrics_ave = lambda metrics: np.sum(metrics) / len(metrics)
        # self.writer.add_scalars(f"Metric/STOI", {
            # "Clean and noisy": get_metrics_ave(stoi_c_n),
            # "Clean and enhanced": get_metrics_ave(stoi_c_e)
        # }, epoch)
        # self.writer.add_scalars(f"Metric/PESQ", {
            # "Clean and noisy": get_metrics_ave(pesq_c_n),
            # "Clean and enhanced": get_metrics_ave(pesq_c_e)
        # }, epoch)

        score = loss_total
        return score

def KDL(t, bs, s_t, s_b):
    kl = torch.nn.L1Loss()

    loss = kl(t, s_t) / kl(bs, s_b)
    return loss

def TaKD(s, t):
    softmax = torch.nn.Softmax(dim=-1)
    atten = torch.matmul(t.permute(0, 1, 3, 2), s)
    s = torch.matmul(s, softmax(atten))
    return s