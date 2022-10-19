import torch
import torch.nn as nn
from model.DCCRN.ConvSTFT import ConvSTFT, ConviSTFT
#sys.path.append(os.path.dirname(__file__))
class CausalConvBlock(nn.Module):
    def __init__(self, in_channels, out_channels):
        super().__init__()
        self.conv = nn.Conv2d(
            in_channels=in_channels,
            out_channels=out_channels,
            kernel_size=(3, 2),
            stride=(2, 1),
            padding=(0, 1)
        )
        self.norm = nn.BatchNorm2d(num_features=out_channels)
        self.activation = nn.ELU()

    def forward(self, x):
        """
        2D Causal convolution.
        Args:
            x: [B, C, F, T]

        Returns:
            [B, C, F, T]
        """
        x = self.conv(x)
        x = x[:, :, :, :-1]  # chomp size
        x = self.norm(x)
        x = self.activation(x)
        return x

class GLayerNorm2d(nn.Module):
    
    def __init__(self, in_channel, eps=1e-12):
        super(GLayerNorm2d, self).__init__()
        self.eps = eps 
        self.beta = nn.Parameter(torch.ones([1, in_channel,1,1]))
        self.gamma = nn.Parameter(torch.zeros([1, in_channel,1,1]))
    
    def forward(self,inputs):
        mean = torch.mean(inputs,[1,2,3], keepdim=True)
        var = torch.var(inputs,[1,2,3], keepdim=True)
        outputs = (inputs - mean)/ torch.sqrt(var+self.eps)*self.beta+self.gamma
        return outputs

class CausalTransConvBlock(nn.Module):
    def __init__(self, in_channels, out_channels, is_last=False, output_padding=(0, 0)):
        super().__init__()
        self.conv = nn.ConvTranspose2d(
            in_channels=in_channels,
            out_channels=out_channels,
            kernel_size=(3, 2),
            stride=(2, 1),
            output_padding=output_padding
        )
        self.norm = nn.BatchNorm2d(num_features=out_channels)
        if is_last:
            self.activation = nn.ReLU()
        else:
            self.activation = nn.ELU()

    def forward(self, x):
        """
        2D Causal convolution.
        Args:
            x: [B, C, F, T]

        Returns:
            [B, C, F, T]
        """
        x = self.conv(x)
        x = x[:, :, :, :-1]  # chomp size
        x = self.norm(x)
        x = self.activation(x)
        return x


class Model(nn.Module):
    """
    Input: [batch size, channels=1, T, n_fft]
    Output: [batch size, T, n_fft]
    """

    def __init__(self, channel_amp = 1, channel_phase=2):
        super(Model, self).__init__()
        self.stft = ConvSTFT(512, 256, 512, 'hanning', 'complex', True)
        self.istft = ConviSTFT(512, 256, 512, 'hanning', 'complex', True)
        self.amp_conv1 = nn.Sequential(
                                nn.Conv2d(2, channel_amp, 
                                        kernel_size=[7,1],
                                        padding=(3,0)
                                    ),
                                nn.BatchNorm2d(channel_amp),
                                nn.ReLU(),
                                nn.Conv2d(channel_amp, channel_amp, 
                                        kernel_size=[1,7],
                                        padding=(0,3)
                                    ),
                                nn.BatchNorm2d(channel_amp),
                                nn.ReLU(),
                        )
        self.phase_conv1 = nn.Sequential(
                                nn.Conv2d(2, channel_phase, 
                                        kernel_size=[3,5],
                                        padding=(1,2)
                                    ),
                                nn.Conv2d(channel_phase, channel_phase, 
                                        kernel_size=[3,25],
                                        padding=(1, 12)
                                    ),
                        )
        self.amp_conv2 = nn.Sequential(
                        nn.Conv2d(channel_amp, 1, kernel_size=[1, 1]),
                        nn.BatchNorm2d(1),
                        nn.ReLU(),
                    )
        self.phase_conv2 = nn.Sequential(
                        nn.Conv1d(3,8,kernel_size=[1,1])
                    )
        # Encoder
        self.conv_block_1 = CausalConvBlock(1, 16)
        self.conv_block_2 = CausalConvBlock(16, 32)
        self.conv_block_3 = CausalConvBlock(32, 64)
        self.conv_block_4 = CausalConvBlock(64, 128)
        self.conv_block_5 = CausalConvBlock(128, 256)

        # LSTM
        self.lstm_layer = nn.LSTM(input_size=251, hidden_size=251, num_layers=2, batch_first=True)
      
        self.tran_conv_block_1 = CausalTransConvBlock(256 + 256, 128)
        self.tran_conv_block_2 = CausalTransConvBlock(128 + 128, 64)
        self.tran_conv_block_3 = CausalTransConvBlock(64, 32)
        self.tran_conv_block_4 = CausalTransConvBlock(32, 16, output_padding=(1, 0))
        self.pha_tran_conv_block_5 = CausalTransConvBlock(16, 2, is_last=True)
        self.tran_conv_block_5 = CausalTransConvBlock(16, 1, is_last=True)

        self.phase_conv5 = nn.Sequential(
                        nn.Conv1d(8, 2, kernel_size=(1,1))
                    )
        self.phase_conv3 = nn.Sequential(
                        nn.Conv2d(8, 8, kernel_size=(5,5), padding=(2,2)),
                        GLayerNorm2d(8),
                    )
        self.phase_conv4 = nn.Sequential(
                        nn.Conv2d(8, 8, kernel_size=(1,25), padding=(0,12)),
                        GLayerNorm2d(8),
                    )

    def forward(self, x):
        self.lstm_layer.flatten_parameters()

        # [B, D*2, T]
        cmp_spec = self.stft(x)
        cmp_spec = torch.unsqueeze(cmp_spec, 1)

        # to [B, 2, D, T]
        cmp_spec = torch.cat([
                                cmp_spec[:,:,:257,:],
                                cmp_spec[:,:,257:,:],
                                ],
                                1)
        mean = torch.mean(cmp_spec, [1, 2, 3], keepdim = True)
        std = torch.std(cmp_spec, [1, 2, 3], keepdim = True)
        cmp_spec = (cmp_spec - mean) / (std + 1e-8)        

        # to [B, 1, D, T]
        amp_spec = torch.sqrt(
                            torch.abs(cmp_spec[:,0])**2+
                            torch.abs(cmp_spec[:,1])**2,
                        )
        amp_spec = torch.unsqueeze(amp_spec, 1)
        
        spec = self.amp_conv1(cmp_spec)
        # phase = self.phase_conv1(cmp_spec)

        #print(s_spec.size())
        el_1 = self.conv_block_1(spec)
        el_2 = self.conv_block_2(el_1)
        el_3 = self.conv_block_3(el_2)
        el_4 = self.conv_block_4(el_3)
        #el_5 = self.conv_block_5(el_4)  # [2, 256, 4, 200]

        er_1 = self.conv_block_1(spec)
        er_2 = self.conv_block_2(er_1)
        er_3 = self.conv_block_3(er_2)
        er_4 = self.conv_block_4(er_3)
        #er_5 = self.conv_block_5(er_4)
        e_5 = torch.cat([el_4, er_4], dim = 1)

        #print(e_5.size())

        batch_size, n_channels, n_f_bins, n_frame_size = e_5.shape

        # [2, 256, 4, 200] = [2, 1024, 200] => [2, 200, 1024]
        lstm_in = e_5.reshape(batch_size, n_channels * n_f_bins, n_frame_size)#.permute(0, 2, 1)
        lstm_out, _ = self.lstm_layer(lstm_in)  # [2, 200, 1024]
        lstm_out = lstm_out.reshape(batch_size, n_channels, n_f_bins, n_frame_size)  # [2, 256, 4, 200]

        #d_1 = self.tran_conv_block_1(lstm_out)
        d_2 = self.tran_conv_block_2(lstm_out)
        d_3 = self.tran_conv_block_3(d_2)
        d_4 = self.tran_conv_block_4(d_3)
        d_5 = self.tran_conv_block_5(d_4)


        phase_pro = self.phase_conv1(cmp_spec)			
        phase_input = torch.cat([phase_pro, self.amp_conv2(d_5)], dim = 1)
      
        phase_input = self.phase_conv2(phase_input)	
        p1 = self.phase_conv3(phase_input)
        p1 = self.phase_conv4(p1)
		
        p2 = self.phase_conv3(p1 + phase_input)
        p2 = self.phase_conv4(p2)
		
        p3 = self.phase_conv3(p2 + p1)
        p3 = self.phase_conv4(p3)
        # p1 = self.pha_conv_block1(phase_input)
        # p2 = self.conv_block_2(p1)
        # p3 = self.conv_block_3(p2)
        # p4 = self.conv_block_4(p3)
        # p5 = self.conv_block_5(p4)  # [2, 256, 4, 200]	

        # batch_size1, n_channels1, n_f_bins1, n_frame_size1 = p5.shape

        # # [2, 256, 4, 200] = [2, 1024, 200] => [2, 200, 1024]
        # lstm_in_phase = p5.reshape(batch_size1, n_channels1 * n_f_bins1, n_frame_size1).permute(0, 2, 1)
        # lstm_out_phase, _ = self.phase_lstm_layer(lstm_in_phase)  # [2, 200, 1024]		
        # lstm_out_phase = lstm_out_phase.permute(0, 2, 1).reshape(batch_size1, n_channels1, n_f_bins1, n_frame_size1)  # [2, 256, 4, 200]

        # p6 = self.pha_tran_conv_block_1(lstm_out_phase)
        # p7 = self.tran_conv_block_2(p6)
        # p8 = self.tran_conv_block_3(p7)
        # p9 = self.tran_conv_block_4(p8)
        # p10 = self.pha_tran_conv_block_5(p9)
        p5 = self.phase_conv5(p3)
        p5 = phase_pro + p5
        p5 = p5/(torch.sqrt(
                            torch.abs(p5[:,0])**2+
                            torch.abs(p5[:,1])**2)
                        +1e-8).unsqueeze(1)
        est_spec = amp_spec * d_5 * p5
        est_spec = torch.cat([est_spec[:,0], est_spec[:,1]], 1)
        est_wav = self.istft(est_spec, None)
        est_wav = torch.squeeze(est_wav, 1)
        return est_spec, est_wav, el_1, er_1, el_2, er_2, el_3, er_3, el_4, er_4
		
    def loss(self, est, labels, mode='Mix'):
        '''
        mode == 'Mix'
            est: [B, F*2, T]
            labels: [B, F*2,T]
        mode == 'SiSNR'
            est: [B, T]
            labels: [B, T]
        '''
        if mode == 'SiSNR':
            if labels.dim() == 3:
                labels = torch.squeeze(labels,1)
            if est.dim() == 3:
                est = torch.squeeze(est,1)
            return -si_snr(est, labels)         
        elif mode == 'Mix':
            b, d, t = est.size()
            gth_cspec = self.stft(labels)
            est_cspec = est  
            gth_mag_spec = torch.sqrt(
                                    gth_cspec[:, :self.feat_dim, :]**2
                                    +gth_cspec[:, self.feat_dim:, :]**2 + 1e-8
                               )
            est_mag_spec = torch.sqrt(
                                    est_cspec[:, :self.feat_dim, :]**2
                                    +est_cspec[:, self.feat_dim:, :]**2 + 1e-8
                                )
            
            # power compress 
            gth_cprs_mag_spec = gth_mag_spec**0.3
            est_cprs_mag_spec = est_mag_spec**0.3
            amp_loss = F.mse_loss(
                                gth_cprs_mag_spec, est_cprs_mag_spec
                            )*d
            compress_coff = (gth_cprs_mag_spec/(1e-8+gth_mag_spec)).repeat(1,2,1)
            phase_loss = F.mse_loss(
                                gth_cspec*compress_coff,
                                est_cspec*compress_coff
                            )*d
            
            all_loss = amp_loss*0.5 + phase_loss*0.5
            return all_loss, amp_loss, phase_loss

def remove_dc(data):
    mean = torch.mean(data, -1, keepdim=True) 
    data = data - mean
    return data
def l2_norm(s1, s2):
    #norm = torch.sqrt(torch.sum(s1*s2, 1, keepdim=True))
    #norm = torch.norm(s1*s2, 1, keepdim=True)
    
    norm = torch.sum(s1*s2, -1, keepdim=True)
    return norm 

def si_snr(s1, s2, eps=1e-8):
    #s1 = remove_dc(s1)
    #s2 = remove_dc(s2)
    s1_s2_norm = l2_norm(s1, s2)
    s2_s2_norm = l2_norm(s2, s2)
    s_target =  s1_s2_norm/(s2_s2_norm+eps)*s2
    e_nosie = s1 - s_target
    target_norm = l2_norm(s_target, s_target)
    noise_norm = l2_norm(e_nosie, e_nosie)
    snr = 10*torch.log10((target_norm)/(noise_norm+eps)+eps)
    return torch.mean(snr)

if __name__ == '__main__':
    layer = Model()
    a = torch.rand(2, 64000)
    c, d = layer(a)	
	
    print(c.shape)
    print(d.shape)
