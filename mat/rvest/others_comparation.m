% rng("default");
% Signal Settings
fc = 24e9; % Carrier frequency (Hz)
Gtx = 50; % Tx antenna gain (dB)
Grx = 50; % Radar Rx antenna gain (dB)
Grx_ue = 30; % UE Rx antenna gain (dB)
NF = 2.9; % Noise figure (dB)
Tref = 290; % Reference temperature (K)
Rmax = 200; % Maximum range of interest
vrelmax = 60; % Maximum relative velocity
c = physconst('LightSpeed');
lambda = c / fc;

num_of_bits = 256;
num_of_chips = 255;
p_prbs = ifft(load("data/p_freq_255.mat").data);
% p_prbs = 2*randi([0,1], num_of_chips, 1) - 1;
P_prbs = fft(p_prbs);

T_d = 5.1e-6;
T_chip = T_d / (2 * num_of_chips);
T_prbs = T_d / 2;
T_D = T_d * num_of_bits;
T_frame = T_D / 2;
fs = 1 / T_chip;
B = fs;


rdr = phased.RangeDopplerResponse(...
    'RangeMethod', 'FFT', ...
    'SampleRate', fs, ...
    'SweepSlope', -B/T_prbs, ...
    'DopplerOutput', 'Speed', ...
    'OperatingFrequency', fc, ...
    'PRFSource', 'Property', ...
    'PRF', 1/T_d, ...
    'ReferenceRangeCentered', false);

% CFAR settings (unchanged)
num_refer = [10, 10]; % [Range, Doppler] refer cells
num_guard = [4, 4]; % [Range, Doppler] guard cells
pfa = 1e-4; % Probability of false alarm
cfar2D = phased.CFARDetector2D(...
    'GuardBandSize', num_guard, ...
    'TrainingBandSize', num_refer, ...
    'ProbabilityFalseAlarm', pfa);





num_of_trials = 990;
SNRs_Tx = [-20, -10, -5, 0, 5, 10, 20, 30, 40, 50];



% Preallocate arrays
num_methods = 5;
num_of_targets = 1;
Numerical_r_tars = zeros(length(time_steps), length(SNRs_Tx), num_of_trials, num_of_targets, num_methods);
Numerical_v_tars = zeros(length(time_steps), length(SNRs_Tx), num_of_trials, num_of_targets, num_methods);



Gtx_lin = 10^(Gtx / 10); % Tx gain (linear)
Grx_lin = 10^(Grx / 10); % Rx gain (linear)
NF_lin = 10^(NF / 10);
noise_power = physconst('Boltzmann') * Tref * NF_lin * B;


for idx_SNR = 1:length(SNRs_Tx)
    SNR_tx = SNRs_Tx(idx_SNR);
    Pt = noise_power * 10^(SNR_tx/10);

    for idx_time = 1:length(time_steps)
        time_step = time_steps(idx_time);


        root_dir = 'data/mats/freq/';
        root_dir_eval = "data/evals/";
        z_PRBS_waveforms_DnCNN_Y_no_afm = load(fullfile(root_dir_eval, 'DnCNN_Y/no_afm/data/', num2str(SNR_tx), '/data.mat')).complex_Z_PRBS_waveforms_pred; 
        z_PRBS_waveforms_PDNet_Y_no_afm = load(fullfile(root_dir_eval, 'PDNet_Y/no_afm/data/', num2str(SNR_tx), '/data.mat')).complex_Z_PRBS_waveforms_pred; 

        for idx_trial = 1:num_of_trials
            % Build file paths
            mat_file_Y_PRBS_waveform = fullfile(root_dir, "Y_PRBS_waveform",  num2str(SNR_tx), [num2str(idx_trial) '.mat']);
            
            % Load .mat files
            Y_PRBS_waveform = load(mat_file_Y_PRBS_waveform, 'data').data;
            Z_PRBS_waveform = Y_PRBS_waveform .* conj(P_prbs);          
            z_PRBS_waveform = ifft(Z_PRBS_waveform);


            % MF + CFAR
            [hat_r_tars, hat_v_tars] = est_rv_cfar_multar_z(z_PRBS_waveform, pfa, T_prbs, num_of_targets, fs, lambda);
            Numerical_r_tars(idx_time, idx_SNR, idx_trial, :, 1) = hat_r_tars;
            Numerical_v_tars(idx_time, idx_SNR, idx_trial, :, 1) = hat_v_tars;

            % MF + CLEAN
            [hat_r_tars, hat_v_tars] = est_rv_clean_multar_z(z_PRBS_waveform, pfa, T_prbs, num_of_targets, fs, lambda);
            Numerical_r_tars(idx_time, idx_SNR, idx_trial, :, 2) = hat_r_tars;
            Numerical_v_tars(idx_time, idx_SNR, idx_trial, :, 2) = hat_v_tars;

            % MF + MUSIC
            [hat_r_tars, hat_v_tars] = est_rv_music_multar_z(z_PRBS_waveform, pfa, T_prbs, num_of_targets, fs, lambda);
            Numerical_r_tars(idx_time, idx_SNR, idx_trial, :, 3) = hat_r_tars;
            Numerical_v_tars(idx_time, idx_SNR, idx_trial, :, 3) = hat_v_tars;
            
            % PDNet + CFAR
            z_PRBS_waveform_PDNet_Y_no_afm = squeeze(z_PRBS_waveforms_PDNet_Y_no_afm(idx_trial, :, :));
            [hat_r_tars, hat_v_tars] = est_rv_cfar_multar_z(z_PRBS_waveform_PDNet_Y_no_afm, pfa, T_prbs, num_of_targets, fs, lambda);
            Numerical_r_tars(idx_time, idx_SNR, idx_trial, :, 4) = hat_r_tars;
            Numerical_v_tars(idx_time, idx_SNR, idx_trial, :, 4) = hat_v_tars;

            % DnCNN + CFAR
            z_PRBS_waveform_DnCNN_Y_no_afm = squeeze(z_PRBS_waveforms_DnCNN_Y_no_afm(idx_trial, :, :));
            [hat_r_tars, hat_v_tars] = est_rv_cfar_multar_z(z_PRBS_waveform_DnCNN_Y_no_afm, pfa, T_prbs, num_of_targets, fs, lambda);
            Numerical_r_tars(idx_time, idx_SNR, idx_trial, :, 5) = hat_r_tars;
            Numerical_v_tars(idx_time, idx_SNR, idx_trial, :, 5) = hat_v_tars;


            % rdr = phased.RangeDopplerResponse(...
            %     'RangeMethod', 'FFT', ...
            %     'SampleRate', fs, ...
            %     'SweepSlope', -B/T_prbs, ...
            %     'DopplerOutput', 'Speed', ...
            %     'OperatingFrequency', fc, ...
            %     'PRFSource', 'Property', ...
            %     'PRF', 1/T_d, ...
            % 'ReferenceRangeCentered', false);
            % figure;
            % plotResponse(rdr, fft(z_PRBS_waveform_PDNet_Y_no_afm), 'Unit', 'db');
            % xlim([-vrelmax vrelmax]);
            % ylim([0 Rmax]);
            % hold on;
            % scatter(hat_v_tars, hat_r_tars, 100, 'o', 'MarkerEdgeColor', 'b', 'LineWidth', 1.5);
            % hold off;
            
        end
    end
end

% Save the results
save('Numerical_r_tars_other_methods.mat', 'Numerical_r_tars', '-v7.3');
save('Numerical_v_tars_other_methods.mat', 'Numerical_v_tars', '-v7.3');
