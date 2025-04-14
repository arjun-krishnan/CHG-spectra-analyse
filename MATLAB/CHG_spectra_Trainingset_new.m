% Constants
c0 = 2.9979e8;
sigmae = 0.0007;
Emod = 0.0060;
t0 = 45e-15;
bwd = sqrt(2)*0.441/t0;
sigf = bwd/2.355;
wl0 = 800e-9;
f0 = c0/wl0;
w0 = 2*pi*f0;
fi = c0/(wl0-100e-9);
fe = c0/(wl0+100e-9);
ff = linspace(fi,fe,401);
ww = 2*pi*ff;
Ampf = exp(-(ff-f0).^2/2/sigf^2);

% Phase parameters
phi2 = (-7000e-30:200e-30:7000e-30);
phi3 = (0e-45:-500e-45:-30000e-45);
N_noise = 10;
N_s = length(phi2)*length(phi3)*N_noise;

% Spectral limits
wl_lim_list = {[380, 420], [255, 275], [190, 210]};
f_lim_list = cellfun(@(wl) c0./(wl.*1e-9), wl_lim_list, 'UniformOutput', false);

% Containers
A_400 = {};
A_266 = {};
A_200 = {};
phase_data = [];

% Waitbar setup
h = waitbar(0,' Please wait ...');
index = 1;
tic;

for D3 = phi3
    for D2 = phi2
        Phi = D2*(ww-w0).^2 + D3*(ww-w0).^3;
        tt = -600e-15:0.005e-15:600e-15;
        pulse = zeros(1,length(tt));
        
        for i = 1:length(ff)
            pulse = pulse + Ampf(i)*sin(2*pi*ff(i)*tt + Phi(i));
        end
        pulse = pulse / max(pulse);
        
        zz0 = tt * c0;
        dE = normrnd(0,sigmae,1,length(zz0)) + pulse * Emod;
        r56_range = linspace(0,130e-6,131);
        
        spec_maps = cell(1, 3);  % For 400, 266, 200 nm

        for i = 1:length(r56_range)
            r56 = r56_range(i);
            zz = zz0 + dE * r56;
            dens = histcounts(zz,8001);
            spec = abs(fftshift(fft(dens))).^2;

            T = (max(tt)-min(tt))/length(dens);
            Fs = 1/T;
            L = length(dens);
            dF = Fs/L;
            freq = -Fs/2:dF:Fs/2-dF;

            % Get and interpolate spectrum for each wavelength band
            for j = 1:3
                f_lim = f_lim_list{j};
                f_idx_start = floor((f_lim(2) - min(freq))/dF);
                f_idx_end = floor((f_lim(1) - min(freq))/dF);
                spec_slice = spec(f_idx_start:f_idx_end);

                if i == 1
                    spec_maps{j} = zeros(length(r56_range), length(spec_slice));
                end
                spec_maps{j}(i,:) = spec_slice;
            end
        end

        % Interpolation + normalization + noise/augment for each map
        for j = 1:3
            F = griddedInterpolant(spec_maps{j});
            x = linspace(1, size(spec_maps{j}, 1), 64);
            y = linspace(1, size(spec_maps{j}, 2), 64);
            spectrum = F({x,y});
            spectrum = spectrum / max(spectrum, [], 'all');

            for n_i = 1:N_noise
                spectrum_noisy = imnoise(spectrum, "gaussian", 0, 0.002);
                tform = randomAffine2d(XTranslation=[-1 1], YTranslation=[-1 1]);
                outputView = affineOutputView(size(spectrum_noisy), tform);
                spectrum_aug = imwarp(spectrum_noisy, tform, OutputView=outputView);
                spectrum_aug = flip(flip(spectrum_aug).');  % Flip twice

                % Store in appropriate container
                switch j
                    case 1
                        A_400{end+1} = spectrum_aug;
                    case 2
                        A_266{end+1} = spectrum_aug;
                    case 3
                        A_200{end+1} = spectrum_aug;
                end

                % Save corresponding phase info
                phase_data(:, index) = [D2; D3];

                % Progress update
                percent = index/N_s/3*100;
                et = toc;
                eta = (et * 100 / percent) - et;
                msg = sprintf("%.2f %% finished... ETA: %.1f minutes", percent, eta/60);
                waitbar(index/N_s, h, msg);
                index = index + 1;
            end
        end
    end
end
close(h);

%%

filename = 'train_set_ph2_ph3_-30k-0k_45fspulse.h5';
filename = ['../TrainingData/', filename];

% Helper function to convert cell array to 3D array
cellTo3D = @(C) cat(3, C{:});

% Convert cell arrays
Spectra_400 = cellTo3D(A_400);
Spectra_266 = cellTo3D(A_266);
Spectra_200 = cellTo3D(A_200);

% Delete file if it already exists
if isfile(filename)
    delete(filename)
end

% Create datasets
h5create(filename, '/Spectra_400', size(Spectra_400), 'Datatype', 'double');
h5create(filename, '/Spectra_266', size(Spectra_266), 'Datatype', 'double');
h5create(filename, '/Spectra_200', size(Spectra_200), 'Datatype', 'double');

h5create(filename, '/GDD', size(phase_data(1,:)), 'Datatype', 'double');
h5create(filename, '/TOD', size(phase_data(2,:)), 'Datatype', 'double');

% Write data
h5write(filename, '/Spectra_400', Spectra_400);
h5write(filename, '/Spectra_266', Spectra_266);
h5write(filename, '/Spectra_200', Spectra_200);

h5write(filename, '/GDD', phase_data(1,:) * 1e30);  % fs^2
h5write(filename, '/TOD', phase_data(2,:) * 1e45);  % fs^3
