
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

phi2 = (-7000e-30:200e-30:7000e-30);
phi3 = (0e-45:-500e-45:-30000e-45);
N_noise = 10;
N_s = length(phi2)*length(phi3)*N_noise;

A= [];
phase= [];
h= waitbar(0,' Please wait ...');
index=1;
tic;
for D3 = phi3
    for D2 = phi2
        Phi = 1/2*D2*(ww-w0).^2 + D3*(ww-w0).^3;
        tt= -600e-15:0.005e-15:600e-15;
        pul = zeros(1,length(tt));
        for i=1:length(ff)
            pul= pul + Ampf(i)*sin(2*pi*ff(i)*tt+Phi(i));
        end
        pul= pul/max(pul);
        
%         [pks,loc] = findpeaks(pul.^2);
%         fit_p = fit(tt(loc).'*1e15,pks.','gauss1');
%         fit_p.c1/sqrt(2)*2.355;
        
        zz0 = tt*c0;
        dE = normrnd(0,sigmae,1,length(zz0));
        dE = dE+pul*Emod;
        
        r56_l = linspace(0,130e-6,131);
        spec_l = [];
        for i=1:length(r56_l)
            r56 = r56_l(i);
            zz = zz0+ dE*r56;
            %plot(zz,dE,'.r');
            dens = histcounts(zz,8001);
            spec = abs(fftshift(fft(dens))).^2;
            
            T = (max(tt)-min(tt))/length(dens);
            Fs = 1/T;
            L = length(dens);
            dF = Fs/L;
            freq = -Fs/2:dF:Fs/2-dF;
            f2i = floor((7.1379e14-min(freq))/dF);
            f2l = floor((7.8892e14-min(freq))/dF);
            %plot(freq(f2i:f2l),spec(f2i:f2l));
            spec_l(i,:) = spec(f2i:f2l);
            %imshow(spec_l);
        end
        %imshow(spec_l);
        F = griddedInterpolant(spec_l);
        x = linspace(1,size(spec_l,1),64);
        y = linspace(1,size(spec_l,2),64);
        sp2_new = F({x,y});
        max_sp  = max(sp2_new,[],'all');
        sp2_new = sp2_new/max_sp;
        for n_i = 1:N_noise
            sp2_new = imnoise(sp2_new,"gaussian",0,0.002);
            tform = randomAffine2d(XTranslation=[-1 1],YTranslation=[-1 1]);
            outputView = affineOutputView(size(sp2_new),tform);
            sp2_new = imwarp(sp2_new,tform,OutputView=outputView);
            %imshow(sp2_new);
            sp2_new = flip(flip(sp2_new).');
            A(:,:,index) = sp2_new;
            phase(:,index) = [D2,D3];
            percent= index/N_s*100;
            et = toc;
            eta = (et * 100/percent)- et;
            msg= string(percent)+" % finished...    ETA: "+string(round(eta/60,1))+" minutes";
            waitbar(index/N_s,h,msg);
            index=index+1;
        end

    end
end
close(h);
%%

filename = 'train_set_ph2_ph3_-30k-0k_45fspulse.h5';
filename = ['../../TrainingData/',filename];

if isfile(filename)
    delete (filename)
end

h5create(filename,'/Spectra',size(A));
h5create(filename,'/GDD',size(phase(1,:)));
h5create(filename,'/TOD',size(phase(2,:)));

h5write(filename,'/Spectra',A);
h5write(filename,'/GDD',phase(1,:)*1e30);
h5write(filename,'/TOD',phase(2,:)*1e45);

