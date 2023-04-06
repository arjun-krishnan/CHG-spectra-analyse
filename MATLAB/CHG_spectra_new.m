
c0=2.9979e8;
sigmae = 0.0007;
Emod = 0.0060;
t0 = 45e-15;
bwd = sqrt(2)*0.441/t0;
sigf = bwd/2.355;
wl0 = 800e-9;
f0 = c0/wl0;
w0 = 2*pi*f0;
%fi = c0/(wl0-100e-9);

%fe = c0/(wl0+100e-9);
fi = f0+50e12;
fe = f0-50e12;
ff = linspace(fi,fe,401);
ww = 2*pi*ff;
Ampf = exp(-(ff-f0).^2/2/sigf^2);
figure()
plot(ff,Ampf);
hold on
D2 = 5000e-30;
D3 = -30000e-45;
Phi = 1/2*D2*(ww-w0).^2 + D3*(ww-w0).^3;
plot(ff,Phi/max(Phi));
hold off

tt= -600e-15:0.005e-15:600e-15;
pul = zeros(1,length(tt));
for i=1:length(ff)
    %pul= pul + Ampf(i)*real(exp(1i*(2*pi*ff(i)*tt+Phi(i))));
    pul= pul + Ampf(i)*sin(2*pi*ff(i)*tt+Phi(i));
end
pul= pul/max(pul);
figure();
plot(tt*1e15,pul)%.^2);

[pks,loc] = findpeaks(pul.^2);
fit_p = fit(tt(loc).'*1e15,pks.','gauss1');
fit_p.c1/sqrt(2)*2.355

zz0 = tt*c0;
dE = normrnd(0,sigmae,1,length(zz0));
dE = dE+pul*Emod;

r56_l = linspace(0,130e-6,131);
spec_l = [];
for i=1:length(r56_l)
    r56 = r56_l(i);
    zz = zz0+ dE*r56;
    %plot(zz,dE,'.r');
    dens = histcounts(zz,16001);
    spec = abs(fftshift(fft(dens))).^2;
    
    T = (max(tt)-min(tt))/length(dens);
    Fs = 1/T;
    L = length(dens);
    dF = Fs/L;
    freq = -Fs/2:dF:Fs/2-dF;
    f2i = floor((7.1379e14-min(freq))/dF);
    f2l = floor((7.892e14-min(freq))/dF);
    %plot(freq(f2i:f2l),spec(f2i:f2l));
    spec_l(i,:) = spec(f2i:f2l);
end

figure();
surf(spec_l);
shading interp;
