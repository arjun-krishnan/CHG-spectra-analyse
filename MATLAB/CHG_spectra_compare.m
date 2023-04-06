%Phi_pred = readNPY("Phi_pred_40fs_1.npy");

wd = fileparts(matlab.desktop.editor.getActiveFilename);
rootdir = '../';
Phi_pred = readtable([rootdir,'Predictions/phi_pred_with_noise_45fs_test_28_03_0-30k.csv']);
compressor = readNPY([rootdir,'cmprsr.npy']);

phi2 = Phi_pred.phi2;
phi3 = Phi_pred.phi3;

phi2_err = Phi_pred.phi2_error;
phi3_err = Phi_pred.phi3_error;

c0 = 2.9979e8;
wl = 800e-9;
d = 1e-3/1500;
theta = 51.3*pi/180;
omega = 2*pi*c0/wl;
cc= (2*pi*c0/omega/d);

D2_slope = 2*(4*pi^2*c0/omega^3/d^2)*(1-(cc-sin(theta))^2)^-1.5;
D3_slope = -(D2_slope/2/omega)*((1+(cc*sin(theta))-(sin(theta))^2)/(1-(cc-sin(theta))^2));

D2_slope = D2_slope*1e27;
[fit2,param] = createFit(compressor, phi2, D2_slope);
phi2_theory = D2_slope*compressor+fit2.c;
ci = confint(fit2);
phi2_l = D2_slope*compressor+ci(1);
phi2_u = D2_slope*compressor+ci(2);

D3_slope = D3_slope*1e42;
[fit3,param] = createFit(compressor, phi3, D3_slope);
phi3_theory = D3_slope*compressor+fit3.c;
ci = confint(fit3);
phi3_l = D3_slope*compressor+ci(1);
phi3_u = D3_slope*compressor+ci(2);

cd([rootdir,'CHG_spectra_observed']);
img_list = dir("*.npy");

imgs = [];
for i = 1: length(img_list)
    img= readNPY(img_list(i).name);
    img= flip(img,1);
    imgs(:,:,i) = img./max(img,[],'all');
end
cd(wd);

w = gausswin(10, 2.5);
w = w/sum(w);
img_smoothed = filter(w, 1, imgs(:,:,:));

cd(rootdir);

x = linspace(380,420,64);
y = linspace(0,130,64);
Sp_pred_list = [];
Sp_theo_list = [];

for i = 1:length(img_list)
    ph2 = phi2(i)*1e-30;
    ph3 = phi3(i)*1e-45;
    Sp_pred = flip(Calc_Spectra(ph2,ph3),2);
    Sp_pred_list(:,:,i) = Sp_pred;

    ph2_theo = phi2_theory(i)*1e-30;
    ph3_theo = phi3_theory(i)*1e-45;
    Sp_theo = flip(Calc_Spectra(ph2_theo,ph3_theo),2);
    Sp_theo_list(:,:,i) = Sp_theo;

    fig = figure();
    t = tiledlayout(3,1);
    set(gcf,'position',[10,10,500,1100])

    nexttile
    contourf(x-1,y,img_smoothed(:,:,i),100,'LineColor','None'); 
    %xlabel('\lambda (nm)',FontSize=15);
    %ylabel('R_{56} (\mum)',FontSize=15);
    ax=gca;
    ax.FontSize = 15;
    title("Observed");
    xlim([382,418]);
    grid

    nexttile
    contourf(x,y,Sp_pred,100,'LineColor','None'); 
    %xlabel('\lambda (nm)',FontSize=15);
    %ylabel('R_{56} (\mum)',FontSize=15);
    ax=gca;
    ax.FontSize = 15;
    title("Predicted");
    xlim([382,418]);
    grid

    nexttile
    contourf(x,y,Sp_theo,100,'LineColor','None'); 
    xlabel('\lambda (nm)',FontSize=15);
    ylabel('R_{56} (\mum)',FontSize=15);
    ax=gca;
    ax.FontSize = 15;
    title("Theoretical");
    xlim([382,418]);
    grid

    title(t,[num2str(compressor(i)+265-28.1),' mm']);
    saveas(fig,['imgs_400_spec\Comparison\',num2str(compressor(i)),'.png'])
end

cd(wd);

%%

compressor_fit = linspace(27.1,29.6,100);
phi2_t = D2_slope*compressor_fit+fit2.c;
ci = confint(fit2);
phi2_tl = D2_slope*compressor_fit+ci(1);
phi2_tu = D2_slope*compressor_fit+ci(2);

fig2 = figure();
t = tiledlayout(2,1);
set(gcf,'position',[10,10,500,750]);

nexttile
errorbar(compressor+265-28.1,phi2,phi2_err,'o','MarkerFaceColor','b');
hold on
plot(compressor_fit+265-28.1,phi2_t,'r');
plot(compressor_fit+265-28.1,phi2_tu,':r');
plot(compressor_fit+265-28.1,phi2_tl,':r');
%xlabel('Grating Separation (mm)',fontsize=15);
ax = gca;
ax.XAxis.FontSize = 12;
ax.YAxis.FontSize = 12;
ylabel('GDD (fs^2)',fontsize=15);
grid()
legend('Predictions','Fit','95% C.I.');
ylim([-6000 6000]);

phi3_t = D3_slope*compressor_fit+fit3.c;
ci = confint(fit3);
phi3_tl = D3_slope*compressor_fit+ci(1);
phi3_tu = D3_slope*compressor_fit+ci(2);

nexttile
errorbar(compressor+265-28.1,phi3,phi3_err,'o','MarkerFaceColor','b');
hold on
plot(compressor_fit+265-28.1,phi3_t,'r');
plot(compressor_fit+265-28.1,phi3_tu,':r');
plot(compressor_fit+265-28.1,phi3_tl,':r');
xlabel('Grating Separation (mm)',fontsize=15);
ax = gca;
ax.XAxis.FontSize = 12;
ax.YAxis.FontSize = 12;
ylabel('TOD (fs^3)',fontsize=15);
grid()

t.Padding = 'compact';
t.TileSpacing = 'compact';

%%

phi2_u_pred = phi2 + phi2_err;
phi2_l_pred = phi2 - phi2_err;
phi3_u_pred = phi3 + phi3_err;
phi3_l_pred = phi3 - phi3_err;

Sp_pred_list_u = [];
Sp_pred_list_l = [];
Sp_theo_list_u = [];
Sp_theo_list_l = [];

for i = 1:length(img_list)
    ph2 = phi2_u_pred(i)*1e-30;
    ph3 = phi3_u_pred(i)*1e-45;
    Sp_pred = flip(Calc_Spectra(ph2,ph3),2);
    Sp_pred_list_u(:,:,i) = Sp_pred;
    
    ph2 = phi2_l_pred(i)*1e-30;
    ph3 = phi3_l_pred(i)*1e-45;
    Sp_pred = flip(Calc_Spectra(ph2,ph3),2);
    Sp_pred_list_l(:,:,i) = Sp_pred;

    ph2 = phi2_u(i)*1e-30;
    ph3 = phi3_u(i)*1e-45;
    Sp_pred = flip(Calc_Spectra(ph2,ph3),2);
    Sp_theo_list_u(:,:,i) = Sp_pred;
    
    ph2 = phi2_l(i)*1e-30;
    ph3 = phi3_l(i)*1e-45;
    Sp_pred = flip(Calc_Spectra(ph2,ph3),2);
    Sp_theo_list_l(:,:,i) = Sp_pred;    
end

%% fitting a smoothing spline and finding the centroid of the central peak

mode = 1 ;  % 1 - Central peak   2 - Secondary peak

v_index = 12;
if mode == 2
    v_index = 50;
end
fx = c0./x*1e9;
ff = [380e-9,420e-9];
x_interp = linspace(c0/ff(1),c0/ff(2),4001);
centroid = [];
centroid_err = [];
for i = 1:11
    SSp1 = img_smoothed(v_index,:,i);
    SSp2 = Sp_pred_list(v_index,:,i);
    SSp2_u = Sp_pred_list_u(v_index,:,i);
    SSp2_l = Sp_pred_list_l(v_index,:,i);
    SSp3 = Sp_theo_list(v_index,:,i);
    SSp3_u = Sp_theo_list_u(v_index,:,i);
    SSp3_l = Sp_theo_list_l(v_index,:,i);

    f = fit(fx.',SSp1.','smoothingspline');
    if mode == 2
        if i<=4
            x_interp = linspace(c0/ff(1),c0/398e-9,4001);
        else
            x_interp = linspace(c0/405.5e-9,c0/ff(2),4001);
        end
    end
    [m,index] = max(f(x_interp));
    centroid(i,1) = x_interp(index);
    
    f = fit(fx.',SSp2.','smoothingspline');
    [m,index] = max(f(x_interp));
    centroid(i,2) = x_interp(index);
    
    f = fit(fx.',SSp2_u.','smoothingspline');
    [m,index] = max(f(x_interp));
    centroid_err(i,1) = abs(c0*1e9./x_interp(index)-c0*1e9./centroid(i,2));
    
    f = fit(fx.',SSp2_l.','smoothingspline');
    [m,index] = max(f(x_interp));
    centroid_err(i,2) = abs(c0*1e9./x_interp(index)-c0*1e9./centroid(i,2));

    f = fit(fx.',SSp3.','smoothingspline');
    [m,index] = max(f(x_interp));
    centroid(i,3) = x_interp(index);

    f = fit(fx.',SSp3_u.','smoothingspline');
    [m,index] = max(f(x_interp));
    centroid_err(i,3) = abs(c0*1e9./x_interp(index)-c0*1e9./centroid(i,3));
    
    f = fit(fx.',SSp3_l.','smoothingspline');
    [m,index] = max(f(x_interp));
    centroid_err(i,4) = abs(c0*1e9./x_interp(index)-c0*1e9./centroid(i,3));
end

%figure();
plot(compressor+265-28.1,c0*1e9./centroid(:,1)-1,'o--r','MarkerFaceColor','r');
hold on
errorbar(compressor+265-28.1,c0*1e9./centroid(:,2),centroid_err(:,2),centroid_err(:,1),'square-.b','MarkerFaceColor','b');
errorbar(compressor+265-28.1,c0*1e9./centroid(:,3),centroid_err(:,4),centroid_err(:,3),'diamond:','Color','#0dbf2e','MarkerFaceColor','#0dbf2e');
hold off
ax = gca;
ax.XAxis.FontSize = 12;
ax.YAxis.FontSize = 12;
legend('Observed','Predicted','Theoretical');
grid();
ylabel('Central peak position (nm)',fontsize=15);
if mode == 2
    ylabel('Secondary peak position (nm)',fontsize=15);
end
xlabel('Grating separation (mm)',fontsize=15);

%% Calculating the Asymmetry

center = c0*1e9./centroid(:,:); 
width = 15;
asymm = [];
asymm_err = [];
for i = 1:11
    [minValue,minIndex] = min(abs(x-center(i,1)));
    temp = img_smoothed(:,minIndex-width:minIndex+width,i);
    temp_i = flip(temp,2);
    asymm(i,1) = sqrt(immse(temp,temp_i));
    
    [minValue,minIndex] = min(abs(x-center(i,2)));
    temp = Sp_pred_list(:,minIndex-width:minIndex+width,i);
    temp_i = flip(temp,2);
    asymm(i,2) = sqrt(immse(temp,temp_i));

    [minValue,minIndex] = min(abs(x-(center(i,2)+centroid_err(i,1))));
    temp = Sp_pred_list_u(:,minIndex-width:minIndex+width,i);
    temp_i = flip(temp,2);
    asymm_err(i,1) = abs(asymm(i,2)-sqrt(immse(temp,temp_i)));

    [minValue,minIndex] = min(abs(x-(center(i,2)+centroid_err(i,2))));
    temp = Sp_pred_list_l(:,minIndex-width:minIndex+width,i);
    temp_i = flip(temp,2);
    asymm_err(i,2) = abs(asymm(i,2)-sqrt(immse(temp,temp_i)));

    [minValue,minIndex] = min(abs(x-center(i,3)));
    temp = Sp_theo_list(:,minIndex-width:minIndex+width,i);
    temp_i = flip(temp,2);
    asymm(i,3) = sqrt(immse(temp,temp_i));

    [minValue,minIndex] = min(abs(x-(center(i,3)+centroid_err(i,3))));
    temp = Sp_theo_list_u(:,minIndex-width:minIndex+width,i);
    temp_i = flip(temp,2);
    asymm_err(i,3) = abs(asymm(i,3)-sqrt(immse(temp,temp_i)));

    [minValue,minIndex] = min(abs(x-(center(i,3)+centroid_err(i,4))));
    temp = Sp_theo_list_l(:,minIndex-width:minIndex+width,i);
    temp_i = flip(temp,2);
    asymm_err(i,4) = abs(asymm(i,3)-sqrt(immse(temp,temp_i)));
end

plot(compressor+265-28.1,asymm(:,1),'o--r','MarkerFaceColor','r');
hold on
errorbar(compressor+265-28.1,asymm(:,2),asymm_err(:,2),asymm_err(:,1),'square-.b','MarkerFaceColor','b');
errorbar(compressor+265-28.1,asymm(:,3),asymm_err(:,4),asymm_err(:,3),'diamond:','Color','#0dbf2e','MarkerFaceColor','#0dbf2e');
legend('Observed','Predicted','Theoretical');
ax = gca;
ax.XAxis.FontSize = 12;
ax.YAxis.FontSize = 12;
grid
xlabel('Grating separation (mm)',FontSize=15);
ylabel('Asymmetry (a.u.)',FontSize=15);

%% fitting the predictions to theory


function [fitresult, gof] = createFit(compressor, phi2, slope)
[xData, yData] = prepareCurveData( compressor, phi2 );

% Set up fittype and options.
ft = fittype( [num2str(slope),'*x+c'], 'independent', 'x', 'dependent', 'y' );
opts = fitoptions( 'Method', 'NonlinearLeastSquares' );
opts.Display = 'Off';
opts.StartPoint = 0.125858274154704;

% Fit model to data.
[fitresult, gof] = fit( xData, yData, ft, opts );

% Plot fit with data.
figure( 'Name', 'untitled fit 1' );
h = plot( fitresult, xData, yData);
legend( h, 'phi2 vs. compressor', 'untitled fit 1', 'Location', 'NorthEast', 'Interpreter', 'none' );
% Label axes
xlabel( 'compressor', 'Interpreter', 'none' );
ylabel( 'phi2', 'Interpreter', 'none' );
grid on
end

%%

function Sp_predicted = Calc_Spectra(D2,D3)
    c0=2.9979e8;
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
    %D2 = 2500e-30;
    %D3 = -27000e-45;
    Phi = 1/2*D2*(ww-w0).^2 + D3*(ww-w0).^3;
    
    tt= -600e-15:0.005e-15:600e-15;
    pul = zeros(1,length(tt));
    for i=1:length(ff)
        pul= pul + Ampf(i)*sin(2*pi*ff(i)*tt+Phi(i));
    end
    pul= pul/max(pul);
    
    zz0 = tt*c0;
    dE = normrnd(0,sigmae,1,length(zz0));
    dE = dE+pul*Emod;
    
    r56_l = linspace(0,130e-6,131);
    spec_l = [];
    for i=1:length(r56_l)
        r56 = r56_l(i);
        zz = zz0+ dE*r56;

        dens = histcounts(zz,8001);
        spec = abs(fftshift(fft(dens))).^2;
        
        T = (max(tt)-min(tt))/length(dens);
        Fs = 1/T;
        L = length(dens);
        dF = Fs/L;
        freq = -Fs/2:dF:Fs/2-dF;
        f2i = floor((7.1379e14-min(freq))/dF);
        f2l = floor((7.8892e14-min(freq))/dF);

        spec_l(i,:) = spec(f2i:f2l);
    end
    Sp0 = spec_l/max(spec_l,[],'all');
    F=griddedInterpolant(Sp0);
    x= linspace(1,size(Sp0,1),64);
    y= linspace(1,size(Sp0,2),64);
    Sp_predicted = F({x,y});
end
