% ------------------------------------------------------------
% DSP Test Vector 
% ------------------------------------------------------------

clc
clear
close all

fprintf("Generating DSP test vectors...\n")

%% Sampling parameters
fs = 16000;

%%  unique folder to hold data
timestamp = datestr(now,'yyyymmdd_HHMMSS');
folder = strcat("test_vectors_",timestamp);
mkdir(folder)

fprintf("Saving test vectors in %s\n",folder)

%% Filter coefficients
lp = lowpass_biquad_coeffs(250,fs);
bp = bandpass_biquad_coeffs(1000,fs);
hp = highpass_biquad_coeffs(4000,fs);

%% MATLAB filter format
b_lp = [lp(1) lp(2) lp(3)];
a_lp = [1 lp(5) lp(6)];

b_bp = [bp(1) bp(2) bp(3)];
a_bp = [1 bp(5) bp(6)];

b_hp = [hp(1) hp(2) hp(3)];
a_hp = [1 hp(5) hp(6)];

%% ------------------------------------------------------------
%% TEST 1 : Impulse
%% ------------------------------------------------------------

impulse = zeros(256,1);
impulse(1) = 1;

save_test("impulse", impulse, lp, bp, hp, folder)

%% ------------------------------------------------------------
%% TEST 2 : Step
%% ------------------------------------------------------------

step = ones(256,1);

save_test("step", step, lp, bp, hp, folder)

%% ------------------------------------------------------------
%% TEST 3 : Multitone
%% ------------------------------------------------------------

t = linspace(0,0.02,fs*0.02)';

multitone = sin(2*pi*100*t) + ...
            sin(2*pi*500*t) + ...
            sin(2*pi*2000*t);

save_test("multitone", multitone, lp, bp, hp, folder)

%% ------------------------------------------------------------
%% TEST 4 : Single Sine
%% ------------------------------------------------------------

sine1000 = sin(2*pi*1000*t);

save_test("sine1000", sine1000, lp, bp, hp, folder)

%% ------------------------------------------------------------
%% TEST 5 : Random Noise
%% ------------------------------------------------------------

rng(0)
noise = randn(256,1);

save_test("random", noise, lp, bp, hp, folder)

%% ------------------------------------------------------------
%% TEST 6 : Sine Sweep
%% ------------------------------------------------------------

sweep = sine_sweep(fs,3);

save_test("sine_sweep", sweep, lp, bp, hp, folder)

fprintf("All test vectors generated successfully.\n")

%% ------------------------------------------------------------
%%  Frequency Response Visualization
%% ------------------------------------------------------------

figure
freqz(b_lp,a_lp,1024,fs)
title("Lowpass Frequency Response")

figure
freqz(b_bp,a_bp,1024,fs)
title("Bandpass Frequency Response")

figure
freqz(b_hp,a_hp,1024,fs)
title("Highpass Frequency Response")

%% ------------------------------------------------------------
%% Mathematical Data from the Filters
%% ------------------------------------------------------------

fvtool(b_lp,a_lp,b_bp,a_bp,b_hp,a_hp)

%% ------------------------------------------------------------
%% Spectrogram of Sine Sweep
%% ------------------------------------------------------------

figure
spectrogram(sweep,256,200,256,fs,'yaxis')
title("Input Sine Sweep Spectrogram")

%% Filter outputs of sweep
lp_out = biquad_filter(sweep,lp);
bp_out = biquad_filter(sweep,bp);
hp_out = biquad_filter(sweep,hp);

figure
spectrogram(lp_out,256,200,256,fs,'yaxis')
title("Lowpass Sweep Response")

figure
spectrogram(bp_out,256,200,256,fs,'yaxis')
title("Bandpass Sweep Response")

figure
spectrogram(hp_out,256,200,256,fs,'yaxis')
title("Highpass Sweep Response")

%% ------------------------------------------------------------
%% FUNCTIONS
%% ------------------------------------------------------------

function coeffs = lowpass_biquad_coeffs(fc,fs,Q)

if nargin<3
    Q = 0.707;
end

w0 = 2*pi*fc/fs;
cos_w0 = cos(w0);
sin_w0 = sin(w0);
alpha = sin_w0/(2*Q);

b0 = (1 - cos_w0)/2;
b1 = 1 - cos_w0;
b2 = (1 - cos_w0)/2;

a0 = 1 + alpha;
a1 = -2*cos_w0;
a2 = 1 - alpha;

coeffs = [b0/a0 b1/a0 b2/a0 1 a1/a0 a2/a0];
end

function coeffs = highpass_biquad_coeffs(fc,fs,Q)

if nargin<3
    Q = 0.707;
end

w0 = 2*pi*fc/fs;
cos_w0 = cos(w0);
sin_w0 = sin(w0);
alpha = sin_w0/(2*Q);

b0 = (1 + cos_w0)/2;
b1 = -(1 + cos_w0);
b2 = (1 + cos_w0)/2;

a0 = 1 + alpha;
a1 = -2*cos_w0;
a2 = 1 - alpha;

coeffs = [b0/a0 b1/a0 b2/a0 1 a1/a0 a2/a0];
end

function coeffs = bandpass_biquad_coeffs(fc,fs,Q)

if nargin<3
    Q = 1;
end

w0 = 2*pi*fc/fs;
cos_w0 = cos(w0);
sin_w0 = sin(w0);
alpha = sin_w0/(2*Q);

b0 = alpha;
b1 = 0;
b2 = -alpha;

a0 = 1 + alpha;
a1 = -2*cos_w0;
a2 = 1 - alpha;

coeffs = [b0/a0 b1/a0 b2/a0 1 a1/a0 a2/a0];
end

function y = biquad_filter(x,coeffs)

b0 = coeffs(1);
b1 = coeffs(2);
b2 = coeffs(3);
a1 = coeffs(5);
a2 = coeffs(6);

y = zeros(size(x));

x1 = 0; x2 = 0;
y1 = 0; y2 = 0;

for n = 1:length(x)

    y(n) = b0*x(n) + b1*x1 + b2*x2 - a1*y1 - a2*y2;

    x2 = x1;
    x1 = x(n);

    y2 = y1;
    y1 = y(n);

end
end

function save_test(name,signal,lp,bp,hp,folder)

lp_out = biquad_filter(signal,lp);
bp_out = biquad_filter(signal,bp);
hp_out = biquad_filter(signal,hp);

writematrix(signal, fullfile(folder,strcat("input_",name,".csv")))
writematrix(lp_out, fullfile(folder,strcat("output_lp_",name,".csv")))
writematrix(bp_out, fullfile(folder,strcat("output_bp_",name,".csv")))
writematrix(hp_out, fullfile(folder,strcat("output_hp_",name,".csv")))

fprintf("%s saved\n",name)

end

function sweep = sine_sweep(fs,duration)

t = linspace(0,duration,fs*duration)';

f_start = 20;
f_end = 8000;

k = (f_end - f_start)/duration;

sweep = sin(2*pi*(f_start*t + 0.5*k*t.^2));

end
