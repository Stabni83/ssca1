%This program will estimate the single source points present in the
%instantaneous mixtures and using the estimated single source points the
%mixing matrix will be estimated. Then the error in mixing matrix
%estimation and the latest estimated mixing matrix will be returned.
%Here the hierarchical clustering algorithm is used to cluster the 
%estimated single source points. It is not necessary to use hierarchical
%clustering algorithm, instead any other suitable clustering algorithm can be used. 

%Reference:
%V. G. Reju, Soo Ngee Koh and Ing Yann Soon, "An algorithm for mixing matrix
%estimation in instantaneous blind source separation", Signal Processing, 
%Volume 89, Issue 9, September 2009, pp.1762-1773.

clear; close all;
Nsources=6;
Mmics=2; % 2 mixtures.
G=1; %enable the plote of graphs. 1-enable, 0-disable.
signal_size=16000*10; %seven sec.
K=1024 %DFT length
win=(window(@hann,K))';
overlap=128/1024%292/2048%0.1;
Number_of_speech_sets=100;


B=floor((signal_size+overlap*K-K)/(overlap*K));%number of blocks 
if rem(overlap*K,1)~=0
    error('change the value of overlap or K');
end

%reading clean signals
S(:,1)=wavread('all_sentences_train_DR1_MKLS0_M.wav',signal_size);
S(:,2)=wavread('all_sentences_train_DR2_FCYL0_F.wav',signal_size);
S(:,3)=wavread('all_sentences_train_DR2_MCEW0_M.wav',signal_size);
S(:,4)=wavread('all_sentences_train_DR5_FLMK0_F.wav',signal_size);
S(:,5)=wavread('all_sentences_train_DR6_FJDM2.wav',signal_size);
S(:,6)=wavread('all_sentences_train_DR7_MFXV0_M.wav',signal_size);


S(:,1)=0.5*S(:,1)/max(abs(S(:,1)));
S(:,2)=0.5*S(:,2)/max(abs(S(:,2)));
S(:,3)=0.5*S(:,3)/max(abs(S(:,3)));
S(:,4)=0.5*S(:,4)/max(abs(S(:,4)));
S(:,5)=0.5*S(:,5)/max(abs(S(:,5)));
S(:,6)=0.5*S(:,6)/max(abs(S(:,6)));



S=S';
%mixing matrix
T=-75;
D=30;
H=[cosd(T) cosd(T+D) cosd(T+2*D) cosd(T+3*D) cosd(T+4*D) cosd(T+5*D);% cosd(T+6*D) cosd(T+7*D) cosd(T+8*D) cosd(T+9*D) cosd(T+10*D) cosd(T+11*D) cosd(T+12*D) cosd(T+13*D) cosd(T+14*D) cosd(T+15*D);
   sind(T) sind(T+D) sind(T+2*D) sind(T+3*D) sind(T+4*D) sind(T+5*D)];% sind(T+6*D) sind(T+7*D) sind(T+8*D) sind(T+9*D) sind(T+10*D) sind(T+11*D) sind(T+12*D) sind(T+13*D) sind(T+14*D) sind(T+15*D)];


H
%generating mixture
X=H*S;
deltaTheta=0.2; %in degree

[error_first_clustering, error_elimination, error_second_clustering, H_est_first_clustering, H_est_after_elimination, H_est_second_clustering]=Mixing_matrix_estimation(H,X,K,B,signal_size,overlap,win,G,Nsources,Mmics,deltaTheta);







