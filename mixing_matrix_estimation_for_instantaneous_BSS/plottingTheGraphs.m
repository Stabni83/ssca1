%Using Samples of Real and Imaginary parts#################################
figure(1); %After first clustering and after elimination
clear;
load errorFirstClusteringArrayRealAndImagParts.mat;
load errorEliminationArrayRealAndImagParts.mat;
temp=errorFirstClusteringArray(:,:,1);
mean_error=mean(temp,2);
plot(mean_error,'-ko');hold on;
temp=errorFirstClusteringArray(:,:,2);
mean_error=mean(temp,2);
plot(mean_error,'-ks');hold on;
temp=errorFirstClusteringArray(:,:,3);
mean_error=mean(temp,2);
plot(mean_error,'-kv');hold on;
temp=errorFirstClusteringArray(:,:,4);
mean_error=mean(temp,2);
mean_error(1:4)=NaN;
plot(mean_error,'-k*');hold on;

temp=errorEliminationArray(:,:,1);
mean_error=mean(temp,2);
plot(mean_error,'-go');hold on;
temp=errorEliminationArray(:,:,2);
mean_error=mean(temp,2);
plot(mean_error,'-gs');hold on;
temp=errorEliminationArray(:,:,3);
mean_error=mean(temp,2);
plot(mean_error,'-gv');hold on;
temp=errorEliminationArray(:,:,4);
mean_error=mean(temp,2);
mean_error(1:4)=NaN;
plot(mean_error,'-g*');hold off;
legend('\Delta\theta = 0.8 (By clustering initial SSPs)','\Delta\theta = 0.2 (By clustering initial SSPs)','\Delta\theta = 0.1 (By clustering initial SSPs)','\Delta\theta = 0.05 (By clustering initial SSPs)','\Delta\theta = 0.8 (After elimination of outliers)','\Delta\theta = 0.2 (After elimination of outliers)','\Delta\theta = 0.1 (After elimination of outliers)','\Delta\theta = 0.05 (After elimination of outliers)');
xlabel('Total no. of frequency bins used');
ylabel('NMSE (dB)');



figure(2);% After elimination and after second clustering
clear;
load errorEliminationArrayRealAndImagParts.mat;
load errorSecondClusteringArrayRealAndImagParts.mat;
temp=errorEliminationArray(:,:,1);
mean_error=mean(temp,2);
plot(mean_error,'-go');hold on;
temp=errorEliminationArray(:,:,2);
mean_error=mean(temp,2);
plot(mean_error,'-gs');hold on;
temp=errorEliminationArray(:,:,3);
mean_error=mean(temp,2);
plot(mean_error,'-gv');hold on;
temp=errorEliminationArray(:,:,4);
mean_error=mean(temp,2);
plot(mean_error,'-g*');hold on;


temp=errorSecondClusteringArray(:,:,1);
mean_error=mean(temp,2);
plot(mean_error,'-ro');hold on;
temp=errorSecondClusteringArray(:,:,2);
mean_error=mean(temp,2);
plot(mean_error,'-rs');hold on;
temp=errorSecondClusteringArray(:,:,3);
mean_error=mean(temp,2);
plot(mean_error,'-rv');hold on;
temp=errorSecondClusteringArray(:,:,4);
mean_error=mean(temp,2);
plot(mean_error,'-r*');hold off;
legend('\Delta\theta = 0.8 (After elimination of outliers)','\Delta\theta = 0.2 (After elimination of outliers)','\Delta\theta = 0.1 (After elimination of outliers)','\Delta\theta = 0.05 (After elimination of outliers)','\Delta\theta = 0.8 (Clustering outlier-free data)','\Delta\theta = 0.2 (Clustering outlier-free data)','\Delta\theta = 0.1 (Clustering outlier-free data)','\Delta\theta = 0.05 (Clustering outlier-free data)');
xlabel('Total no. of frequency bins used');
ylabel('NMSE (dB)');

%Using only the data from real part########################################
clear;
load errorEliminationArrayRealAndImagParts.mat;
figure(3);
temp=errorEliminationArray(:,:,1);
mean_error=mean(temp,2);
plot(mean_error,'-go');hold on;
temp=errorEliminationArray(:,:,2);
mean_error=mean(temp,2);
plot(mean_error,'-gs');hold on;
temp=errorEliminationArray(:,:,3);
mean_error=mean(temp,2);
plot(mean_error,'-gv');hold on;
temp=errorEliminationArray(:,:,4);
mean_error=mean(temp,2);
plot(mean_error,'-g*');hold on;

clear;
load errorEliminationArrayRealPartOnly.mat;
temp=errorEliminationArray(:,:,1);
mean_error=mean(temp,2);
plot(mean_error,'-ro');hold on;
temp=errorEliminationArray(:,:,2);
mean_error=mean(temp,2);
plot(mean_error,'-rs');hold on;
temp=errorEliminationArray(:,:,3);
mean_error=mean(temp,2);
plot(mean_error,'-rv');hold on;
temp=errorEliminationArray(:,:,4);
mean_error=mean(temp,2);
plot(mean_error,'-r*');hold off;
legend('\Delta\theta = 0.8 (Using data from both real and imaginary parts)','\Delta\theta = 0.2 (Using data from both real and imaginary parts)','\Delta\theta = 0.1 (Using data from both real and imaginary parts)','\Delta\theta = 0.05 (Using data from both real and imaginary parts)','\Delta\theta = 0.8 (Using data from real part only)','\Delta\theta = 0.2 (Using data from real part only)','\Delta\theta = 0.1 (Using data from real part only)','\Delta\theta = 0.05 (Using data from real part only)');
xlabel('Total no. of frequency bins used');
ylabel('NMSE (dB)');
% %Comparison of outlier elimination methods (removing samples which are away
% %from the mean direction and by checking the continuety of the SSPs)
% figure(4);% After elimination and after second clustering
% clear;
% load errorEliminationArrayRealAndImagParts.mat;
% temp=errorEliminationArray(:,:,1);
% mean_error=mean(temp,2);
% plot(mean_error,'-go');hold on;
% temp=errorEliminationArray(:,:,2);
% mean_error=mean(temp,2);
% plot(mean_error,'-gs');hold on;
% temp=errorEliminationArray(:,:,3);
% mean_error=mean(temp,2);
% plot(mean_error,'-gv');hold on;
% temp=errorEliminationArray(:,:,4);
% mean_error=mean(temp,2);
% mean_error(1:4)=NaN;
% plot(mean_error,'-g*');hold on;
% 
% load errorFirstClusteringArrayCheckingAdjacencyOfSSPs.mat
% temp=errorFirstClusteringArray(:,:,1);
% mean_error=mean(temp,2);
% plot(mean_error,'-ko');hold on;
% temp=errorFirstClusteringArray(:,:,2);
% mean_error=mean(temp,2);
% plot(mean_error,'-ks');hold on;
% temp=errorFirstClusteringArray(:,:,3);
% mean_error=mean(temp,2);
% plot(mean_error,'-kv');hold on;
% temp=errorFirstClusteringArray(:,:,4);
% mean_error=mean(temp,2);
% mean_error(1:4)=NaN;
% plot(mean_error,'-k*');hold off;