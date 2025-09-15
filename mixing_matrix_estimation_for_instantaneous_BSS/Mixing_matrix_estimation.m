function [error_first_clustering, error_elimination, error_second_clustering, H_est_first_clustering, H_est_after_elimination, H_est_second_clustering]=Mixing_matrix_estimation(H,X,K,B,signal_size,overlap,win,G,Nsources,Mmics,deltaTheta)


X_sparse_points=[]; %to store single source points


if rem(K,2)==0 %because of the symmetry, only half of the sequence is returned.
    half_of_K=K/2+1;
else
    half_of_K=(K+1)/2;
end

%memory buffer
real_fft_array=zeros(Mmics,B,half_of_K);
imag_fft_array=zeros(Mmics,B,half_of_K);

b=0;
for n=1:K*overlap:signal_size-K
    b=b+1;
    for m=1:Mmics
        x=X(m,n:n+K-1).*win;
        temp=fft(x);
        real_fft_array(m,b,:)=real(temp(1:half_of_K));
        imag_fft_array(m,b,:)=imag(temp(1:half_of_K));
    end
end
B=b;

%arranging the frequency bins in the decenting order of the variances of
%the DFT coefficients
V=var(squeeze(real_fft_array(1,:,:)),0,1);
[V IDXk]=sort(V,'descend');

starting_bin=1;
for k=starting_bin:80%half_of_K-1 %need not use all the frequency bins for the single source point estimation. If the sources are highly sparse and if we use more frequency bins, matlab may show memory full error
%    k
    Rk=real_fft_array(:,:,IDXk(k));
    Ik=imag_fft_array(:,:,IDXk(k));
    one_minus_AbsCosTheta_between_R_and_I= 1 - abs(sum(Rk.*Ik,1)./(sqrt(sum(Rk.^2,1)).*sqrt(sum(Ik.^2,1))));
    
    sparse_points_idx=find(one_minus_AbsCosTheta_between_R_and_I<(1-cosd(deltaTheta)));

    if ~isempty(sparse_points_idx)
%                                         %eliminating outliers by removing
%                                         %samples which are not contineous
%                                         initial_idx=find(one_minus_AbsCosTheta_between_R_and_I<(1-cosd(deltaTheta)));
%                                         vector_to_shift=(diff(initial_idx)==1);
%                                         [c r]=size(vector_to_shift);
%                                         if c>1 & r== 1
%                                             vector_to_shift=vector_to_shift';
%                                         end
%                                         vector_to_shift=[vector_to_shift 0];
%                                         vector_to_AND = vector_to_shift | circshift(vector_to_shift, [0 1]);
%                                         sparse_points_idx_idx = find(vector_to_AND.*sparse_points_idx);
%                                         sparse_points_idx = sparse_points_idx(sparse_points_idx_idx);
        
        from_R=Rk(:,sparse_points_idx);%real part of the coefficients for clustering
        from_I=Ik(:,sparse_points_idx);%imag part of the coefficients for clustering
        %discarding small amplitude values
        magnitude= sqrt(sum(from_R.^2,1));
        high_mag_idx=find(magnitude>0.25);%<-----------------------------------------------------------------------------Threshold value
        from_R=from_R(:,high_mag_idx);
        
        magnitude= sqrt(sum(from_I.^2,1));
        high_mag_idx=find(magnitude>0.25);%<-----------------------------------------------------------------------------Threshold value        
        from_I=from_I(:,high_mag_idx);
                    
        X_sparse_points = [X_sparse_points from_R];% from_I]; %There is not much improvement in performance even if we use both real part and imaginary part.
    end
    
    %pause(0.1);
    bin_multiple=1; %just to determine at what bin interval the error is to be estimated, see the next if statement.
    [temp Lsparse]=size(X_sparse_points);
    if rem(k,bin_multiple)==0 && Lsparse>Nsources
        [H_est_first_clustering,H_est_after_elimination,H_est_second_clustering]=mixing_matrix_estimation_by_hierachical_clustering(X_sparse_points',Nsources,Mmics,G);                  
        [nmse_1,nmse_2,nmse_3]=plot_H_est_error(H,H_est_first_clustering,H_est_after_elimination,H_est_second_clustering,Nsources);
        
        error_first_clustering(k)=nmse_1; %error before removing the outliers
        error_elimination(k)=nmse_2;%error after removing the outliers
        error_second_clustering(k)=nmse_3; %error after removing the outliers and second clustering
        
        if ~rem(k,40)
            figure(6);
            plot(error_first_clustering,':g.','Markersize',6); hold on;
            plot(error_elimination,':b.','Markersize',6); hold on;
            plot(error_second_clustering,':ro','Markersize',6);hold off;
            xlabel('Total no. of frequency bins used');
            ylabel('NMSE (dB)');
        end
    end 
    if ~rem(k,40)   
        pause(0.1);
        
    end
    

end    

if Lsparse<=Nsources
    error_first_clustering=zeros(1,40);
    error_elimination=zeros(1,40);
    error_second_clustering=zeros(1,40);
end



