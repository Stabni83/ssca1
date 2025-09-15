function [nmse_1,nmse_2,nmse_3]=plot_H_est_error(H,H_est_first_clustering,H_est_after_elimination,H_est_second_clustering,Nsources)

%normalise the column vectors
for n=1:Nsources
    H(:,n)=H(:,n)/norm(H(:,n));
    H_est_first_clustering(:,n)=H_est_first_clustering(:,n)/norm(H_est_first_clustering(:,n));
    H_est_after_elimination(:,n)=H_est_after_elimination(:,n)/norm(H_est_after_elimination(:,n));
    H_est_second_clustering(:,n)=H_est_second_clustering(:,n)/norm(H_est_second_clustering(:,n));
end
p=1:Nsources;
I=eye(Nsources);
E=perms(p);   

d_sum=zeros(length(E),1);
%align H_est_first_clustering with H
He=H_est_first_clustering;
if ~isnan(sum(sum(He)))
    for n=1:length(E)
        P=I(E(n,:),:);
        Hn=He*P';
        d = 1- abs(sum(Hn.*H,1)./(sqrt(sum(Hn.^2,1)).*sqrt(sum(H.^2,1))));
        d_sum(n)=sum(d);
    end
    [min_d idx]=min(d_sum);
    P=I(E(idx,:),:);
    Hn=He*P';
    d_sign = 1- sum(Hn.*H,1)./(sqrt(sum(Hn.^2,1)).*sqrt(sum(H.^2,1)));
    change_sign_idx=find(d_sign>1);
    Hn(:,change_sign_idx)=-1*Hn(:,change_sign_idx);
    nmse_1=10*log10(mse(H-Hn)/mse(H));   
else
    nmse_1=NaN;
end

%align H_est_after_elimination with H
He=H_est_after_elimination;
if ~isnan(sum(sum(He)))
    for n=1:length(E)
        P=I(E(n,:),:);
        Hn=He*P';
        d = 1- abs(sum(Hn.*H,1)./(sqrt(sum(Hn.^2,1)).*sqrt(sum(H.^2,1))));
        d_sum(n)=sum(d);
    end
    [min_d idx]=min(d_sum);
    P=I(E(idx,:),:);
    Hn=He*P';
    d_sign = 1- sum(Hn.*H,1)./(sqrt(sum(Hn.^2,1)).*sqrt(sum(H.^2,1)));
    change_sign_idx=find(d_sign>1);
    Hn(:,change_sign_idx)=-1*Hn(:,change_sign_idx);
    nmse_2=10*log10(mse(H-Hn)/mse(H));   
else
    nmse_2=NaN;
end    

%align H_est_second_clustering with H
He=H_est_second_clustering;
if ~isnan(sum(sum(He)))
    for n=1:length(E)
        P=I(E(n,:),:);
        Hn=He*P';
        d = 1- abs(sum(Hn.*H,1)./(sqrt(sum(Hn.^2,1)).*sqrt(sum(H.^2,1))));
        d_sum(n)=sum(d);
    end
    [min_d idx]=min(d_sum);
    P=I(E(idx,:),:);
    Hn=He*P';
    d_sign = 1- sum(Hn.*H,1)./(sqrt(sum(Hn.^2,1)).*sqrt(sum(H.^2,1)));
    change_sign_idx=find(d_sign>1);
    Hn(:,change_sign_idx)=-1*Hn(:,change_sign_idx);
    nmse_3=10*log10(mse(H-Hn)/mse(H));   
else
    nmse_3=NaN;
end
