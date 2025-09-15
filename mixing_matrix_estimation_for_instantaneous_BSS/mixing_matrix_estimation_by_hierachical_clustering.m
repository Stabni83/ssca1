function [H_est_first_clustering,H_est_after_elimination,H_est_second_clustering]=mixing_matrix_estimation_by_hierachical_clustering(X,Nsources,Mmics,G)

Nc=Nsources-1;
colour=['b','g','r','c','m','y','k'];
%mapping points to one side
Xrhs=X;
left_idx=find(X(:,1)<0);
Xrhs(left_idx,:)=-1*Xrhs(left_idx,:);

        
Y = pdist(X,'cosine'); 
Y=1-abs(1-Y);
Z = linkage(Y,'average');
%                                 dendrogram(Z);
lengthX=length(X(:,1));
sorted_T_sum_array(Nsources)=0;
sorted_T_sum_array(1)=1;
while sorted_T_sum_array(Nsources)<mean(sorted_T_sum_array(1:Nsources-1))*0.05 & Nc<10*Nsources & lengthX>=Nc %<sorted_T_sum_array(1)/20
    if Nc>Nsources & sum(sorted_T_sum_array(Nsources+1:end))<0.05*sum(sorted_T_sum_array(1:Nsources))
        break;
    end
           
    Nc=Nc+1;
    if Nc>2*Nsources
        display('Nc>2*Nsources');
    end
    T = cluster(Z,'maxclust',Nc);

    if G==1 & Mmics==2
        figure(1);
        scatter(X(:,1),X(:,2),3,T);title('Data input for first clustering');
    end

    %to plot the highest Nsources ( Nsources is the number of sources) clusters and to
    %calculate the corresponding directions
    T_sum_array=zeros(Nc,1);
    for t=1:Nc
        T_sum_array(t)=sum((T==t));
    end

    [sorted_T_sum_array sorted_idx]=sort(T_sum_array,'descend');
    if length(X)<Nc
        break;
    end    
end

%centroid calculation
centroid=zeros(Nsources,Mmics);
for n=1:Nsources
    if G==1 & Mmics==2
        figure(2);title('After first clustering, centroid also shown');
        Xcluster=X(find(T==sorted_idx(n)),:);
        scatter(Xcluster(:,1),Xcluster(:,2),3,n*ones(length(Xcluster(:,1)),1));hold on;
        centroid(n,:)=mean(Xrhs(find(T==sorted_idx(n)),:),1);
        scatter(centroid(n,1),centroid(n,2),3,(n+Nsources));
    end
    centroid(n,:)=mean(Xrhs(find(T==sorted_idx(n)),:),1);
end
%hold off;
H_est_first_clustering=centroid';
    
%---------------------------H estimation after Elimination of outliers------
Xnew=[];
scale_std=0.5;
centroid_eli=zeros(Nsources,Mmics);
for n=1:Nsources
    Xcluster=Xrhs(find(T==sorted_idx(n)),:);
    one_minus_CosTheta=1 - (Xcluster*centroid(n,:)')./(sqrt(sum(Xcluster.^2,2))*sqrt(centroid(n,:)*centroid(n,:)'));
    points_taken_idx = find(one_minus_CosTheta < scale_std*std(one_minus_CosTheta));
    if ~isempty(points_taken_idx)
        Xcluster = Xcluster(points_taken_idx,:);
    end
    centroid_eli(n,:) = mean(Xcluster,1);
    %new data from second clustering (next stage)
    Xcluster=X(find(T==sorted_idx(n)),:);
    Xcluster=Xcluster(points_taken_idx,:);
    Xnew = [Xnew; Xcluster];
    if G==1 & Mmics==2
        figure(3);title('After elimination, centroid also shown');
       scatter(Xcluster(:,1),Xcluster(:,2),3,n*ones(length(Xcluster(:,1)),1));hold on;
       scatter(centroid_eli(n,1),centroid_eli(n,2),3,(n+Nsources));
    end
end
%hold off;
H_est_after_elimination=centroid_eli';

%------------------------second stage of clustering using new data (after eliminating the outliers)---------
if length(Xnew(:,1))<Nsources
    H_est_second_clustering=NaN*ones(Mmics,Nsources);
    return;
end
Nc=Nsources-1;
X=Xnew;
clear Xnew;
lengthX=length(X(:,1));
if lengthX<Nc
    H_est_second_clustering=NaN*rand(Mmics,Nsources);
    return;
end

Xrhs=X;
left_idx=find(X(:,1)<0);
Xrhs(left_idx,:)=-1*Xrhs(left_idx,:);

        
Y = pdist(X,'cosine'); 
Y=1-abs(1-Y);
Z = linkage(Y,'average');

sorted_T_sum_array(Nsources)=0;
sorted_T_sum_array(1)=1;
while sorted_T_sum_array(Nsources)<mean(sorted_T_sum_array(1:Nsources-1))*0.05 & Nc<10*Nsources & lengthX>=Nc %<sorted_T_sum_array(1)/20
    if Nc>Nsources & sum(sorted_T_sum_array(Nsources+1:end))<0.05*sum(sorted_T_sum_array(1:Nsources))
        break;
    end    
    Nc=Nc+1;
    if Nc>2*Nsources
        display('Nc>2*Nsources');
    end    
    T = cluster(Z,'maxclust',Nc);

    if G==1 & Mmics==2
        figure(4);
        scatter(X(:,1),X(:,2),3,T);title('Input data for second clustering');
    end

    %to plot the highest Nsources ( Nsources is the number of sources) clusters and to
    %calculate the corresponding directions
    T_sum_array=zeros(Nc,1);
    for t=1:Nc
        T_sum_array(t)=sum((T==t));
    end

    [sorted_T_sum_array sorted_idx]=sort(T_sum_array,'descend');
    if length(X)<Nc
        break;
    end    
end

%centroid calculation
centroid_second_clustering=zeros(Nsources,Mmics);
for n=1:Nsources
    if G==1 & Mmics==2
        figure(5);title('After second clustering, centroid also is shown');
        Xcluster=X(find(T==sorted_idx(n)),:);
        scatter(Xcluster(:,1),Xcluster(:,2),3,n*ones(length(Xcluster(:,1)),1));hold on;
    end
    centroid_second_clustering(n,:)=mean(Xrhs(find(T==sorted_idx(n)),:),1);
end
%hold off;
H_est_second_clustering=centroid_second_clustering';