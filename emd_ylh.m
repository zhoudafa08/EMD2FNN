function imf=emd_ylh(x,t,method,epsilon)
% calculate the EMD decomposition of a signal
% x: the original signal of size 1x N
% method='spline' or 'pchip' corresponding to cubic spline interpolation
%       or piecewise cubic Hermite interpolation   
% epsilon: the stop criteria, if sum(abs(mean))<epsilon*sum(abs(r)), the
% inner iteration ends. here mean stand for the mean of r.
% imf: is the IMFs of x, in which imf(k,:)is the k-th imf and the last row 
% of imf is the residue
% 2012,8  by Lihua Yang at SYSU


r=x;
n=1;  %n: the number of imfs obtained
r_max_size=size(locmax(r));  %% r_max_size: the number of maximum points of r   
r_min_size=size(locmin(r));  %% r_max_size: the number of minimum points of r
r_max_min_num=r_max_size(1)+r_min_size(1);   %% r_max_min_size: the number of extreme points of r 
k=0;
while r_max_min_num>=4 && k<13 %% k: the upper limit of the number of IMFs
    h_mean_norm=1; 
    h_mean=0;   
    h=r;
   %% Inner iteration 
    while h_mean_norm>epsilon %% h_mean_norm: the norm of the average of h
        h=h-h_mean;
        [envx, envm]=envlp_y(h',method); %% envx, envm the upper and lower envelopes of h
        h_mean=(envx'+envm')/2;   %% h_mean: the average of h
       %% compute h_mean_norm to determine whether the inner iteration ends.
        h_mean_norm=sum(abs(h_mean))/sum(abs(h)); 
    end
    imf(n,:)=h; %% the n-th IMF
    r=r-imf(n,:);                  
    %% compute r_max_min_num to determine whether the outer iteration ends.
    n=n+1;
    r_max_size=size(locmax(r));     
    r_min_size=size(locmin(r));   
    r_max_min_num=r_max_size(1)+r_min_size(1);       
    k=k+1;
end
imf(n,:)=r; %% the residue