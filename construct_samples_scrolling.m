function [Feats, Labels]=construct_samples_scrolling(x, h, k, o)
% obtain the features and labels by scrolling the window of time series
% x: the raw time series
% h: the minimum time series length for decomposing into IMFs
% k: window length
% o: output (label) size
% coder: Feng Zhou (fengzhou@gdufe.edu.cn)
% wrote in June 2019

Feats=[];
Labels=[];
Upper_limit_IMF = 10; %% the upper limit of the numbers of IMFs
for i=0:length(x)-h-k-o
    disp(['Total iterations:', num2str(length(x)-h-k-o)]) 
    disp(['Now is:', num2str(i)])
    data = x(1:h+i+k);
    max_data = max(data);
    data = data / max_data;
    IMF = emd_ylh(x, 1:length(x), 'spline', 0.01);
    IMF = IMF * max_data;
    features=IMF(:,end-k+1:end)';
    labels=x(h+i+k+1:h+i+k+o);
    if size(IMF, 1) < Upper_limit_IMF
        features = [features, zeros(k, 10-size(IMF, 1))];
    else if size(IMF, 1) > Upper_limit_IMF
        size(IMF, 1)
        error('Error! Please modify the size feats!')
        end
    end
    features = reshape(features, 1,Upper_limit_IMF * k);
    labels = reshape(labels, 1, o);
    Feats = [Feats; features];
    Labels = [Labels; labels];
end
