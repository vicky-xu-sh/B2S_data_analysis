%% Load saved data

load('../eeg_data_time_freq_z_power_labels.mat');

% set seed
rng(1, 'twister');  % Use this before TreeBagger, fitcecoc, etc.

%% Classification using all 23 components 

% z_power is [IC x Band x Time x Trials] = 23 x 5 x 1000 x 109
z_power_slice = z_power(:, :, 251:975, :);  % taking data from 0ms to 1450ms

[nICs, nBands, nTime, nTrials] = size(z_power_slice);
X = reshape(z_power_slice, [], nTrials);  % size: (83375 x 109)
X = X';  % transpose so each row is a trial: (109 x 83375)

y = labels(:);

%% Consonant/Vowel labels

vowel_classes = {'i', 'u'};
y_v = ismember(y, [2 4 6]);  % Returns logical array: true (1) for 2, 4, 6; false (0) otherwise
y_v = double(y_v);           % Convert to numeric 0/1

consonant_classes = {'g', 'm', 's'};
y_c = zeros(size(y));              % Initialize all to 0
y_c(ismember(y, [3 4])) = 1;
y_c(ismember(y, [5 6])) = 2;

%% SVM

cv = cvpartition(labels, 'KFold', 5, 'Stratify', true);
t = templateSVM('KernelFunction', 'linear', 'BoxConstraint', 1, 'Standardize', true);

% 6-class
% SVMModel = fitcecoc(X, y, 'Learners', t, 'CVPartition', cv, 'Coding', 'onevsall');
% classLoss = kfoldLoss(SVMModel);
% fprintf('SVM 5-fold accuracy: %.2f%%\n', (1 - classLoss) * 100);

% vowel
SVMModel_v = fitcsvm(X, y_v, ...
    'KernelFunction', 'linear', ...
    'BoxConstraint', 1, ...
    'Standardize', true, ...
    'CVPartition', cv);
classLoss = kfoldLoss(SVMModel_v);
fprintf('Vowel SVM 5-fold accuracy: %.2f%%\n', (1 - classLoss) * 100); 
% Get predicted labels for each held-out fold
predictedLabels = kfoldPredict(SVMModel_v);
compute_confMat(y_v, predictedLabels, vowel_classes);

% consonant
% SVMModel_c = fitcecoc(X, y_c, 'Learners', t, 'CVPartition', cv, 'Coding', 'onevsall');
% classLoss = kfoldLoss(SVMModel_c);
% fprintf('Consonant SVM 5-fold accuracy: %.2f%%\n', (1 - classLoss) * 100);
% % Get predicted labels for each held-out fold
% predictedLabels = kfoldPredict(SVMModel_c);
% compute_confMat(y_c, predictedLabels, consonant_classes);

% Results:
% 6-class SVM 5-fold accuracy: 38.53%
% Vowel SVM 5-fold accuracy: 71.56%
% Consonant SVM 5-fold accuracy: 51.38%

% Generate confusion matrix
function compute_confMat(labels, predictedLabels, classes)

    % Generate confusion matrix
    confMat = confusionmat(labels, predictedLabels);
    
    % Generate a confusion chart
    figure;
    confusionchart(confMat, classes);
    
    % Customize title and appearance
    title('SVM Confusion Matrix (5-fold CV)');
    xlabel('Predicted Class');
    ylabel('True Class');

end 

%% SVM (with PCA)

[coeff, score, ~, ~, explained] = pca(X);
cumExplained = cumsum(explained);

nComponents = find(cumExplained >= 95, 1);  % Find # of components for 95% variance
fprintf('Retained %d PCA components to capture 95%% variance.\n', nComponents);
nComponents = find(cumExplained >= 99, 1);  % Find # of components for 99% variance
fprintf('Retained %d PCA components to capture 99%% variance.\n', nComponents);
% nComponents = find(cumExplained >= 99.99, 1);  % Find # of components for 99.99% variance
% fprintf('Retained %d PCA components to capture 99.99%% variance.\n', nComponents);

% Retained 28 PCA components to capture 95% variance.
% Retained 58 PCA components to capture 99% variance.
% Retained 107 PCA components to capture 99.99% variance.

X_pca = score(:, 1:nComponents);

%% 5-fold SVM on PCA-reduced data

t = templateSVM('KernelFunction', 'linear', 'BoxConstraint', 1, 'Standardize', true);

SVMModel = fitcecoc(X_pca, y, 'Learners', t);

CVSVMModel = crossval(SVMModel, 'KFold', 5);
classLoss = kfoldLoss(CVSVMModel);
fprintf('SVM 5-fold accuracy: %.2f%%\n', (1 - classLoss) * 100);

% SVM 5-fold accuracy: 17.43%

%% Classification on brain IC data only

% z_power is [IC x Band x Time x Trials] = 23 x 5 x 1000 x 109
ICs = [2,3,6:10,14,21];
z_power_slice = z_power(ICs, :, 251:975, :);  

[nICs, nBands, nTime, nTrials] = size(z_power_slice);
X = reshape(z_power_slice, [], nTrials);  
X = X';  % transpose so each row is a trial

y = labels(:);

% Consonant/Vowel labels

vowel_classes = {'i', 'u'};
y_v = ismember(y, [2 4 6]);  % Returns logical array: true (1) for 2, 4, 6; false (0) otherwise
y_v = double(y_v);           % Convert to numeric 0/1

consonant_classes = {'g', 'm', 's'};
y_c = zeros(size(y));  % Initialize all to 0
y_c(ismember(y, [3 4])) = 1;
y_c(ismember(y, [5 6])) = 2;

%% SVM

cv = cvpartition(labels, 'KFold', 5, 'Stratify', true);

% linear kernel
t = templateSVM('KernelFunction', 'linear', 'BoxConstraint', 1, 'Standardize', true);

% RBF kernel
% t = templateSVM('KernelFunction', 'rbf', ...
%                 'BoxConstraint', 1, ...
%                 'KernelScale', 'auto', ...  
%                 'Standardize', true);

% 6-class
% SVMModel = fitcecoc(X, y, 'Learners', t, 'CVPartition', cv, 'Coding', 'onevsall');
% classLoss = kfoldLoss(SVMModel);
% fprintf('SVM 5-fold accuracy: %.2f%%\n', (1 - classLoss) * 100);

% vowel
% SVMModel_v = fitcsvm(X, y_v, ...
%     'KernelFunction', 'linear', ...
%     'BoxConstraint', 1, ...
%     'Standardize', true, ...
%     'CVPartition', cv);
% classLoss = kfoldLoss(SVMModel_v);
% fprintf('Vowel SVM 5-fold accuracy: %.2f%%\n', (1 - classLoss) * 100); 
% % Get predicted labels for each held-out fold
% predictedLabels = kfoldPredict(SVMModel_v);
% compute_confMat(y_v, predictedLabels, vowel_classes);

% consonant
SVMModel_c = fitcecoc(X, y_c, 'Learners', t, 'CVPartition', cv, 'Coding', 'onevsall');
classLoss = kfoldLoss(SVMModel_c);
fprintf('Consonant SVM 5-fold accuracy: %.2f%%\n', (1 - classLoss) * 100);
% Get predicted labels for each held-out fold
predictedLabels = kfoldPredict(SVMModel_c);
compute_confMat(y_c, predictedLabels, consonant_classes);

% Results:
% 6-class SVM 5-fold accuracy, linear kernel, C = 1: around 30%
% Vowel SVM 5-fold accuracy: 66.06%
% Consonant SVM 5-fold accuracy: 56.88%


%% PCA 

[coeff, score, ~, ~, explained] = pca(X);
cumExplained = cumsum(explained);

nComponents = find(cumExplained >= 95, 1);  % Find # of components for 95% variance
fprintf('Retained %d PCA components to capture 95%% variance.\n', nComponents);
nComponents = find(cumExplained >= 99, 1);  % Find # of components for 99% variance
fprintf('Retained %d PCA components to capture 99%% variance.\n', nComponents);

% Retained 80 PCA components to capture 95% variance.
% Retained 100 PCA components to capture 99% variance.

X_pca = score(:, 1:nComponents);

%% PCA space scatter plot

% Define unique classes and colors
colors = lines(length(classes));  % colormap with distinct colors

figure; hold on;
for i = 1:length(classes)
    idx = y == i;
    scatter3(X_pca(idx,1), X_pca(idx,2), X_pca(idx,3), 50, colors(i,:), 'filled');
end
xlabel('PC1'); ylabel('PC2'); zlabel('PC3');
title('3D PCA projection of EEG features');
legend(cellstr(classes), 'Location', 'best');
grid on; view(3);
hold off;