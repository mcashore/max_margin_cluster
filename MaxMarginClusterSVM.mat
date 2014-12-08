%% Attempt Maximum Margin Clustering, using the Dual SVM

clear all
close all
clc

% Import data
% dataset = 'twomoons';
% dataset = 'Concentric_rings';
% dataset = 'sonar';
% dataset = 'linear_new';
% dataset = 'noisy-linear_new';
% dataset = 'quadratic_new';
dataset = 'TwoCircles';
load([dataset '.mat'])

global X % Set data as global - set up as n obs x d features
global b % Set b as a global variable
global K % Make a global kernel matrix

% Set global variables
b = 10;
X = X;

% Set random seed
seed = 392;
rng(seed)

% Select initial guess
a0 = randn(size(X,1)+1,1);

% Kernel parameters
erf = 1;
kernelize = 0; % Decide whether or not to use a kernel
ker_type = 'erbf';
ker_param1 = 1;
ker_param2 = 0;

% If using a kernel, build it
if kernelize == 1;    
    K = zeros(size(X,1),size(X,1));
    for i = 1:size(X,1);
        for j = 1:size(X,1);
            K(i,j) = kernel(ker_type,X(i,:),X(j,:),ker_param1,ker_param2);
        end
    end    
end

% Run the solver
if kernelize == 1;
    a = fsolve(@dLagrangeErfKernel,a0);
else
    a = fsolve(@dLagrangeErf,a0);
end

% Check the Hessian of the Lagrangian for negative semi-definiteness
H = LagrangeHessian(a);
hessian_eigs = eig(H(1:end-1,1:end-1));

% Build a beta vector and beta0
beta = sum(bsxfun(@times,X,a(1:size(X,1))));
for i = 1:size(X,1);
    if a(i) ~= 0;
        y = sign(a(i));
        beta0 = 1/y - beta*X(i,:)';
    end
end

% Plot the data vs the generated plane
plot_x = linspace(min(X(:,1)),max(X(:,1)));
plot_y = -1 / beta(2) * (beta0 + plot_x * beta(1));
figure(1)
hold on
scatter(X(:,1),X(:,2),'r')
plot(plot_x,plot_y,'b')
hold off


% Build the classifier
classifier = (a(1:size(X,1)) > 0);

% Classify the results and plot
figure(2)
hold on
scatter(X(classifier,1),X(classifier,2),'r')
scatter(X(~classifier,1),X(~classifier,2),'b')
title(['Approximation, b=' num2str(b) ', kernel = ' num2str(kernelize) ...
    ', kernel type = ' ker_type ', erf = ' num2str(erf)])
hold off

% Save the plot
saveas(gcf,['ClusterResults/Approximation_data_' dataset '_erf' num2str(erf)...
    '_kernel' num2str(kernelize) '_kernelType ' ker_type '_kernelParam' ...
    num2str(ker_param1) '_b' num2str(b) '_seed' num2str(seed) '.jpg'])

% Save the solution for comparison with others
save(['a_results/' dataset '_' ker_type '_' num2str(ker_param1) '_' ...
    num2str(seed)],'a')
