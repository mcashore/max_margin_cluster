function [a, b, beta, beta0, clusters] = svm_max_margin_cluster
%     load twomoons;
%     [n, ~] = size(x);
    %load Concentric_rings;
    %x = X;
    %[n, ~] = size(x);
    load TwoCircles
    x = X;
    [n,~] = size(x);
    
    
    % construct kernel matrix
    
%     K = zeros(n, n);
%     for ii = 1:n
%         for jj = 1:n
%             K(ii,jj) = kernel('rbf', x(ii, :), x(jj, :), 5, 0);
%         end
%     end
    K = x*x';
    
    % set the smallest p% of entries to 0
%     for ii = 1:n
%         thresh = quantile(K(ii,:), 0.94);
%         for jj = 1:n
%             if K(ii,jj) < thresh
%                 K(ii,jj) = 0;
%                 K(jj,ii) = 0;
%             end
%         end
%     end
    
    if rank(K) ~= 200
        disp('kernel not full rank');
    end
    %K = K / norm(K);
    
    % project K onto space orthogonal to e
    e = ones(n, 1);
    P = (1/n) * e*e';
    Kproj = P * K;
    %Kproj = (Kproj + Kproj') / 2;
    Korthog = K - Kproj;
    r = rank(Korthog)+1;
    [a_mat, b] = eigs(Korthog, r);
    
    for i = 1:r
        a = a_mat(:,i);

        % Build a beta vector and beta0
        beta = sum(bsxfun(@times,x,a));
        for j = 1:size(x,1);
            if a(j) ~= 0;
                y = sign(a(j));
                beta0 = 1/y - beta*x(j,:)';
            end
        end
        
        % Plot the data vs the generated plane
        plot_x = linspace(min(x(:,1)),max(x(:,1)));
        plot_y = -1 / beta(2) * (beta0 + plot_x * beta(1));
        figure;
        hold on
        scatter(X(:,1),X(:,2),'r')
        plot(plot_x,plot_y,'b')
        title(sprintf('principal component: %d', i));
        hold off
    end
    
end
