function [a, b, beta, beta0, clusters] = svm_max_margin_cluster
    load twomoons;
    [n, ~] = size(x);
    %load Concentric_rings;
    %x = X;
    %[n, ~] = size(x);
    
    % construct kernel matrix
    
    K = zeros(n, n);
    for ii = 1:n
        for jj = 1:n
            K(ii,jj) = kernel('rbf', x(ii, :), x(jj, :), 5, 0);
        end
    end
    
    % set the smallest p% of entries to 0
    for ii = 1:n
        thresh = quantile(K(ii,:), 0.94);
        for jj = 1:n
            if K(ii,jj) < thresh
                K(ii,jj) = 0;
                K(jj,ii) = 0;
            end
        end
    end
    
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
    r = rank(Korthog);
    [a, b] = eigs(Korthog, r);
    
    
    %b = b(i,i);
    %a = a(:,i);
    for i = 1:r
        figure;
        gscatter(x(:, 1), x(:, 2), sign(a(:,i)));
        title(sprintf('principal component: %d', i));
    end
    %{
    disp('asdf');
    % find row where a is nonzero
    beta0 = NaN;
    beta0list = ones(sum(beta0~=0), 1);
    listindex = 1;
    for ii = 1:n
        if a(ii) ~= 0
            yi = sign(a(ii));
            beta0list(listindex) = yi - a' * K(:,ii);
            listindex = listindex + 1;
            break;
        end
    end
    %beta0 = sum(beta0list) / (listindex - 1);
    beta0 = -median(a' * K);
    % cluster points!
    clusters = sign(a);
    %{
    clusters = ones(n, 1);
    for ii = 1:n
        support_vec_sum = a' * K(:,ii);
        clusters(ii) = sign(support_vec_sum(1,1)+beta0);
    end
    %}
                
            
    %{
    % solve for a
    cvx_begin
        variable a(n, 1);
        variable t(n, 1);
        variable b(n, 1);
        minimize(0.5 * trace(quad_form(a, K)));
        subject to
            sum(a) == 0;
            
            % force ||w||_1 <= 1
            sum(t) == 1;
            for ii = 1:n
                t(ii) >= 0;
                a(ii) >= -t(ii);
                a(ii) <= t(ii);
            end
            
            % force -||w||_1 <= -1
            sum(b) == -1;
            for ii = 1:n
                b(ii) <= 0;
                a(ii) <= -b(ii);
                a(ii) >= b(ii);
            end
    cvx_end
    %}
    gscatter(x(:,1), x(:, 2), clusters);
    %}
end