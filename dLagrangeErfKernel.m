function dL = dLagrangeErfKernel(a)
%DLAGRANGEERFKERNEL Computes the derivative of the Lagrangian with erf approx
%   
% Assume that there is a global Kernel matrix K defined to make it easier to
% deal with, and a global b.
%
% a contains the 'a' elements seen in the formulation of this problem, as
% well as a final element that contains the lambda value
%

global K
global b

dL = zeros(length(a),1);

for i = 1:(length(a)-1);
    non_ai_inds = setdiff(1:(length(a)-1),i);
    cross_sum = -1/2 * a(non_ai_inds)' * K(i,non_ai_inds)';
    dL(i) = a(i)*(2*b/sqrt(pi) * exp(-(b*a(i))^2) - K(i,i)) + ...
        erf(b*a(i)) + cross_sum - a(end);    
end

dL(end) = sum(a(1:(length(a)-1)));

end
