function [R] = gaas(g, k)
% GAAS Creates the gradient aligned adversarial "subspace"
%      described in [tra2017].
%
%  Note that the matrices R and S in the proof of lemma 1 in
%  [tra2017] can be directly implemented as Householder 
%  reflections [gvl96]:
%
%      H = I - \beta vv'
%
%  for suitable choices of vector v and scalar \beta.
%
%
%  Example usage:
%     g = rand(10,1);
%     k = 3;
%     R = gaas(g, k);
%
%  Example for large dimensions:
%     g = randn(299*299*3,1);
%     k = 3;
%     R = gaas(g, k);
%
%
%  REFERENCES:
%   [tra17] Tramer et al. "The Space of Transferable Adversarial
%           Examples," arXiv 2017.
%   [gvl96] Golub and Van Loan "Matrix Computations" 1996.

% mjp, november 2017

d = numel(g);
R = zeros(d,k);
z = zeros(d,1);  z(1:k) = 1/sqrt(k);  % z from proof of lemma 1 in [tra17]

[v_s, beta_s] = householder_vec(z);  % note: ||z||_2 = 1
[v_r, beta_r] = householder_vec(g);

% Applying a householder transformation H to a vector x
% does not require the explict construction of H since
%
%   H x = (I - \beta vv') x = x - \beta v (v' x)
%
% In particular, this avoids the deadly outer product vv'.
% See also 5.1.4 in Golub and Van Loan.
%
apply_householder_to_vector = @(v, beta, x)  x - beta * v * dot(v, x);


% To calculate the r_i we use:
%
%     r_i := Q' e_i
%          = R' S e_i
%          = R S e_i
%
% where R = R' from the symmetry of Householder matrices
% (follows from symmetry of I and vv').
%
for ii = 1:k
    e_i = zeros(d,1);  e_i(ii) = 1;
    sei = apply_householder_to_vector(v_s, beta_s, e_i);
    
    r_i = apply_householder_to_vector(v_r, beta_r, sei);
    R(:,ii) = r_i;
end

% check the solution for correctness
if 1
    % the r_i should be orthogonal
    RtR = R' * R;
    assert(norm(RtR-eye(k,k), 'fro') < 1e-9);
   
    % make sure Qg = ||g||_2 z
    %
    % Note: the transpose on R below is because I stored
    %       the r_i as columns in R.
    %
    err = R'*g - norm(g,2) * z(1:k);
    assert(norm(err,2) < 1e-6);
 
    % make sure <g,r_i> behaves as expected.
    for ii = 1:k
        gtr = dot(g, R(:,ii));
        err = dot(g, r_i) - norm(g,2) / sqrt(k);
        assert(abs(err) < 1e-5);
    end
    fprintf('[%s]: calculation looks ok!\n', mfilename);
end



function [v, beta] = householder_vec(x)
% Returns elements needed to construct householder reflection matrix
%
%     P = I_n - beta * v * v'
%
%  where Px = ||x||_2 e_1
% 
% See Algorithm 5.1.1 in Golub and Van Loan.
%
% Note that one does not need to explicitly construct P to
% implement the action of P on a matrix A.
%
n = length(x);
v = ones(n,1);  v(2:n) = x(2:n);

sigma = dot(x(2:n), x(2:n));
if sigma == 0
    beta = 0;
else
    mu = sqrt(x(1)^2 + sigma);
    if x(1) <= 0
        v(1) = x(1) - mu;
    else
        v(1) = -sigma / (x(1) + mu);
    end
    beta = 2 * v(1)^2 / (sigma + v(1)^2);
    v = v / v(1);
end
