function fg = MLEDPP(theta, X, S, Y)

D = size(X,2);

q = exp(X * theta);
L = S .* (q*q');        L = 0.5*(L+L');
[V, Lam] = eig(full(L));
Lam(Lam<0) = 0;
L = V*Lam*V';    % just in case, project L into the PSD cone

lambda = double(diag(Lam));
Kii = sum(bsxfun(@times, (lambda./(lambda+1))', V.^2),2);

f = -log(det(L(Y,Y)) / prod(lambda+1));

% gradient
g = - sum(X(Y,:),1)' + sum(X.*repmat(Kii,1,D),1)';

% cat the output f & g
fg = full([f; g(:)]);