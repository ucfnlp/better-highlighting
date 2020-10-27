function docs = testdpp(docs, theta, min_words, is_concat_feat, alpha)
% predict the Y's for docs
% assign Ypred to each doc of docs
for i = 1 : length(docs)
    docs(i).Ypred = predictY(docs(i), theta, min_words, is_concat_feat, alpha);
end

end


function Y = predictY(doc, theta, min_words, is_concat_feat, alpha)
b = 665; % the standard by DUC => change to 100 words

if is_concat_feat
    if size(doc.W_single_vec, 1) ~= size(doc.F, 1)
        X = cat(2, doc.W_single_vec', full(doc.F));
    else
        X = cat(2, doc.W_single_vec, full(doc.F));
    end
    S = doc.W_pair*(1-alpha) + doc.W_tfidf*alpha;    
else
    X = full(doc.F);
    S = doc.W_pair*(1-alpha) + doc.W_tfidf*alpha;
end

q = exp(X * theta);
L = S .* (q*q');        L = 0.5*(L+L');
[V, Lam] = eig(full(L));
Lam(Lam<0) = 0;
L = V*Lam*V';    % project L into the PSD cone

flag = true(size(doc.cost));
flag(sum(doc.txt>0,2) < min_words) = false;

% get rid of segments that are original sentences
directory = doc.dir;
name = doc.name;
base = [directory '/' name '/' name];
if exist([base '.fpos'], 'file')
    full_pos = load([base '.fpos']);
end
if exist('full_pos', 'var')
    flag_ = true(size(full_pos));
    flag_(logical(full_pos)) = false;
    flag = flag & flag_;
end

% greedy method
Y = [];
val_old = 0;
while any(flag)
    inds = find(flag);
    p = zeros(size(inds));
    for iter = 1 : length(inds)
        i = inds(iter);
        Ytmp = [i; Y];
        p(iter) = (det(L(Ytmp,Ytmp)) - val_old);
    end
    [val, pos] = max(p);
    
    Y = [inds(pos); Y];
    val_old = det(L(Y,Y));
    lenY = sum(doc.cost(Y));
    flag(Y) = false;
    
    if b-lenY < 0
        break;
    end
end

end
