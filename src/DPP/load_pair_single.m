function docs = load_pair_single(base_dir, docs, is_sigmoid, isSegOn, th)
%% load CNN similarity scores

% load pair & single prediction results
w_file = fullfile(base_dir, 'pair.mat');
if exist(w_file, 'file')
    W_pairs = load(w_file);
    W_pairs = W_pairs.pair;
end

w_file = fullfile(base_dir, 'single.mat');
if exist(w_file, 'file')
    W_singles = load(w_file);
    W_singles = W_singles.single;
end
sz_vec = numel(W_singles);

w_file = fullfile(base_dir, 'imp_vector.h5');
if exist(w_file, 'file')
    W_singles_vector = cell(sz_vec,1);
    for i=1:sz_vec
        dn = sprintf('/%d', i-1);
        vec = h5read(w_file, dn);
        W_singles_vector{i} = vec'; % Nx768
    end
else
    w_file = fullfile(base_dir, 'single_vector.mat');
    if exist(w_file, 'file')
        W_singles_vector = load(w_file);
        W_singles_vector = W_singles_vector.single_vector;
    end
end


w_file = fullfile(base_dir, 'name.mat');
if exist(w_file, 'file')
    doc_names_pred = load(w_file);
    doc_names_pred = cellstr(doc_names_pred.name);
end

keySet = doc_names_pred;
valueSet = 1:size(doc_names_pred, 1);
name2id = containers.Map(keySet, valueSet);

assert (size(doc_names_pred, 1) == length(docs));

for i = 1:length(docs)
    doc_name = docs(i).name;
    num_doc = docs(i).N;
    
    p = name2id(doc_name);   
    tfidf = docs(i).tfidf * docs(i).tfidf';
    
    W_p = W_pairs{p};
    if is_sigmoid % || 1
        W_p = sigmoid(W_p);
        for pi = 1:num_doc
            W_p(pi, pi) = 1.0;
        end
    end
    
    % update similarity scores
    if isSegOn
        % load seg
        base = fullfile(docs(i).dir, doc_name, doc_name);
        seg = load([base '.seg']);
        
        tfidf = triu(tfidf,1);
        W_p = triu(W_p,1);
        % set 1 if segments are from same sentence
        for si=1:numel(seg)
            for sj=si+1:numel(seg)
                if seg(si) == seg(sj)
                    if tfidf(si, sj) >= th % 0.1 THRESHOLD!
                        tfidf(si, sj) = 1.0;
                        W_p(si, sj) = 1.0;
                    end
                else
                    break;
                end
            end
        end
        tfidf = tfidf + tfidf' + eye(size(tfidf));
        W_p = W_p + W_p' + eye(size(W_p));
    end    
    
    docs(i).W_tfidf = tfidf;
    
    if exist('W_singles', 'var')
        W_s = W_singles{p}';
        if is_sigmoid
            W_s = sigmoid(W_s);
        end
    end
    W_sv = W_singles_vector{p};    
    
    if exist('W_s', 'var')
        assert (size(W_s,1) == size(W_p,1), '[%d] %d != %d', i, size(W_s,1), size(W_p,1))
    end
    assert (size(W_sv,1) == size(W_p,1))
    assert (size(W_sv,1) == docs(i).N)
    assert (size(W_sv,2) == 768)
    
    docs(i).W_pair = double(W_p);    
    docs(i).W_single_vec = double(W_sv);
        
    if exist('W_s', 'var')
        docs(i).W_single = double(W_s);
        
        num_sents = docs(i).N;
        W = ones(num_sents);
        for r = 1:num_sents
            for c = 1:num_sents
                if W_p(r,c) ~= 1 % r ~= c
                    W(r,c) = W_s(r) * W_p(r,c) * W_s(c);
                end
            end
        end
        docs(i).W_CNN = W;        
    end    
end
end

function res = sigmoid(x)
res = 1./(1+exp(-x));
end
