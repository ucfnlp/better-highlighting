function [docs, spec, means] = assign_features(docs,prpfn,which,use_CNN_sim,alpha)
% compute features for all docs, leaving them in docs(i).F
% - spec is a cell array of feature descriptions
% - means gives the mean of the feature (fraction of sentence with the
%   feature, if it's binary)

% config
if ~exist('prpfn','var')
    prpfn = '/zain/users/kulesza/dpp/data/mine2/shared/prp';
end

if ~exist('which','var')
    which = zeros(11,1);
    which(1:6) = 1; % const sim pos cost lexrank prp
end

% eigs uses some internal randomization, so let's do this for consistency
RandStream.setGlobalStream(RandStream('mt19937ar','seed',5546));

is_tfidf_update = false;
if ~isfield(docs, 'W_tfidf')
    is_tfidf_update = true;
end

for i = 1:length(docs)

    if is_tfidf_update
        docs(i).W_tfidf = docs(i).tfidf * docs(i).tfidf';
    end
    
    % precompute cosine distance matrix
    if use_CNN_sim==0        
        W = docs(i).W_tfidf;    % TF-IDF
    elseif use_CNN_sim==1
        W = docs(i).W_CNN;      % CNN        
    elseif use_CNN_sim==2        
        W = max(docs(i).W_CNN, docs(i).W_tfidf);                % element-wise max
    elseif use_CNN_sim==3        
        W = docs(i).W_CNN*(1-alpha) + docs(i).W_tfidf*alpha;    % element-wise avg
    end
        
    % 1. constant
    if which(1)
        F{1} = ones(docs(i).N,1);
        S{1} = {'constant'};
    end
    
    % 2. similarity to all other sentences
    if which(2)        
        sim = sum(W, 2) - 1;
        sim = sim / (docs(i).N - 1);
        
        step = 0.05:0.05:0.95;
        F{2} = [sim ... 
            binarize(sim,quantile(sim, step))];
        S{2} = [{'raw document similarity'} ...
            binspec('local-bin docsim', length(step))];        
    end
    
    % 3. sentence length (in bytes)
    if which(3)
        F{3} = binarize(docs(i).cost, [78 115.1 152 197]);
        S{3} = binspec('sentence length',4);
    end
    
    % 4. position in article
    if which(4)
        F{4} = [binarize(docs(i).pos, [1 2 3 4 5])];
        S{4} = binspec('sentence position',5);
    end
    
    % 5. lexrank
    if which(5)
        d = 0.1;
        M = bsxfun(@rdivide,W,sum(W,2)); % row normalize cosine dist
        M(isnan(M)) = 0; % in case of empty sentences
        M = (1-d)*M + d*(1/docs(i).N); % dampen
        [v,~] = eigs(M',1);
        v = v / sum(v); % normalize
        
        step = 0.05:0.05:0.95;        
        F{5} = [v, binarize(v,quantile(v,step))];
        S{5} = [{'raw lexrank'} ...
            binspec('local-bin lexrank',length(step))];        
    end
    
    % 6. personal pronouns
    if which(6)
        if ~exist('prp','var')
            prp = load(prpfn);
        end
        sent_len = size(docs(i).txt, 2);
        if sent_len >= 8
            word_len = 8;
        else
            word_len = sent_len;
        end
        indices = find(docs(i).txt(:,1:word_len)); % up to first 8 words
        [sents,~] = ind2sub(size(docs(i).txt),indices);
        firstwords = sparse(sents,docs(i).txt(indices),1, ...
            docs(i).N,size(docs(i).words,2));
        F{6} = any(firstwords(:,prp),2);
        S{6} = {'personal pronouns'};
    end
    
    % 7. centroid centrality
    if which(7)
        center = sum(docs(i).tfidf);
        center1 = center; % top ~5 words
        center1(find(center < quantile(center,0.9999))) = 0;
        center2 = center; % top ~50 words
        center2(find(center < quantile(center,0.999))) = 0;
        center3 = center; % top ~500 words
        center3(find(center < quantile(center,0.99))) = 0;
        F1 = docs(i).words * center1';
        F1 = binarize(F1,quantile(F1,0.2:0.2:0.8));
        F2 = docs(i).words * center2';
        F2 = binarize(F2,quantile(F2,0.2:0.2:0.8));
        F3 = docs(i).words * center3';
        F3 = binarize(F3,quantile(F3,0.2:0.2:0.8));
        F{7} = [F1 F2 F3];
        S{7} = binspec('centroid centrality',14);
    end
    
    % 8. degree centrality
    if which(8)
        thresh1 = quantile(W(:),0.5);
        thresh2 = quantile(W(:),0.9);
        thresh3 = quantile(W(:),0.99);
        F1 = sum(W>thresh1,2);
        F1 = binarize(F1,quantile(F1,0.2:0.2:0.8));
        F2 = sum(W>thresh2,2);
        F2 = binarize(F2,quantile(F2,0.2:0.2:0.8));
        F3 = sum(W>thresh3,2);
        F3 = binarize(F3,quantile(F3,0.2:0.2:0.8));
        F{8} = [F1 F2 F3];
        S{8} = binspec('degree centrality',14);
    end
    
    % 9. signature tokens/bigrams
    if which(9)
        sigs = find(center > quantile(center,0.999)); % top 0.1% of words
        F{9} = log(sum(docs(i).words(:,sigs),2)+1);
        S{9} = {'signature tokens'};
    end
    
    % 10. threshold lexrank
    if which(10)
        d = 0.1;
        threshold = quantile(W(:),0.8);
        M = W > threshold;
        M = bsxfun(@rdivide,M,sum(M,2)); % row normalize
        M(isnan(M)) = 0; % in case of empty sentences
        M = (1-d)*M + d*(1/docs(i).N); % dampen
        [v,~] = eigs(M',1);
        v = v / sum(v); % normalize
        F{10} = [v ... % unbinnned, in [0,1]
            binarize(v,[0.0018 0.0029 0.0043 0.0063]) ... % pre-binned
            binarize(v,quantile(v,0.2:0.2:0.8))]; % locally binned
        S{10} = [{'raw lexrank'} ...
            binspec('global-bin lexrank',4) ...
            binspec('local-bin lexrank',4)];
    end
    
    % 11. Similarity to the top 3 sentences
    if which(11)
        ix1 = (docs(i).pos==1);
        ix2 = (docs(i).pos==2);
        ix3 = (docs(i).pos==3);
        ix = (ix1 | ix2 | ix3);
        C = docs(i).tfidf * docs(i).tfidf(ix,:)';
        F{11} = [mean(C,2) ... % mean sim
            median(C,2) ... % median
            max(C,[],2) ... % max
            min(C,[],2) ... % min
            std(C,0,2)...
            %quantile(C,[0.25 0.75 0.9],2) ... % 1/4, 3/4, and 0.9 quantile
            %binarize(mean(C,2),[0.0078    0.0236    0.0490    0.0864]) ...
            %binarize(max(C,[],2),[0.0822    0.1608    0.2452    0.3870]) ...
            %binarize(std(C,0,2),[0.0204    0.0426    0.0662    0.1048]) ...
            ]; % locally binned
        S{11} = [{'mean sim to top 3 sentences'} ...
            {'median sim to top 3 sentences'} ...
            {'max sim to top 3 sentences'} ...
            {'min sim to top 3 sentences'} ...
            {'std sim to top 3 sentences'}...
            %{'1/4 quantile of sim to top 3'} ...
            %{'3/4 quantile of sim to top 3'} ...
            %{'0.9 quantile of sim to top 3'} ...
            %binspec('bin mean sim',4), ...
            %binspec('bin max sim',4), ...
            %binspec('bin std sim',4)
            ];
    end
    
    % put it all together
    docs(i).F = [F{find(which)}];
    if ~exist('spec','var')
        spec = [S{find(which)}];
    end
end

% compute counts/means
means = mean(vertcat(docs.F));
