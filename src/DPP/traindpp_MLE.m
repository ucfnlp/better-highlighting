function theta = traindpp_MLE(traindocs, C, theta0, is_concat_feat, alpha)

cX = cell(size(traindocs));
cS = cX;
cY = cX;
for c = 1 : length(cX)
    if is_concat_feat
        cX{c} = cat(2, traindocs(c).W_single_vec, full(traindocs(c).F));
        cS{c} = traindocs(c).W_pair*(1-alpha) + traindocs(c).W_tfidf*alpha;
    else
        cX{c} = full(traindocs(c).F);
        cS{c} = traindocs(c).W_pair*(1-alpha) + traindocs(c).W_tfidf*alpha;
    end        
    
    N = traindocs(c).N;
    cY{c} = false(1,N);
    cY{c}(traindocs(c).Y) = true;
end

% minimize the hinge loss
options.Display = 'iter';
options.Method = 'lbfgs'; %'scg' 'qnewton'; 'lbfgs'; pnewton0;
options.optTol = 1e-8;
options.progTol = 1e-12;
options.MaxIter = 700;
options.MaxFunEvals = 700;

if is_concat_feat
    theta0_ = ones(size(traindocs(1).F, 2), 1) * 1e-1;
    theta0 = cat(1, theta0, theta0_);
else
    theta0 = ones(size(traindocs(1).F, 2), 1) * 1e-1;
end
funObj = @(arg)FG_QualityDiversity_MLE(arg, cX, cS, cY, C);

theta = minFunc(funObj, theta0, options);
