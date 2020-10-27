function [f, g] = FG_QualityDiversity_MLE(theta, cX, cS, cY, C)

FG = cat(2, cellfun(@(X,S,Y)MLEDPP(theta,X,S,Y), cX, cS, cY, ...
    'UniformOutput', false));
FG = cell2mat(FG);

if 1
    f = C*sum(FG(1,:)) + 0.5*(theta'*theta);
    g = C*sum(FG(2:end,:),2) + theta;
else
    f = sum(FG(1,:));
    g = sum(FG(2:end,:),2);
end